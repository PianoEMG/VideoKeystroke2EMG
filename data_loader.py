import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle as pkl
import librosa
from tqdm import tqdm
import torch.nn.functional as F
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import matplotlib.pyplot as plt
import mlflow

def linear_interpolation(features, input_fps, output_fps, output_len=None):
    features = features.transpose(1, 2)
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    output_features = F.interpolate(features,size=output_len,align_corners=True,mode='linear')
    return output_features.transpose(1, 2)

# 创建自定义的数据集类
class Audio2EMGDataset(Dataset):
    def __init__(self, file_path, win_len=1024):
        self.data = []
        self.win_len = win_len
        self.file_list = os.listdir(file_path)
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_encoder = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
        print(f'Cropping data into windows length={self.win_len}')
        for file in tqdm(self.file_list):
            if file == 'processed':
                continue
            with open(f'./dataset/emg_data_filtered/{file}.pkl', 'rb') as f:
                emg_data = pkl.load(f)
            emg_data = torch.tensor(emg_data).float()
            audio_path = f'./dataset/audio_data/{file}.wav'
            audio_input, _ = librosa.load(audio_path, sr=2000)
            input_values = self.processor(audio_input, sampling_rate=16000, return_tensors="pt").input_values
            # print(audio_input.shape, input_values.shape)
            input_values = input_values[:,:emg_data.shape[0]]
            audio_features = self.audio_encoder(input_values).last_hidden_state
            audio_features = linear_interpolation(audio_features, 50, 30, output_len=input_values.shape[1])
            audio_features = audio_features.squeeze(0)

            frame_num = emg_data.shape[0]
            num_group = frame_num // self.win_len
            emg_data = emg_data[:num_group * self.win_len]
            audio_features = audio_features[:num_group * self.win_len]

            split_emg = np.split(emg_data, num_group, axis=0)
            split_audio = np.split(audio_features, num_group, axis=0)

            # self.data.append((split_audio[i], split_emg[i]) for i in range(num_group))
            for i in range(num_group):
                self.data.append((split_audio[i], split_emg[i]))
            # print(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
def log_plot(batch_gt, batch_output, step, save_path, writer):
    batch_size = len(batch_gt)  # 获取批量大小
    fig, axes = plt.subplots(nrows=batch_size, ncols=2, figsize=(15, 2 * batch_size))
    
    for i, (emg_gt, emg_output) in enumerate(zip(batch_gt, batch_output)):
        if isinstance(emg_gt, torch.Tensor):
            emg_gt = emg_gt.cpu().numpy()
        if isinstance(emg_output, torch.Tensor):
            emg_output = emg_output.detach().cpu().numpy()
        
        # 绘制实际EMG信号
        if batch_size == 1:
            ax_gt = axes[0]
            ax_output = axes[1]
        else:
            ax_gt = axes[i, 0]
            ax_output = axes[i, 1]
        
        ax_gt.plot(emg_gt)
        ax_gt.set_title(f'EMG Ground Truth for Sample {i+1}')
        ax_gt.set_xlabel('Time')
        ax_gt.set_ylabel('Amplitude')
        ax_gt.grid(True)

        # 绘制输出EMG信号
        ax_output.plot(emg_output)
        ax_output.set_title(f'EMG Output for Sample {i+1}')
        ax_output.set_xlabel('Time')
        ax_output.set_ylabel('Amplitude')
        ax_output.grid(True)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.tight_layout()
    plt.savefig(f'{save_path}/batch_{step}.png')  # 保存图表为PNG文件
    plt.close(fig)
    img = plt.imread(f'{save_path}/batch_{step}.png')
    mlflow.log_artifact(f'{save_path}/batch_{step}.png')
    writer.add_image('EMG_signals', img.transpose((2, 0, 1)), global_step=step)