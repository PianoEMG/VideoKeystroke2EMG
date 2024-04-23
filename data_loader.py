import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle as pkl
import librosa
from tqdm import tqdm
import torch.nn.functional as F
from transformers import Wav2Vec2Processor, Wav2Vec2Model

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
            with open(f'./dataset/emg_data/{file}.pkl', 'rb') as f:
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

# # 创建数据加载器
# def create_dataloader(batch_size, shuffle=True):
#     dataset = Audio2EMGDataset('./data/')
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#     return dataloader

# # 示例用法
# # data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# batch_size = 8
# dataloader = create_dataloader(batch_size)
# print(len(dataloader))
# # 遍历数据加载器
# for batch in dataloader:
#     print(batch[0].shape, batch[1].shape)