from model.a2emg import Audio2EMG
from data_loader import Audio2EMGDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
import mlflow
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--feature_dim", type=int, default=768)
    parser.add_argument("--period", type=int, default=256)
    parser.add_argument("--device", type=str, default=device)
    args = parser.parse_args()

    dataset = Audio2EMGDataset('./data/')
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    criterion = nn.MSELoss().to(device)
    mlflow.start_run()
    # mlflow.set_experiment('Audio2EMG')
    model = Audio2EMG(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in tqdm(range(args.num_epochs)):
        for i, (audio, emg) in enumerate(data_loader):
            if audio.shape[0] != args.batch_size:
                continue
            audio = audio.to(device)
            emg = emg.to(device)

            loss, output = model(audio, emg, criterion)
            # loss = criterion(output, emg)
            # print(loss)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{args.num_epochs}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item()}')
                mlflow.log_metric("loss", loss.item())
                writer.add_scalar('Loss/train', loss.item(), epoch * len(data_loader) + i)

    mlflow.end_run()

if __name__ == '__main__':
    main()