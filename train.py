from model.a2emg import Audio2EMG
from data_loader import Audio2EMGDataset, log_plot
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
import mlflow
from torch.utils.tensorboard import SummaryWriter
import datetime


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()
current_time = datetime.datetime.now()
current_time = current_time.strftime("%Y%m%d-%H%M%S")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--feature_dim", type=int, default=768)
    parser.add_argument("--period", type=int, default=256)
    parser.add_argument("--device", type=str, default=device)
    parser.add_argument("--model_path", type=str, default='./model/')
    parser.add_argument("--log_path", type=str, default=f'./logs/{current_time}')
    args = parser.parse_args()

    dataset = Audio2EMGDataset('./data/')
    total_size = len(dataset)
    train_size = int(0.9 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    criterion = nn.MSELoss().to(device)
    # mlflow.start_run()
    # mlflow.set_experiment('piano')
    model = Audio2EMG(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in tqdm(range(args.num_epochs)):
        for i, (audio, emg) in enumerate(train_loader):
            # continue
            if audio.shape[0] != args.batch_size:
                continue
            audio = audio.to(device)
            emg = emg.to(device)

            loss, output = model(audio, emg, criterion)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{args.num_epochs}], Step [{i+1}/{len(train_loader)}], Train Loss: {loss.item()}')
                mlflow.log_metric("train_loss", loss.item())
                writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)
                log_plot(emg, output, epoch * len(train_loader) + i + 1, args.log_path, writer)
                
            # if (i+1) % 100 == 0:
                # log_plot(emg, output, epoch * len(train_loader) + i, args.log_path, writer)


        for i, (audio, emg) in enumerate(test_loader):
            if audio.shape[0] != args.batch_size:
                continue
            audio = audio.to(device)
            emg = emg.to(device)
            loss, output = model.inference(audio, emg, criterion)

            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{args.num_epochs}], Step [{i+1}/{len(test_loader)}], Test Loss: {loss.item()}')
                mlflow.log_metric("test_loss", loss.item())
                writer.add_scalar('Loss/test', loss.item(), epoch * len(test_loader) + i)
                log_plot(emg, output, epoch * len(test_loader) + i + 1, args.log_path, writer)
            # if (i+1) % 100 == 0:
                # log_plot(emg, output, epoch * len(test_loader) + i, args.log_path, writer)
        
        if epoch % 25 == 0:
            torch.save(model.state_dict(), f'./save_model/model_{epoch+1}.pth')
    # mlflow.end_run()

if __name__ == '__main__':
    mlflow.start_run()
    main()
    mlflow.end_run()