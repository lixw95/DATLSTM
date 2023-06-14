import torch
import argparse
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn import preprocessing
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils import *
from NASQ_model import *
from sklearn.preprocessing import MinMaxScaler

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch implementation of paper 'A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction'")
    parser.add_argument('--dataroot', type=str, default="/home/lxw/TCN -attn/NASQ_dataset/nasdaq100_padding.csv", help='path to dataset')
    parser.add_argument('--batchsize', type=int, default=128, help='input batch size [128]')
    parser.add_argument('--nhidden_encoder', type=int, default=128, help='size of hidden states for the encoder m [64, 128]')
    parser.add_argument('--nhidden_decoder', type=int, default=128, help='size of hidden states for the decoder p [64, 128]')
    parser.add_argument('--ntimestep', type=int, default=9, help='the number of time steps in the window T [10]')
    parser.add_argument('--test_timestep', type=int, default=500, help='the timestep of testing')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train [10, 200, 500]')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate [0.001] reduced by 0.1 after each 10000 iterations')
    args = parser.parse_args()
    return args
def main():
    """Main pipeline of DA-RNN."""
    args = parse_args()

    # Read dataset
    print("==> Load dataset ...")
    X, y = read_data(args.dataroot, debug=False)
    print(X.shape)
    # minmaxscaler
    '''
    y = y.reshape(y.shape[0], 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y)
    y = y.reshape(y.shape[0])
    '''
    print("==> Initialize model ...")
    model = Project(X, y, args.ntimestep, args.nhidden_encoder, args.nhidden_decoder, args.batchsize, args.lr, args.epochs)
    print("==> Start training ...")
    #model.train()
    #save model
    #torch.save(model, '/home/lxw/TCN -attn/NASQ_dataset/model_200_128m.pth')
    #torch.save(model, '/home/lxw/TCN -attn/NASQ_dataset/model_40.pth')
    model = torch.load('/home/lxw/TCN -attn/NASQ_dataset/model_100.pth')
    #load model
    y_pred = model.test()
    y_pred = y_pred[:args.test_timestep]
    y_true = model.y[model.train_timesteps:model.train_timesteps + args.test_timestep]
    # inverse minmaxscaler
    #y_pred, y_true = y_pred.reshape(y_pred.shape[0], 1), y_true.reshape(y_true.shape[0], 1)
    #y_pred = scaler.inverse_transform(y_pred)
    #y_true = scaler.inverse_transform(y_true)
    #y_pred, y_true = y_pred.reshape(y_pred.shape[0]), y_true.reshape(y_true.shape[0])
    #picture
    fig1 = plt.figure()
    plt.semilogy(range(len(model.iter_losses)), model.iter_losses)
    plt.savefig("/home/lxw/TCN -attn/NASQ_dataset/1.png")
    plt.close(fig1)
    fig2 = plt.figure()
    plt.semilogy(range(len(model.epoch_losses)), model.epoch_losses)
    plt.savefig("/home/lxw/TCN -attn/NASQ_dataset/2.png")
    plt.close(fig2)
    fig3 = plt.figure()
    plt.plot(y_pred, label='Predicted')
    plt.plot(y_true, label="True")
    plt.legend(loc='upper left')
    plt.savefig("/home/lxw/TCN -attn/NASQ_dataset/4.png")
    plt.close(fig3)
    print('Finished Training')
    mse = mean_squared_error(y_true, y_pred)
    print('mse:', mse)
    rmse = np.sqrt(mse)
    print('rmse:', rmse)
    mae = mean_absolute_error(y_true, y_pred)
    print('mae:', mae)
    print('r2_score: %.2f' % r2_score(y_true, y_pred))

if __name__ == '__main__':
    main()
