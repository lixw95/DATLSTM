import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import numpy
import torch
import os
from sklearn.metrics import r2_score, mean_absolute_error
from self_attn import SelfAttention
from tcn import TemporalConvNet
class Encoder(nn.Module):
    def __init__(self, T, input_size, encoder_num_hidden, parallel=False):
        super(Encoder, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.input_size = 15
        self.parallel = parallel
        self.T = T
        self.K = input_size
        self.weight = nn.Parameter()
        self.bias = nn.Parameter()
        self.encoder_lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.encoder_num_hidden, num_layers=1)
        self.encoder_attn = nn.Linear(in_features=self.encoder_num_hidden, out_features=1)
        self.encoder_attn_v = nn.Linear(in_features=2 * self.encoder_num_hidden + 1, out_features=1)
        self.attn = SelfAttention(num_attention_heads=1, input_size=self.input_size, hidden_size=128,
                                  hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.5)
    def forward(self, X):
        X_tilde = Variable(X.data.new(X.size(0), self.T - 1, self.input_size).zero_())
        X_encoded = Variable(X.data.new(X.size(0),self.T -1, self.encoder_num_hidden).zero_())
        h_n = self._init_states(X)
        s_n = self._init_states(X)
        for t in range(self.T - 1):
            alph = self.attn(X)
            x = self.encoder_attn(alph.view(-1, self.encoder_num_hidden))
            alpha = F.softmax(x.view(-1, self.input_size), dim=1)
            x_tilde = torch.mul(alpha, X[:, t, :])

            self.encoder_lstm.flatten_parameters()
            _, final_state = self.encoder_lstm(x_tilde.unsqueeze(0), (h_n, s_n))
            h_n = final_state[0]
            s_n = final_state[1]
            X_tilde[:, t, :] = x_tilde
            X_encoded[:, t, :] = h_n
        return X_tilde, X_encoded
    def _init_states(self, X):
        return Variable(X.data.new(1, X.size(0), self.encoder_num_hidden).zero_())


class Decoder(nn.Module):
    def __init__(self, T, decoder_num_hidden, encoder_num_hidden):
        super(Decoder, self).__init__()
        self.decoder_num_hidden = decoder_num_hidden
        self.encoder_num_hidden = encoder_num_hidden
        self.T = T
        self.input_size = 15
        self.h = torch.nn.Parameter(data=torch.empty(self.T - 1, decoder_num_hidden).uniform_(0, 1), requires_grad=True)
        self.c = torch.nn.Parameter(data=torch.empty(self.T - 1, decoder_num_hidden).uniform_(0, 1), requires_grad=True)
        self.attn_layer = nn.Sequential(nn.Linear(self.decoder_num_hidden * 2 + self.input_size, decoder_num_hidden),
                                        nn.Linear(encoder_num_hidden, 1))

        self.lstm_layer = nn.LSTM(input_size=1, hidden_size=self.decoder_num_hidden, num_layers=1)
        self.gru_layer = nn.GRU(input_size=1, hidden_size=self.decoder_num_hidden * 2, num_layers=1)
        self.fc = nn.Linear(self.input_size + 1, 1)
        self.fc_final = nn.Linear(self.decoder_num_hidden + self.input_size, 1)
        self.fc.weight.data.normal_()
        self.num_chans = [encoder_num_hidden] * (4 - 1) + [self.input_size]
        self.tcn = TemporalConvNet(self.encoder_num_hidden,  self.num_chans, kernel_size=3, dropout=0.45)
        self.conv1 = nn.Conv1d(128, out_channels=128, kernel_size=5, stride=1, padding=0, dilation=1, padding_mode='zeros')
    def forward(self, X_encoded,y_prev):
        encoder_cn = self.tcn(X_encoded.transpose(1, 2))
        encod = encoder_cn.transpose(1, 2)
        d_n = self._init_states(encod)
        c_n = self._init_states(encod)
        for t in range(self.T - 1):
            x = torch.cat((d_n.repeat(self.T - 1, 1, 1).permute(1, 0, 2), c_n.repeat(self.T - 1, 1, 1).permute(1, 0, 2),encod), dim=2)
            beta = F.softmax(self.attn_layer(x.view(-1, self.decoder_num_hidden * 2 + self.input_size)).view(-1, self.T - 1), dim=1)
            context = torch.bmm(beta.unsqueeze(1), encod)[:, 0, :]
            if t < self.T - 1:
                y = torch.cat((context, y_prev[:, t].unsqueeze(1)), dim=1)
                y_tilde = self.fc(y)
                self.lstm_layer.flatten_parameters()
                _, final_states = self.lstm_layer(y_tilde.unsqueeze(0), (d_n, c_n))
                d_n = final_states[0]
                c_n = final_states[1]
        y_pred = self.fc_final(torch.cat((d_n[0], context), dim=1))
        return y_pred
    def _init_states(self, X):
        return Variable(X.data.new(1, X.size(0), self.decoder_num_hidden).zero_())

class DA_RNN(nn.Module):
    def __init__(self, T, encoder_num_hidden, decoder_num_hidden, batch_size, learning_rate, epochs, parallel=False):
        super(DA_RNN, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.decoder_num_hidden = decoder_num_hidden
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.parallel = parallel
        self.shuffle = False
        self.epochs = epochs
        self.T = T
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("==> Use accelerator: ", self.device)
        self.input_size = 15
        self.Encoder = Encoder(input_size=self.input_size, encoder_num_hidden=encoder_num_hidden, T=T).to(self.device)
        self.Decoder = Decoder(encoder_num_hidden=encoder_num_hidden, decoder_num_hidden=decoder_num_hidden, T=T).to(
            self.device)
        self.criterion = nn.MSELoss()

        if self.parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)
        self.encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, self.Encoder.parameters()), lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, self.Decoder.parameters()), lr=self.learning_rate)
    def train(self):
        train_num = np.array(range(1, 221))
        n_iter = 0
        self.train_timesteps = 51
        for epoch in range(self.epochs):
            idx = 0
            ref_idx = np.array(range(self.train_timesteps - self.T))
            self.error_sum = np.zeros(self.train_timesteps - (self.T - 1))
            self.iter_losses = np.zeros(self.train_timesteps - (self.T - 1))
            # indices = ref_idx[idx:(idx + self.T - 1)]
            indices = np.array(range(self.train_timesteps - (self.T - 1)))
            x = np.zeros((len(train_num), self.T - 1, self.input_size))
            y_prev = np.zeros((len(train_num), self.T - 1))
            y_gt = np.zeros((len(train_num), 1))
            for bs in range(len(indices)):
                for i in range(len(train_num)):
                    num = train_num[i]
                    self.X, self.y = self.load_train_dataset(num)
                    self.y = self.y.reshape(-1)
                    if indices[bs] + self.T - 1 < self.y.shape[0]:
                        x[i, :, :] = self.X[indices[bs]:(indices[bs] + self.T - 1), :]
                        y_prev[i, :] = self.y[indices[bs]:(indices[bs] + self.T - 1)]
                        y_gt[i, :] = self.y[indices[bs] + self.T - 1]
                    else:
                        i += 1
                loss, error = self.train_forward(x, y_prev, y_gt)
                self.iter_losses[bs] = loss
                self.error_sum[bs] = error
                idx += self.batch_size
                n_iter += 1
                if n_iter % 10000 == 0 and n_iter != 0:
                    for param_group in self.encoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9
                    for param_group in self.decoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9
            epochs_loss = np.mean(self.iter_losses)
            error_epoch = np.mean(self.error_sum)
            print('Epochs: ', epoch, 'iteration ', n_iter, 'loss: ', epochs_loss, 'error: ', error_epoch)

    def train_forward(self, X, y_prev, y_gt):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        train_num = np.array(range(1, 221))
        error = np.zeros(len(train_num))
        input_weighted, input_encoded = self.Encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor).to(self.device)))
        y_pred = self.Decoder(input_encoded, Variable(torch.from_numpy(y_prev).type(torch.FloatTensor).to(self.device)))
        y_true = Variable(torch.from_numpy(y_gt).type(torch.FloatTensor).to(self.device))
        y_true = y_true.view(-1, 1)
        for j in range(len(train_num)):
            if y_true[j] == 0:
                er = y_pred[j]
            else:
                er = abs(y_true[j] - y_pred[j])
            error[j] = er
        error_mean = np.mean(error)
        loss = self.criterion(y_pred, y_true)
        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        return loss.item(), error_mean

    def save_checkpoint(self, epoch, model, n_iter, loss):
        state = {'epoch': epoch, 'loss': loss, 'model': model, 'iter': n_iter}
        filename = './checkpoint_' + str(epoch) + '_' + str(loss) + '.pth'
        torch.save(state, filename)

    def load_train_dataset(self, year):
        #main_path = r"/home/lxw/1-th_code/MALSTM/DA-RNN-master/train_set220_knn"
        main_path = r"/home/lxw/DAT-LSTM/train_set220_knn"
        file_name = '{}.csv'.format(year)
        file_path = os.path.join(main_path, file_name)
        df = pd.read_csv(file_path)
        data = df.iloc[:, 1:].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)
        x1 = data[:, 0:11]
        x2 = data[:, 12:]
        x = np.hstack((x1, x2))
        y = data[:, 11]
        return numpy.array(x, dtype=np.float32), numpy.array(y, dtype=np.float32)

    def load_test_dataset(self, year):
        #main_path = r"/home/lxw/1-th_code/MALSTM/DA-RNN-master/test_dataset"
        main_path = r"/home/lxw/DAT-LSTM/test_dataset"
        file_name = '{}.csv'.format(year)
        file_path = os.path.join(main_path, file_name)
        df = pd.read_csv(file_path)
        data = df.iloc[:, 1:].values
        x1 = data[:, 0:11]
        x2 = data[:, 12:]
        x = np.hstack((x1, x2))
        y = data[:, 11]
        y = np.array(y).reshape(-1, 1)
        return numpy.array(x, dtype=np.float32), numpy.array(y, dtype=np.float32)

    def test(self, on_train=False):
        test_num = np.array(range(82, 221))
        y_gt = np.zeros(len(test_num))
        y_pre = np.zeros(len(test_num))
        y_gt_inve = np.zeros(len(test_num))
        y_pre_inve = np.zeros(len(test_num))
        rmse_sum = np.zeros(len(test_num))
        error_sum = np.zeros(len(test_num))
        for k in range(len(test_num)):
            num = test_num[k]
            if num != 83 and num != 99 and num != 117 and num != 115 and num != 100 and num != 106 and num != 109 and num != 110 and num != 84 and num != 107 and num != 156 and num != 173 and num != 208 and num != 215:
                self.X, self.y = self.load_test_dataset(num)
                scaler = MinMaxScaler(feature_range=(0, 1))
                self.X = scaler.fit_transform(self.X)
                self.y = scaler.fit_transform(self.y)
                self.y = self.y.reshape(-1)
                self.train_timesteps = int(self.X.shape[0] - 1)
                y_pred = np.zeros(self.X.shape[0] - self.train_timesteps)
                i = 0
                while i < len(y_pred):
                    batch_idx = np.array(range(len(y_pred)))[i: (i + self.batch_size)]
                    X = np.zeros((len(batch_idx), self.T - 1, self.X.shape[1]))
                    y_history = np.zeros((len(batch_idx), self.T - 1))
                    for j in range(len(batch_idx)):
                        X[j, :, :] = self.X[self.train_timesteps - (self.T - 1):self.train_timesteps, :]
                        y_history[j, :] = self.y[self.train_timesteps - (self.T - 1):self.train_timesteps]

                    y_history = Variable(torch.from_numpy(y_history).type(torch.FloatTensor).to(self.device))
                    _, input_encoded = self.Encoder(
                        Variable(torch.from_numpy(X).type(torch.FloatTensor).to(self.device)))
                    y_pred[i:(i + self.batch_size)] = self.Decoder(input_encoded, y_history).cpu().data.numpy()[:, 0]
                    i += self.batch_size
                y_true = self.y[self.train_timesteps:]
                RMSE = np.square(y_pred - y_true)
                rmse_sum[k] = RMSE
                y_gt[k] = y_true
                y_pre[k] = y_pred
                y_pred = y_pred.reshape(-1, 1)
                y_true = y_true.reshape(-1, 1)
                y_pred = scaler.inverse_transform(y_pred)
                y_true = scaler.inverse_transform(y_true)
                y_pred = y_pred.reshape(-1)
                y_true = y_true.reshape(-1)
                error = abs((y_true - y_pred) / y_true)
                error_sum[k] = error
                y_gt_inve[k] = y_true
                y_pre_inve[k] = y_pred
                # print('y_true and y_pred:', y_true, y_pred, error)
        rmse_arr = np.sqrt(np.mean(rmse_sum))
        print('RMSE: %.4f' % rmse_arr)
        error_s = np.mean(error_sum)
        print('average error: %.4f' % error_s)
        mae = mean_absolute_error(y_gt, y_pre)
        print('mae: %.4f' % mae)
        print('r2_score: %.4f' % r2_score(y_gt_inve, y_pre_inve))

        fig3 = plt.figure()
        plt.plot(y_pre_inve, label='Predicted')
        plt.plot(y_gt_inve, label="True")
        plt.legend(loc='upper left')
        plt.savefig('/home/lxw/DAT-LSTM/img/result1.png')
        plt.close(fig3)



if __name__ == '__main__':
    print("==> Load dataset ...")
    model = DA_RNN(12, 128, 128, 52, 0.001, 1200)
    model.train()
    #model5
    #torch.save(model, '/home/lxw/TCN -attn/model.pth') #aver_error 0.09003 result.png
    torch.save(model, '/home/lxw/DAT-LSTM/model_new.pth')
    #model = torch.load('/home/lxw/TCN -attn/model.pth')
    model.test()