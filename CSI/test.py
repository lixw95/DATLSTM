import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import argparse
from util import *
from model import *
import torch.optim as optim
from sklearn.metrics import r2_score
parser = argparse.ArgumentParser(description='financial time series data Modeling')
parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='batch size (default: 16)')
parser.add_argument('--cuda', action='store_false', help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.45, help='dropout applied to layers (default: 0.45)')
parser.add_argument('--T', type=int, default=11, help='time step')
parser.add_argument('--emb_dropout', type=float, default=0.25, help='dropout applied to the embedded layer (default: 0.25)')
parser.add_argument('--clip', type=float, default=0.35, help='gradient clip, -1 means no clip (default: 0.35)')
parser.add_argument('--epochs', type=int, default=300, help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=3, help='kernel size (default: 3)')
parser.add_argument('--data', type=str, default='./data/penn', help='location of the data corpus (default: ./data/penn)')
parser.add_argument('--input_size', type=int, default=16, help='input size of data for TCN')
parser.add_argument('--levels', type=int, default=4, help='# of levels (default: 4)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='report interval (default: 100)')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate (default: 4)')
parser.add_argument('--nhid', type=int, default=600, help='number of hidden units per layer (default: 600)')
parser.add_argument('--seed', type=int, default=1111, help='random seed (default: 1111)')
parser.add_argument('--tied', action='store_false', help='tie the encoder-decoder weights (default: True)')
parser.add_argument('--optim', type=str, default='SGD', help='optimizer type (default: SGD)')
parser.add_argument('--validseqlen', type=int, default=40, help='valid sequence length (default: 40)')
parser.add_argument('--seq_len', type=int, default=80,  help='total sequence length, including effective history (default: 80)')
parser.add_argument('--corpus', action='store_true', help='force re-make the corpus (default: False)')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
args = parser.parse_args()
n_words = 1
d_model = 128
#num_chans = [args.nhid] * (args.levels - 1) + [args.input_size]
num_chans = [args.nhid] * (args.levels - 1) + [d_model]
dropout = args.dropout
emb_dropout = args.emb_dropout
k_size = args.ksize
tied = args.tied
lr = args.lr
criterion = nn.MSELoss()
TCN_inputs = 128
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = TCN(128, n_words, num_chans, d_model,  dropout=dropout, emb_dropout=emb_dropout, kernel_size=k_size, tied_weights=tied)
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


def train():
    train_num = np.array(range(1, 221))
    n_iter = 0
    train_timesteps = 51
    batch_size = 52
    # model_optim = self._select_optimizer()
    for epoch in range(args.epochs):
        idx = 0
        error_sum = np.zeros(train_timesteps - (args.T - 1))
        iter_losses = np.zeros(train_timesteps - (args.T - 1))
        # indices = ref_idx[idx:(idx + self.T - 1)]
        indices = np.array(range(train_timesteps - (args.T - 1)))
        x = np.zeros((len(train_num), args.T - 1, args.input_size))
        y_prev = np.zeros((len(train_num), args.T - 1))
        y_gt = np.zeros((len(train_num), 1))
        for bs in range(len(indices)):
            #print('{}/41'.format(bs))
            for i in range(len(train_num)):
                num = train_num[i]
                X, y = load_train_dataset(num)
                y = y.reshape(-1)
                if indices[bs] + args.T - 1 < y.shape[0]:
                    x[i, :, :] = X[indices[bs]:(indices[bs] + args.T - 1), :]
                    y_gt[i, :] = y[indices[bs] + args.T - 1]
                else:
                    i += 1
            loss, error = train_forward(x, y_gt)
            iter_losses[bs] = loss
            error_sum[bs] = error
            idx += batch_size
            n_iter += 1
            # model_optim.zero_grad()
            # adjust_learning_rate(model_optim, epoch + 1, self.args)
        epochs_loss = np.mean(iter_losses)
        error_epoch = np.mean(error_sum)
        print('Epochs: ', epoch, 'iteration ', n_iter, 'loss: ', epochs_loss, 'error: ', error_epoch)

def train_forward(X, y_gt):
    optimizer.zero_grad()
    train_num = np.array(range(1, 221))
    error = np.zeros(len(train_num))
    X = Variable(torch.from_numpy(X).type(torch.FloatTensor).to(device))
    out = model(X)
    #print('out',out)
    y_pred = out[:, -1, -1]

    #print('y_pred', y_pred)
    y_pred = y_pred.unsqueeze(1)
    y_true = Variable(torch.from_numpy(y_gt).type(torch.FloatTensor).to(device))
    y_true = y_true.view(-1, 1)
    for j in range(len(train_num)):
        if y_true[j] == 0:
            er = y_pred[j]
        else:
            er = abs(y_true[j] - y_pred[j])
        error[j] = er
    error_mean = np.mean(error)
    loss = criterion(y_pred, y_true)
    loss.backward()
    optimizer.step()
    return loss.item(), error_mean


def test():
    test_num = np.array(range(82, 221))
    batch_size = 52
    y_gt = np.zeros(len(test_num))
    y_pre = np.zeros(len(test_num))
    rmse_sum = np.zeros(len(test_num))
    y_true_all = np.zeros(len(test_num))
    y_pred_all = np.zeros(len(test_num))
    error_sum = np.zeros(len(test_num))
    n = 0
    for k in range(len(test_num)):
        num = test_num[k]
        if num != 83 and num != 99 and num != 117 and num != 115 and num != 100 and num != 106 and num != 109 and num != 110 and num != 84 and num != 107 and num != 156 and num != 173 and num != 208 and num != 215:
            test_X, y = load_test_dataset(num)
            scaler = MinMaxScaler(feature_range=(0, 1))
            test_X = scaler.fit_transform(test_X)
            y = scaler.fit_transform(y)
            y = y.reshape(-1)
            train_timesteps = int(test_X.shape[0] - 1)
            y_pred = np.zeros(test_X.shape[0] - train_timesteps)
            i = 0
            while i < len(y_pred):
                batch_idx = np.array(range(len(y_pred)))[i: (i + batch_size)]
                X = np.zeros((len(batch_idx), args.T - 1, test_X.shape[1]))
                y_history = np.zeros((len(batch_idx), args.T - 1))
                for j in range(len(batch_idx)):
                    X[j, :, :] = test_X[train_timesteps - (args.T - 1):train_timesteps, :]
                X = Variable(torch.from_numpy(X).type(torch.FloatTensor).to(device))
                y_pred = model(X)
                i += batch_size
            #print('pred', y_pred.shape)
            y_pred = y_pred[:, -1]
            #print('y_pred', y_pred.shape)
            # y_pred = y_pred[:1]
            y_pred = y_pred.detach().numpy()
            y_true = y[train_timesteps:]
            RMSE = np.square(y_pred - y_true)
            rmse_sum[k] = RMSE
            y_pred = y_pred.reshape(-1, 1)
            y_true = y_true.reshape(-1, 1)
            y_pred = scaler.inverse_transform(y_pred)
            y_true = scaler.inverse_transform(y_true)
            y_pred = y_pred.reshape(-1)
            y_true = y_true.reshape(-1)
            error = abs((y_true - y_pred) / y_true)
            error_sum[k] = error
            y_gt[k] = y_true
            y_pre[k] = y_pred
            print('y_true and y_pred:', y_true, y_pred, error)


    rmse_arr = np.sqrt(np.mean(rmse_sum))
    print('RMSE', rmse_arr)
    error_s = np.mean(error_sum)
    print('average error：', error_s)
    print('r2_score: %.2f' % r2_score(y_gt, y_pre))


    fig3 = plt.figure()
    plt.plot(y_pre, label='Predicted')
    plt.plot(y_gt, label="True")
    plt.legend(loc='upper left')
    plt.savefig('/home/lxw/TCN -attn/img/result.png')
    plt.close(fig3)


if __name__ == '__main__':
    print("==> Load dataset ...")
    train()
    test()
