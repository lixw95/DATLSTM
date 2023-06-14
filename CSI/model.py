import torch
from torch.nn import LayerNorm
from causal_con_layer import *
from self_attn import *
from torch.nn.utils import weight_norm
from torch import nn, tensor
from torch.autograd import Variable
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):

        return self.network(x)

class Time_Embeddings(nn.Module):
    def __init__(self, d_model, inputsize, dropout, n_RNN_layers=1):
        super(Time_Embeddings, self).__init__()
        self.hidden_dimensions = d_model
        self.n_hidden_dimensions = n_RNN_layers
        self.RNN = nn.RNN(
            input_size=inputsize,
            hidden_size=d_model,
            num_layers=n_RNN_layers,
            batch_first=True
            )
        self.dropout = nn.Dropout(dropout)
        #self.h_in = nn.Parameter(torch.zeros(self.n_hidden_dimensions, 220, d_model), requires_grad=True)
        #self.s_in = nn.Parameter(torch.zeros(self.n_hidden_dimensions, 220, d_model), requires_grad=True)

    def forward(self, x):
        h_in = self._init_states(x)
        out, h_out = self.RNN(x, h_in)

        return out

    def _init_states(self, X):
        return Variable(X.data.new(1, X.size(0), self.hidden_dimensions).zero_())


class LayerNorm(nn.Module):

    def __init__(self, feature, eps=1e-6):

        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(feature))
        self.b_2 = nn.Parameter(torch.zeros(feature))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class TCN(nn.Module):

    def __init__(self, input_size, output_size, num_channels, d_model,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1, tied_weights=False):
        super(TCN, self).__init__()
        #self.encoder = nn.Embedding(output_size, input_size)
        self.attn = SelfAttention(num_attention_heads=1, input_size=16, hidden_size=16,
                                  hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.5)
        self.encoder = Time_Embeddings(d_model, 16, dropout)
        self.LayerNorm = LayerNorm(d_model)
        self.tcn = TemporalConvNet(d_model, num_channels, kernel_size, dropout=dropout)

        self.decoder = nn.Linear(num_channels[-1], output_size)
        if tied_weights:
            if num_channels[-1] != input_size:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            #self.decoder.weight = self.encoder.weight
            print("Weight tied")
        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        self.init_weights()


    def init_weights(self):
        #self.encoder.weight.data.normal_(0, 0.01)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        #input = input.reshape(input.shape[0], -1)
        hidden = self.attn(input)
        emb = self.encoder(hidden)
        #print('emb', emb.shape)
        emb = self.LayerNorm(emb)
        y = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
        y = self.decoder(y)
        return y.contiguous()
