import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


class LSTM_cell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTM_cell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias) # 망각/입력/셀/출력 게이트 4개로 쪼개져서 들어간다.(chunk(4, 1)) chunk(몇개의 텐서로 나눌지, 어떤 차원으로 나눌지)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std) # -std, std 사이의 임의의 실수 생성

    def forward(self, x, hidden):
        hx, cx = hidden # hidden : 이전 cell -> hx : 은닉 상태 / cx : cell 상태
        x = x.view(-1, x.size(1)) # 입력

        gates = self.x2h(x) + self.h2h(hx) # 입력 + 이전 기억
        gates = gates.squeeze()
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate) # 시그모이드 적용
        forgetgate = F.sigmoid(forgetgate)# 시그모이드 적용
        cellgate = F.tanh(cellgate)# tanh 적용
        outgate = F.sigmoid(outgate)# 시그모이드 적용

        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
        hy = torch.mul(outgate, F.tanh(cy))

        return (hy, cy)
    

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim # 은닉층의 뉴런/유닛 개수
        self.layer_dim = layer_dim
        self.lstm = LSTM_cell(input_dim, hidden_dim, layer_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        if torch.cuda.is_available():
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        outs = []
        cn = c0[0, :, :]
        hn = h0[:, :, :]

        for seq in range(x.size(1)): # 셀 계층
            hn, cn = self.lstm(x[:, seq, :], (hn, cn))
            outs.append(hn)

        out = outs[-1].squeeze()
        out = self.fc(out)

        return out