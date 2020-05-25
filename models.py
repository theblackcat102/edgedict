import copy
import numpy as np
import torch
from torch import nn, autograd
import torch.nn.functional as F
from warprnnt_pytorch import RNNTLoss
from ctc_decoder import decode as ctc_beam

class RNNModel(nn.Module):
    def __init__(self, input_size, vocab_size, hidden_size, num_layers, dropout=.2, blank=0, bidirectional=False):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.blank = blank
        # lstm hidden vector: (h_0, c_0) num_layers * num_directions, batch, hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        if bidirectional: hidden_size *= 2
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, xs, hid=None):
        h, hid = self.lstm(xs, hid)
        return self.linear(h), hid

    def greedy_decode(self, xs):
        xs = self(xs)[0][0] # only one sequence
        xs = F.log_softmax(xs, dim=1)
        logp, pred = torch.max(xs, dim=1)
        return pred.data.cpu().numpy(), -float(logp.sum())

    def beam_search(self, xs, W):
        ''' CTC '''
        xs = self(xs)[0][0] # only one sequence
        logp = F.log_softmax(xs, dim=1)
        return ctc_beam(logp.data.cpu().numpy(), W)

class Transducer(nn.Module):
    def __init__(self, input_size, vocab_size, vocab_embed_size, hidden_size, num_layers, dropout=.5, blank=0, bidirectional=False):
        super(Transducer, self).__init__()
        self.blank = blank
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.loss = RNNTLoss()
        # NOTE encoder & decoder only use lstm
        self.encoder = RNNModel(input_size, hidden_size, hidden_size, num_layers, dropout, bidirectional=bidirectional)
        self.embed = nn.Embedding(vocab_size, vocab_embed_size, padding_idx=blank)
        # self.embed.weight.data[1:] = torch.eye(vocab_embed_size)
        # self.embed.weight.requires_grad = False
        # self.decoder = RNNModel(vocab_embed_size, vocab_size, hidden_size, 1, dropout)
        self.decoder = nn.LSTM(vocab_embed_size, hidden_size, 1, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(2*hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)

    def joint(self, f, g):
        ''' `f`: encoder lstm output (B,T,U,2H)
        `g`: decoder lstm output (B,T,U,H)
        NOTE f and g must have the same size except the last dim'''
        dim = len(f.shape) - 1
        out = torch.cat((f, g), dim=dim)
        out = F.tanh(self.fc1(out))
        return self.fc2(out)

    def forward(self, xs, ys, xlen, ylen):
        xs, _ = self.encoder(xs)
        # concat first zero
        zero = autograd.Variable(torch.zeros((ys.shape[0], 1)).long())
        if ys.is_cuda: zero = zero.cuda()
        ymat = torch.cat((zero, ys), dim=1)
        # forwoard pm
        ymat = self.embed(ymat)
        ymat, _ = self.decoder(ymat)
        xs = xs.unsqueeze(dim=2)
        ymat = ymat.unsqueeze(dim=1)
        # expand 
        sz = [max(i, j) for i, j in zip(xs.size()[:-1], ymat.size()[:-1])]
        xs = xs.expand(torch.Size(sz+[xs.shape[-1]])); ymat = ymat.expand(torch.Size(sz+[ymat.shape[-1]]))
        out = self.joint(xs, ymat)
        if ys.is_cuda:
            xlen = xlen.cuda()
            ylen = ylen.cuda()
        loss = self.loss(out, ys.int(), xlen, ylen)
        return loss

    def greedy_decode(self, x, bos_idx=1):
        x = self.encoder(x)[0][0]
        vy = autograd.Variable(torch.LongTensor([bos_idx])).view(1,1) # vector preserve for embedding
        if x.is_cuda: vy = vy.cuda()
        y, h = self.decoder(self.embed(vy)) # decode first zero 
        y_seq = []; logp = 0
        for i in x:
            ytu = self.joint(i, y[0][0])
            out = F.log_softmax(ytu, dim=0)
            p, pred = torch.max(out, dim=0) # suppose blank = -1
            pred = int(pred); logp += float(p)
            if pred != self.blank:
                y_seq.append(pred)
                vy.data[0][0] = pred # change pm state
                y, h = self.decoder(self.embed(vy), h)
        return y_seq, -logp


    def beam_search(self, xs, W=10, prefix=False, bos_idx=1):
        '''''
        `xs`: acoustic model outputs
        NOTE only support one sequence (batch size = 1)
        '''''
        use_gpu = xs.is_cuda
        def forward_step(label, hidden):
            ''' `label`: int '''
            label = autograd.Variable(torch.LongTensor([label]), volatile=True).view(1,1)
            if use_gpu: label = label.cuda()
            label = self.embed(label)
            pred, hidden = self.decoder(label, hidden)
            return pred[0][0], hidden

        def isprefix(a, b):
            # a is the prefix of b
            if a == b or len(a) >= len(b): return False
            for i in range(len(a)):
                if a[i] != b[i]: return False
            return True

        xs = self.encoder(xs)[0][0]
        B = [Sequence(blank=self.blank)]
        for i, x in enumerate(xs):
            sorted(B, key=lambda a: len(a.k), reverse=True) # larger sequence first add
            A = B
            B = []
            if prefix:
                # for y in A:
                #     y.logp = log_aplusb(y.logp, prefixsum(y, A, x))
                for j in range(len(A)-1):
                    for i in range(j+1, len(A)):
                        if not isprefix(A[i].k, A[j].k): continue
                        # A[i] -> A[j]
                        pred, _ = forward_step(A[i].k[-1], A[i].h)
                        idx = len(A[i].k)
                        ytu = self.joint(x, pred)
                        logp = F.log_softmax(ytu, dim=0)
                        curlogp = A[i].logp + float(logp[A[j].k[idx]])
                        for k in range(idx, len(A[j].k)-1):
                            ytu = self.joint(x, A[j].g[k])
                            logp = F.log_softmax(ytu, dim=0)
                            curlogp += float(logp[A[j].k[k+1]])
                        A[j].logp = log_aplusb(A[j].logp, curlogp)

            while True:
                y_hat = max(A, key=lambda a: a.logp)
                # y* = most probable in A
                A.remove(y_hat)
                # calculate P(k|y_hat, t)
                # get last label and hidden state
                pred, hidden = forward_step(y_hat.k[-1], y_hat.h)
                ytu = self.joint(x, pred)
                logp = F.log_softmax(ytu, dim=0) # log probability for each k
                # TODO only use topk vocab
                for k in range(self.vocab_size):
                    yk = Sequence(y_hat)
                    yk.logp += float(logp[k])
                    if k == self.blank:
                        B.append(yk) # next move
                        continue
                    # store prediction distribution and last hidden state
                    # yk.h.append(hidden); yk.k.append(k)
                    yk.h = hidden; yk.k.append(k); 
                    if prefix: yk.g.append(pred)
                    A.append(yk)
                # sort A
                # sorted(A, key=lambda a: a.logp, reverse=True) # just need to calculate maximum seq
                
                # sort B
                # sorted(B, key=lambda a: a.logp, reverse=True)
                y_hat = max(A, key=lambda a: a.logp)
                yb = max(B, key=lambda a: a.logp)
                if len(B) >= W and yb.logp >= y_hat.logp: break

            # beam width
            sorted(B, key=lambda a: a.logp, reverse=True)
            B = B[:W]

        # return highest probability sequence
        print(B[0])
        return B[0].k, -B[0].logp


import math
def log_aplusb(a, b):
    return max(a, b) + math.log1p(math.exp(-math.fabs(a-b)))

class Sequence():
    def __init__(self, seq=None, blank=0):
        if seq is None:
            self.g = [] # predictions of phoneme language model
            self.k = [blank] # prediction phoneme label
            # self.h = [None] # input hidden vector to phoneme model
            self.h = None
            self.logp = 0 # probability of this sequence, in log scale
        else:
            self.g = seq.g[:] # save for prefixsum
            self.k = seq.k[:]
            self.h = seq.h
            self.logp = seq.logp

    def __str__(self):
        return 'Prediction: {}\nlog-likelihood {:.2f}\n'.format(' '.join([rephone[i] for i in self.k]), -self.logp)


if __name__ == "__main__":
    import torch
    from torch.autograd import Variable
    import numpy as np

    model = Transducer(128, 3600, 64, 2).cuda()
    x = torch.randn((32, 128, 128)).float().cuda()
    y = torch.randint(0, 3500, (32, 10)).long().cuda()
    xlen = torch.from_numpy(np.array([128]*32)).int()
    ylen = torch.from_numpy(np.array([10]*32)).int()
    loss = model(x, y, xlen, ylen)
    print(loss)