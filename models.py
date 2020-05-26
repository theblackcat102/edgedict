import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from warprnnt_pytorch import RNNTLoss
from ctc_decoder import decode as ctc_beam
from tokenizer import DEFAULT_TOKEN2ID


class RNNModel(nn.Module):
    def __init__(self, input_size, vocab_size, hidden_size, num_layers,
                 dropout=.2, blank=0, bidirectional=False):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.blank = blank
        # lstm hidden vector: (h_0, c_0) num_layers * num_directions, batch, hidden_size
        self.lstm = nn.GRU(
            input_size, hidden_size, num_layers, batch_first=True,
            dropout=dropout, bidirectional=bidirectional)
        if bidirectional:
            hidden_size *= 2
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, xs, hid=None):
        h, hid = self.lstm(xs, hid)
        return self.linear(h), hid

    def greedy_decode(self, xs):
        xs = self(xs)[0][0]     # only one sequence
        xs = F.log_softmax(xs, dim=1)
        logp, pred = torch.max(xs, dim=1)
        return pred.data.cpu().numpy(), -float(logp.sum())

    def beam_search(self, xs, W):
        ''' CTC '''
        xs = self(xs)[0][0]     # only one sequence
        logp = F.log_softmax(xs, dim=1)
        return ctc_beam(logp.data.cpu().numpy(), W)


class Transducer(nn.Module):
    def __init__(self, input_size, vocab_size, vocab_embed_size, hidden_size,
                 num_layers, dropout=.5, blank=0, bidirectional=False):
        super(Transducer, self).__init__()
        self.blank = blank
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.loss = RNNTLoss()
        # NOTE encoder & decoder only use lstm
        self.encoder = RNNModel(input_size, hidden_size, hidden_size,
                                num_layers, dropout, blank, bidirectional)
        self.embed = nn.Embedding(
            vocab_size, vocab_embed_size, padding_idx=blank)
        # self.embed.weight.data[1:] = torch.eye(vocab_embed_size)
        # self.embed.weight.requires_grad = False
        # self.decoder = RNNModel(vocab_embed_size, vocab_size, hidden_size, 1, dropout)
        self.decoder = nn.LSTM(vocab_embed_size, hidden_size, 1,
                               batch_first=True, dropout=0.)
        self.fc1 = nn.Linear(2*hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)

    def joint(self, f, g):
        ''' `f`: encoder lstm output (B,T,U,2H)
        `g`: decoder lstm output (B,T,U,H)
        NOTE f and g must have the same size except the last dim'''
        dim = len(f.shape) - 1
        out = torch.cat((f, g), dim=dim)
        out = torch.tanh(self.fc1(out))
        return self.fc2(out)

    def forward(self, xs, ys, xlen, ylen):
        xs, _ = self.encoder(xs)
        # concat first zero
        zero = ys.new_zeros((ys.shape[0], 1)).long()
        ymat = torch.cat((zero, ys), dim=1)
        # forwoard pm
        ymat = self.embed(ymat)
        ymat, _ = self.decoder(ymat)
        xs = xs.unsqueeze(dim=2)
        ymat = ymat.unsqueeze(dim=1)
        # expand
        sz = [max(i, j) for i, j in zip(xs.size()[:-1], ymat.size()[:-1])]
        xs = xs.expand(torch.Size(sz+[xs.shape[-1]]))
        ymat = ymat.expand(torch.Size(sz+[ymat.shape[-1]]))
        out = self.joint(xs, ymat)
        if ys.is_cuda:
            xlen = xlen.cuda()
            ylen = ylen.cuda()
        loss = self.loss(out, ys.int(), xlen, ylen)
        return loss

    def greedy_decode(self, x, xlen, bos_idx=DEFAULT_TOKEN2ID['<bos>']):
        x, _ = self.encoder(x)
        # vector preserve for embedding
        embed_bos = x.new_ones(x.shape[0], 1).long() * bos_idx
        y, (h, c) = self.decoder(self.embed(embed_bos))     # decode first zero
        y_seq = []
        log_p = []
        for i in range(x.shape[1]):
            ytu = self.joint(x[:, i], y[:, 0])
            out = F.log_softmax(ytu, dim=1)
            prob, pred = torch.max(out, dim=1)
            y_seq.append(pred)
            log_p.append(prob)
            embed_pred = self.embed(pred.unsqueeze(1))
            new_y, (new_h, new_c) = self.decoder(embed_pred, (h, c))
            # change pm state
            y[pred != self.blank, ...] = new_y[pred != self.blank, ...]
            h[:, pred != self.blank, :] = new_h[:, pred != self.blank, :]
            c[:, pred != self.blank, :] = new_c[:, pred != self.blank, :]
        y_seq = torch.stack(y_seq, dim=1)
        log_p = torch.stack(log_p, dim=1).sum(dim=1)
        ret_y = []
        for seq, seq_len in zip(y_seq, xlen):
            seq = seq.cpu().numpy()[:seq_len]
            ret_y.append(list(filter(lambda x: x != self.blank, seq)))
        return ret_y, -log_p

    def beam_search(self, xs, W=10, prefix=False,
                    bos_idx=DEFAULT_TOKEN2ID['<bos>']):
        '''''
        `xs`: acoustic model outputs
        NOTE only support one sequence (batch size = 1)
        '''''

        def forward_step(label, hidden):
            ''' `label`: int '''
            label = xs.new_tensor([label]).long().view(1, 1)
            label = self.embed(label)
            pred, hidden = self.decoder(label, hidden)
            return pred[0][0], hidden

        def isprefix(a, b):
            # a is the prefix of b
            if a == b or len(a) >= len(b):
                return False
            for i in range(len(a)):
                if a[i] != b[i]:
                    return False
            return True

        xs = self.encoder(xs)[0][0]
        B = [Sequence(blank=self.blank)]
        for i, x in enumerate(xs):
            # larger sequence first add
            sorted(B, key=lambda a: len(a.k), reverse=True)
            A = B
            B = []
            if prefix:
                # for y in A:
                #     y.logp = log_aplusb(y.logp, prefixsum(y, A, x))
                for j in range(len(A)-1):
                    for i in range(j+1, len(A)):
                        if not isprefix(A[i].k, A[j].k):
                            continue
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
                logp = F.log_softmax(ytu, dim=0)  # log probability for each k
                # TODO only use topk vocab
                for k in range(self.vocab_size):
                    yk = Sequence(y_hat)
                    yk.logp += float(logp[k])
                    if k == self.blank:
                        B.append(yk)              # next move
                        continue
                    # store prediction distribution and last hidden state
                    # yk.h.append(hidden); yk.k.append(k)
                    yk.h = hidden
                    yk.k.append(k)
                    if prefix:
                        yk.g.append(pred)
                    A.append(yk)
                # sort A
                # sorted(A, key=lambda a: a.logp, reverse=True) # just need to calculate maximum seq

                # sort B
                # sorted(B, key=lambda a: a.logp, reverse=True)
                y_hat = max(A, key=lambda a: a.logp)
                yb = max(B, key=lambda a: a.logp)
                if len(B) >= W and yb.logp >= y_hat.logp:
                    break

            # beam width
            sorted(B, key=lambda a: a.logp, reverse=True)
            B = B[:W]

        # return highest probability sequence
        print(B[0])
        return B[0].k, -B[0].logp


def log_aplusb(a, b):
    return max(a, b) + math.log1p(math.exp(-math.fabs(a-b)))


class Sequence():
    def __init__(self, seq=None, blank=0):
        if seq is None:
            self.g = []         # predictions of phoneme language model
            self.k = [blank]    # prediction phoneme label
            # self.h = [None]   # input hidden vector to phoneme model
            self.h = None
            self.logp = 0       # probability of this sequence, in log scale
        else:
            self.g = seq.g[:]   # save for prefixsum
            self.k = seq.k[:]
            self.h = seq.h
            self.logp = seq.logp

    def __str__(self):
        return 'Prediction: {}\nlog-likelihood {:.2f}\n'.format(' '.join([rephone[i] for i in self.k]), -self.logp)


if __name__ == "__main__":
    model = Transducer(128, 3600, 8, 64, 2).cuda()
    x = torch.randn((32, 128, 128)).float().cuda()
    y = torch.randint(0, 3500, (32, 10)).long().cuda()
    xlen = torch.from_numpy(np.array([128]*32)).int()
    ylen = torch.from_numpy(np.array([10]*32)).int()
    loss = model(x, y, xlen, ylen)
    print(loss)
