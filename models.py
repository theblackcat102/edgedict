import copy
import numpy as np
import torch
from torch import nn, autograd
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from recurrent import ResidualRNNModel
from modules.tokenizer import BOS

def fast_tanh(x):
    return x / (1 + x.abs())

class RNNModel(nn.Module):
    def __init__(self, input_size, vocab_size, hidden_size, num_layers, dropout=.2, blank=0, bidirectional=False):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.blank = blank
        # normalize spectrum feature
        self.spectrum_norm = nn.BatchNorm1d(input_size)
        # lstm hidden vector: (h_0, c_0) num_layers * num_directions, batch, hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        if bidirectional: hidden_size *= 2
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, xs, hid=None):
        xs = xs.permute(0, 2, 1)
        xs = self.spectrum_norm(xs)
        xs = xs.permute(0, 2, 1)
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
    def __init__(self, input_size, vocab_size, vocab_embed_size, hidden_size, num_layers, pred_hidden_size=-1, pred_num_layers=1,dropout=.2, blank=0, bidirectional=False):
        super(Transducer, self).__init__()
        self.blank = blank
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if pred_hidden_size == -1:
            pred_hidden_size = hidden_size
        # NOTE encoder & decoder only use lstm
        self.encoder = ResidualRNNModel(input_size, hidden_size, hidden_size, num_layers, dropout, bidirectional=False)
        self.embed = nn.Embedding(vocab_size, vocab_embed_size, padding_idx=1)
        # self.embed.weight.data[1:] = torch.eye(vocab_embed_size)
        # self.embed.weight.requires_grad = False
        # self.decoder = RNNModel(vocab_embed_size, vocab_size, hidden_size, 1, dropout)
        self.decoder = nn.LSTM(vocab_embed_size, pred_hidden_size, pred_num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size+pred_hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)

    def joint(self, f, g):
        ''' `f`: encoder lstm output (B,T,U,2H)
        `g`: decoder lstm output (B,T,U,H)
        NOTE f and g must have the same size except the last dim'''
        out = torch.cat((f, g), dim=-1)
        out = fast_tanh(self.fc1(out))
        return self.fc2(out)

    def forward(self, i_xs, ys, xlen, ylen):
        xs, _ = self.encoder(i_xs)
        # concat first zero
        bos = ys.new_ones((ys.shape[0], 1)).long() * BOS
        h_pre = torch.cat([bos, ys.long()], dim=-1)
        ymat, _ = self.decoder(self.embed(h_pre))
        xs = xs.unsqueeze(dim=2)
        ymat = ymat.unsqueeze(dim=1)
        # expand 
        sz = [max(i, j) for i, j in zip(xs.size()[:-1], ymat.size()[:-1])]
        xs = xs.expand(torch.Size(sz+[xs.shape[-1]])); ymat = ymat.expand(torch.Size(sz+[ymat.shape[-1]]))
        out = self.joint(xs, ymat)
        # loss = self.loss(out, ys.int(), xlen, ylen)
        return out

    def greedy_decode(self, xs, xlen):
        # encoder
        h_enc, _ = self.encoder(xs)
        # initialize decoder
        bos = xs.new_ones(xs.shape[0], 1).long() * BOS
        h_pre, (h, c) = self.decoder(self.embed(bos))     # decode first zero
        y_seq = []
        log_p = []
        # greedy
        for i in range(h_enc.shape[1]):
            # joint
            logits = self.joint(h_enc[:, i], h_pre[:, 0])
            probs = F.log_softmax(logits, dim=1)
            prob, pred = torch.max(probs, dim=1)
            y_seq.append(pred)
            log_p.append(prob)
            embed_pred = self.embed(pred.unsqueeze(1))
            new_h_pre, (new_h, new_c) = self.decoder(embed_pred, (h, c))
            # replace non blank entities with new state
            h_pre[pred != self.blank, ...] = new_h_pre[pred != self.blank, ...]
            h[:, pred != self.blank, :] = new_h[:, pred != self.blank, :]
            c[:, pred != self.blank, :] = new_c[:, pred != self.blank, :]
        y_seq = torch.stack(y_seq, dim=1)
        log_p = torch.stack(log_p, dim=1).sum(dim=1)
        ret_y = []
        # truncat to xlen and remove blank token
        for seq, seq_len in zip(y_seq, xlen):
            seq = seq.cpu().numpy()[:seq_len]
            ret_y.append(list(filter(lambda tok: tok != self.blank, seq)))
        return ret_y, -log_p



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
        # print(B[0])
        return B[0].k, -B[0].logp


import math
def log_aplusb(a, b):
    return max(a, b) + math.log1p(math.exp(-math.fabs(a-b)))

class Sequence():
    def __init__(self, seq=None, blank=0):
        if seq is None:
            self.g = [] # predictions of phoneme language model
            self.k = [1] # prediction phoneme label
            # self.h = [None] # input hidden vector to phoneme model
            self.h = None
            self.logp = 0 # probability of this sequence, in log scale
        else:
            self.g = seq.g[:] # save for prefixsum
            self.k = seq.k[:]
            self.h = seq.h
            self.logp = seq.logp


class LMModel(nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(LMModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout, batch_first=True)
        self.decoder = nn.Linear(nhid, ntoken)
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
        self.init_weights()
        self.nhid = nhid
        self.rnn_type = 'LSTM'
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)
    
    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=-1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

if __name__ == "__main__":
    import torch
    from torch.autograd import Variable
    import numpy as np

    # model = Transducer(128,3600,8 ,64, 4).cuda()
    # x = torch.randn((32, 128, 128)).float().cuda()
    # y = torch.randint(0, 3500, (32, 10)).long().cuda()
    # xlen = torch.from_numpy(np.array([128]*32)).int()
    # ylen = torch.from_numpy(np.array([10]*32)).int()

    # x = pad_sequence(x, batch_first=True)
    # x = pack_padded_sequence(x, lengths=xlen, batch_first=True)

    # loss = model(x, y, xlen, ylen)
    # loss.backward()
    # print(loss)
    model = LMModel(1024, 64, 256, 3, dropout=0.2, tie_weights=False)
    checkpoint = torch.load('lm_model.pt',map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)