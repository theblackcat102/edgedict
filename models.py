import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tokenizer import NUL, PAD, BOS
# from ctc_decoder import decode as ctc_beam


class Transducer(nn.Module):
    def __init__(self, input_size, vocab_size, vocab_embed_size, hidden_size,
                 num_layers, dropout=0, blank=NUL):
        super(Transducer, self).__init__()
        self.blank = blank
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Encoder
        self.encoder = nn.GRU(
            input_size, hidden_size, num_layers, batch_first=True,
            dropout=dropout)
        self.encoder_fc = nn.Linear(hidden_size, hidden_size)
        # Decoder
        self.embed = nn.Embedding(
            vocab_size, vocab_embed_size, padding_idx=PAD)
        # NOTE!!!!, (h, c) is always length First
        self.decoder = nn.LSTM(
            vocab_embed_size, hidden_size, num_layers=1, batch_first=True)
        # Joint
        self.joint = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, vocab_size),
        )

    def forward(self, xs, ys):
        # encoder
        self.encoder.flatten_parameters()
        h_enc, _ = self.encoder(xs)
        h_enc = self.encoder_fc(h_enc)
        # decoder
        self.decoder.flatten_parameters()
        bos = ys.new_ones((ys.shape[0], 1)).long() * BOS
        h_pre = torch.cat([bos, ys], dim=-1)
        h_pre, _ = self.decoder(self.embed(h_pre))
        # expand
        h_enc = h_enc.unsqueeze(dim=2)
        h_pre = h_pre.unsqueeze(dim=1)
        h_enc = h_enc.expand(-1, -1, h_pre.shape[2], -1)
        h_pre = h_pre.expand(-1, h_enc.shape[1], -1, -1)
        # joint
        prob = self.joint(torch.cat([h_enc, h_pre], dim=-1))
        return prob

    def greedy_decode(self, xs, xlen):
        # encoder
        h_enc, _ = self.encoder(xs)
        h_enc = self.encoder_fc(h_enc)
        # initialize decoder
        bos = xs.new_ones(xs.shape[0], 1).long() * BOS
        h_pre, (h, c) = self.decoder(self.embed(bos))     # decode first zero
        y_seq = []
        log_p = []
        # greedy
        for i in range(h_enc.shape[1]):
            # joint
            logits = self.joint(torch.cat([h_enc[:, i], h_pre[:, 0]], dim=-1))
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

#     def beam_search(self, xs, W=10, prefix=False,
#                     bos_idx=DEFAULT_TOKEN2ID['<bos>']):
#         '''''
#         xs: acoustic model outputs
#         NOTE only support one sequence (batch size = 1)
#         '''''

#         def forward_step(label, hidden):
#             ''' `label`: int '''
#             label = xs.new_tensor([label]).long().view(1, 1)
#             label = self.embed(label)
#             pred, hidden = self.decoder(label, hidden)
#             return pred[0][0], hidden

#         def isprefix(a, b):
#             # a is the prefix of b
#             if a == b or len(a) >= len(b):
#                 return False
#             for i in range(len(a)):
#                 if a[i] != b[i]:
#                     return False
#             return True

#         xs, _ = self.encoder(xs)
#         xs = self.encoder2vocab(xs)
#         B = [Sequence(blank=self.blank)]
#         for i, x in enumerate(xs):
#             # larger sequence first add
#             sorted(B, key=lambda a: len(a.k), reverse=True)
#             A = B
#             B = []
#             if prefix:
#                 # for y in A:
#                 #     y.logp = log_aplusb(y.logp, prefixsum(y, A, x))
#                 for j in range(len(A)-1):
#                     for i in range(j+1, len(A)):
#                         if not isprefix(A[i].k, A[j].k):
#                             continue
#                         # A[i] -> A[j]
#                         pred, _ = forward_step(A[i].k[-1], A[i].h)
#                         idx = len(A[i].k)
#                         ytu = self.joint(x, pred)
#                         logp = F.log_softmax(ytu, dim=0)
#                         curlogp = A[i].logp + float(logp[A[j].k[idx]])
#                         for k in range(idx, len(A[j].k)-1):
#                             ytu = self.joint(x, A[j].g[k])
#                             logp = F.log_softmax(ytu, dim=0)
#                             curlogp += float(logp[A[j].k[k+1]])
#                         A[j].logp = log_aplusb(A[j].logp, curlogp)

#             while True:
#                 y_hat = max(A, key=lambda a: a.logp)
#                 # y* = most probable in A
#                 A.remove(y_hat)
#                 # calculate P(k|y_hat, t)
#                 # get last label and hidden state
#                 pred, hidden = forward_step(y_hat.k[-1], y_hat.h)
#                 ytu = self.joint(x, pred)
#                 logp = F.log_softmax(ytu, dim=0)  # log probability for each k
#                 # TODO only use topk vocab
#                 for k in range(self.vocab_size):
#                     yk = Sequence(y_hat)
#                     yk.logp += float(logp[k])
#                     if k == self.blank:
#                         B.append(yk)              # next move
#                         continue
#                     # store prediction distribution and last hidden state
#                     # yk.h.append(hidden); yk.k.append(k)
#                     yk.h = hidden
#                     yk.k.append(k)
#                     if prefix:
#                         yk.g.append(pred)
#                     A.append(yk)
#                 # sort A
#                 # just need to calculate maximum seq
#                 # sorted(A, key=lambda a: a.logp, reverse=True)

#                 # sort B
#                 # sorted(B, key=lambda a: a.logp, reverse=True)
#                 y_hat = max(A, key=lambda a: a.logp)
#                 yb = max(B, key=lambda a: a.logp)
#                 if len(B) >= W and yb.logp >= y_hat.logp:
#                     break

#             # beam width
#             sorted(B, key=lambda a: a.logp, reverse=True)
#             B = B[:W]

#         # return highest probability sequence
#         print(B[0])
#         return B[0].k, -B[0].logp


# def log_aplusb(a, b):
#     return max(a, b) + math.log1p(math.exp(-math.fabs(a-b)))


# class Sequence():
#     def __init__(self, seq=None, blank=0):
#         if seq is None:
#             self.g = []         # predictions of phoneme language model
#             self.k = [blank]    # prediction phoneme label
#             # self.h = [None]   # input hidden vector to phoneme model
#             self.h = None
#             self.logp = 0       # probability of this sequence, in log scale
#         else:
#             self.g = seq.g[:]   # save for prefixsum
#             self.k = seq.k[:]
#             self.h = seq.h
#             self.logp = seq.logp

#     def __str__(self):
#         return 'Prediction: {}\nlog-likelihood {:.2f}\n'.format(
#             ' '.join([rephone[i] for i in self.k]), -self.logp)


if __name__ == "__main__":
    model = Transducer(128, 3600, 8, 64, 2).cuda()
    x = torch.randn((32, 128, 128)).float().cuda()
    y = torch.randint(0, 3500, (32, 10)).long().cuda()
    xlen = torch.from_numpy(np.array([128]*32)).int()
    ylen = torch.from_numpy(np.array([10]*32)).int()
    loss = model(x, y, xlen, ylen)
    print(loss)
