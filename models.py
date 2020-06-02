import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tokenizer import NUL, PAD, BOS
# from ctc_decoder import decode as ctc_beam


class TimeReduction(nn.Module):
    def __init__(self, reduction_factor):
        super().__init__()
        self.reduction_factor = reduction_factor

    def forward(self, xs):
        batch_size, xlen, hidden_size = xs.shape
        pad_shape = [[0, 0], [0, xlen % self.reduction_factor], [0, 0]]
        xs = nn.functional.pad(xs, pad_shape)
        xs = xs.view(batch_size, -1, self.reduction_factor, hidden_size)
        xs = xs.mean(dim=2)
        return xs


class LayerNormRNN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 dropout=0,
                 proj_size=None,
                 time_reductions=None,
                 reduction_factor=2):
        super().__init__()
        self.rnns = nn.ModuleList()
        self.projs = nn.ModuleList()
        if proj_size is None:
            proj_size = hidden_size
        for i in range(num_layers):
            self.rnns.append(
                nn.LSTM(input_size, hidden_size, 1, batch_first=True))
            if time_reductions is not None and i in time_reductions:
                proj = [TimeReduction(reduction_factor)]
            else:
                proj = []
            if proj_size is not None:
                proj.append(nn.Linear(hidden_size, proj_size))
                output_size = proj_size
            else:
                output_size = hidden_size
            proj.extend([
                nn.Dropout(dropout),
                nn.LayerNorm(output_size)
            ])
            self.projs.append(nn.Sequential(*proj))
            input_size = output_size

    def forward(self, xs, hiddens=None):
        if hiddens is None:
            hiddens = [None for _ in range(len(self.rnns))]
        else:
            hs, cs = hiddens[0].unsqueeze(1), hiddens[1].unsqueeze(1)
            hiddens = zip(hs, cs)
        new_hiddens = []
        for rnn, proj, hidden in zip(self.rnns, self.projs, hiddens):
            rnn.flatten_parameters()
            xs, new_hidden = rnn(xs, hidden)
            new_hiddens.append(new_hidden)
        hs, cs = zip(*new_hiddens)
        hs = torch.cat(hs, dim=0)
        cs = torch.cat(cs, dim=0)
        return xs, (hs, cs)


class Transducer(nn.Module):
    def __init__(self,
                 vocab_size,
                 vocab_embed_size,
                 input_size,
                 hidden_size=256,
                 enc_layers=3,
                 enc_dropout=0,
                 dec_layers=1,
                 dec_dropout=0,
                 proj_size=None,
                 time_reductions=[1],
                 reduction_factor=3,
                 blank=NUL):
        super(Transducer, self).__init__()
        self.blank = blank
        # Encoder
        self.encoder = LayerNormRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=enc_layers,
            dropout=enc_dropout,
            proj_size=proj_size,
            time_reductions=time_reductions,
            reduction_factor=reduction_factor)
        # Decoder
        self.embed = nn.Embedding(
            vocab_size, vocab_embed_size, padding_idx=PAD)
        self.decoder = LayerNormRNN(
            input_size=vocab_embed_size,
            hidden_size=hidden_size,
            num_layers=dec_layers,
            dropout=dec_dropout,
            proj_size=proj_size,
            time_reductions=None)
        # Joint
        self.joint = nn.Sequential(
            nn.Linear(hidden_size, proj_size),
            nn.Tanh(),
            nn.Linear(proj_size, vocab_size),
        )

    def forward(self, xs, ys):
        # encoder
        h_enc, _ = self.encoder(xs)
        # decoder
        bos = ys.new_ones((ys.shape[0], 1)).long() * BOS
        h_pre = torch.cat([bos, ys.long()], dim=-1)
        h_pre, _ = self.decoder(self.embed(h_pre))
        # expand
        h_enc = h_enc.unsqueeze(dim=2)
        h_pre = h_pre.unsqueeze(dim=1)
        # joint
        prob = self.joint(h_enc + h_pre)
        return prob

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
            logits = self.joint(h_enc[:, i] + h_pre[:, 0])
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
        y_seq_truncated = []
        for seq, seq_len in zip(y_seq, xlen):
            y_seq_truncated.append(seq[:seq_len].cpu().numpy())
        return y_seq_truncated, -log_p

    def beam_search_batch(self, xs, xlen, k=10, bos_idx=BOS):
        y_seq_truncated = []
        log_p = []
        for x, xlen_ in zip(xs, xlen):
            seq, prob = self.beam_search_old(x[None, :xlen_], xlen_, k)
            y_seq_truncated.append(seq.cpu().numpy())
            log_p.append(prob)
        return y_seq_truncated, -torch.stack(log_p)

    def beam_search(self, xs, xlen, k=10, bos_idx=BOS):
        def log_plus(a, b):
            return (torch.max(torch.stack([a, b], dim=-1), dim=-1)[0] +
                    torch.log1p(torch.exp(-torch.abs(a - b))))
        # encoder
        h_enc, _ = self.encoder(xs)
        print(h_enc.shape)
        # initialize decoder
        bos = xs.new_ones(xs.shape[0], 1).long() * BOS
        h_pre, (h, c) = self.decoder(self.embed(bos))     # decode first zero
        h_pre = h_pre.expand(k, -1, -1).contiguous()
        h = h.expand(-1, k, -1).contiguous()
        c = c.expand(-1, k, -1).contiguous()
        # print(h_pre.shape)
        # print(h.shape)
        # print(c.shape)
        # h_pre [k, 1, H]
        # h [1, k, H]
        # c [1, k, H]
        # joint
        prob = F.log_softmax(self.joint(h_enc[:, 0] + h_pre[:, 0]), dim=-1)
        # print(prob)
        B_prob, B_seqs = torch.topk(prob[0], k=k, dim=-1)
        B_prob = B_prob.view(-1, 1)   # [k, 1]
        B_seqs = B_seqs.view(-1, 1)   # [k, 1]
        for i in range(1, h_enc.shape[1]):
            embed_pred = self.embed(B_seqs[:, -1:])
            # print('0 -----')
            # print(embed_pred.shape)
            # embed_pred [k, 1, Hi]
            new_h_pre, (new_h, new_c) = self.decoder(embed_pred, (h, c))
            # print('1 -----')
            # print(new_h_pre.shape)
            # print(new_h.shape)
            # print(new_c.shape)
            # new_h_pre [k, 1, H]
            # new_h [1, k, H]
            # new_c [1, k, H]
            # print('2 -----')
            # print(h_pre.shape)
            # print(h.shape)
            # print(c.shape)
            h_pre[B_seqs[:, -1] != self.blank] = \
                new_h_pre[B_seqs[:, -1] != self.blank]
            h[:, B_seqs[:, -1] != self.blank, :] = \
                new_h[:, B_seqs[:, -1] != self.blank, :]
            c[:, B_seqs[:, -1] != self.blank, :] = \
                new_c[:, B_seqs[:, -1] != self.blank, :]
            # h_pre = h_pre.contiguous()
            # h = h.contiguous()
            # c = c.contiguous()
            # print('3 -----')
            # print(h_pre.shape)
            # print(h.shape)
            # print(c.shape)

            logits = self.joint(h_enc[:, i] + h_pre[:, 0])
            prob = F.log_softmax(logits, dim=1)
            # [k, k], [k, k]
            prob, topk = torch.topk(prob, k=k, dim=-1)
            new_B_prob = log_plus(B_prob.expand_as(prob), prob)  # [k, k]
            topk = topk.unsqueeze(-1)
            new_B_seqs = B_seqs.unsqueeze(1).expand(-1, k, -1)
            new_B_seqs = torch.cat([new_B_seqs, topk], dim=-1)  # [k, k, L]
            B_prob, topk = torch.topk(new_B_prob.view(-1), k=k)
            B_seqs = new_B_seqs.view(k * k, -1)[topk]
        return B_seqs[0], B_prob[0]

    def beam_search_old(self, xs, xlen, k=10, prefix=False, bos_idx=BOS):
        '''''
        xs: acoustic model outputs
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

        xs, _ = self.encoder(xs)
        # print(xs.shape)
        B = [Sequence(blank=self.blank)]
        for i, x in enumerate(xs[0]):
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
                        ytu = self.joint(x + pred)
                        logp = F.log_softmax(ytu, dim=0)
                        curlogp = A[i].logp + float(logp[A[j].k[idx]])
                        for k in range(idx, len(A[j].k)-1):
                            ytu = self.joint(x + A[j].g[k])
                            logp = F.log_softmax(ytu, dim=0)
                            curlogp += float(logp[A[j].k[k+1]])
                        A[j].logp = log_plus(A[j].logp, curlogp)

            while True:
                y_hat = max(A, key=lambda a: a.logp)
                # y* = most probable in A
                A.remove(y_hat)
                # calculate P(k|y_hat, t)
                # get last label and hidden state
                pred, hidden = forward_step(y_hat.k[-1], y_hat.h)
                print(x.shape, pred.shape)
                ytu = self.joint(x + pred)
                print(ytu.shape)
                logp = F.log_softmax(ytu, dim=0)
                # TODO only use topk vocab
                for k in range(len(logp)):
                    yk = Sequence(y_hat)
                    yk.logp += float(logp[k])
                    assert yk.logp <= 0
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
                # just need to calculate maximum seq
                # sorted(A, key=lambda a: a.logp, reverse=True)

                # sort B
                # sorted(B, key=lambda a: a.logp, reverse=True)
                y_hat = max(A, key=lambda a: a.logp)
                yb = max(B, key=lambda a: a.logp)
                if len(B) >= k and yb.logp >= y_hat.logp:
                    break

            # beam width
            sorted(B, key=lambda a: a.logp, reverse=True)
            B = B[:k]

        # return highest probability sequence
        # print(B[0])
        return torch.tensor(B[0].k), torch.tensor(-B[0].logp)


def log_plus(a, b):
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

    # def __str__(self):
    #     return 'Prediction: {}\nlog-likelihood {:.2f}\n'.format(
    #         ' '.join([rephone[i] for i in self.k]), -self.logp)


if __name__ == "__main__":
    # test model
    # from warprnnt_pytorch import RNNTLoss
    # model = Transducer(
    #     vocab_size=34,
    #     vocab_embed_size=16,
    #     input_size=40,
    #     proj_size=256)
    # loss_fn = RNNTLoss(blank=0)
    # xs = torch.randn((2, 200, 40)).float()
    # ys = torch.randint(0, 20, (2, 32)).int()
    # xlen = torch.ones(2).int() * 200
    # ylen = torch.ones(2).int() * 32
    # prob = model(xs, ys)
    # loss = loss_fn(prob, ys, xlen, ylen)
    # print(prob.shape)

    # test beamsearch
    model = Transducer(
        vocab_size=34,
        vocab_embed_size=16,
        input_size=40,
        enc_layers=2,
        dec_layers=1,
        proj_size=256).cuda()
    xs = torch.randn((1, 50, 40)).float().cuda()
    xlen = torch.ones(1).int().cuda() * 50
    seqs, prob = model.beam_search(xs, xlen, k=3)
    print(seqs.shape, prob.shape)
