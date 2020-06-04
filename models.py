import torch
import torch.nn.functional as F
from torch import nn

from tokenizer import NUL, BOS


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0,
                 proj_size=None):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout)
        self.proj = nn.Linear(hidden_size, proj_size)

    def forward(self, xs, hidden=None):
        self.lstm.flatten_parameters()
        xs, hidden = self.lstm(xs, hidden)
        xs = self.proj(xs)
        return xs, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout=0,
                 proj_size=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout)
        self.proj = nn.Linear(hidden_size, proj_size)

    def forward(self, ys, hidden=None):
        if hidden is None:
            ys = F.pad(ys, [1, 0, 0, 0], value=BOS).long()
        ys = self.embed(ys)
        self.lstm.flatten_parameters()
        ys, hidden = self.lstm(ys, hidden)
        ys = self.proj(ys)
        return ys, hidden


class Joint(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size):
        super().__init__()
        self.joint = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, vocab_size),
        )

    def forward(self, h_enc, h_dec):
        if len(h_enc.shape) == 3 and len(h_dec.shape) == 3:
            h_enc = h_enc.unsqueeze(dim=2)
            h_dec = h_dec.unsqueeze(dim=1)
            h_enc = h_enc.expand(-1, -1, h_dec.size(2), -1)
            h_dec = h_dec.expand(-1, h_enc.size(1), -1, -1)
        else:
            assert len(h_enc.shape) == len(h_dec.shape)
        h = torch.cat([h_enc, h_dec], dim=-1)
        h = self.joint(h)
        return h


class Transducer(nn.Module):
    def __init__(self,
                 vocab_size, input_size,
                 enc_hidden_size, enc_layers, enc_dropout,
                 dec_hidden_size, dec_layers, dec_dropout,
                 proj_size, joint_size, blank=NUL):
        super().__init__()
        self.blank = blank
        # Encoder
        self.encoder = Encoder(
            input_size=input_size,
            hidden_size=enc_hidden_size,
            num_layers=enc_layers,
            dropout=enc_dropout,
            proj_size=proj_size)
        # Decoder
        self.decoder = Decoder(
            vocab_size=vocab_size,
            hidden_size=dec_hidden_size,
            num_layers=dec_layers,
            dropout=dec_dropout,
            proj_size=proj_size)
        # Joint
        self.joint = Joint(
            input_size=proj_size * 2,
            hidden_size=joint_size,
            vocab_size=vocab_size)

    def forward(self, xs, ys):
        h_enc, _ = self.encoder(xs)
        h_dec, _ = self.decoder(ys)
        logits = self.joint(h_enc, h_dec)
        return logits

    def greedy_decode(self, xs, xlen):
        # encoder
        h_enc, _ = self.encoder(xs)
        # decoder
        h_dec, (h_prev, c_prev) = self.decoder(xs.new_empty(xs.shape[0], 0))
        y_seq = []
        log_p = []
        # greedy
        for i in range(h_enc.shape[1]):
            # joint
            logits = self.joint(h_enc[:, i], h_dec[:, 0])
            probs = F.log_softmax(logits, dim=1)
            prob, pred = torch.max(probs, dim=1)
            y_seq.append(pred)
            log_p.append(prob)
            # replace non blank entities with new state
            h_dec_new, (h_next, c_next) = self.decoder(
                pred.unsqueeze(-1), (h_prev, c_prev))
            h_dec[pred != self.blank, ...] = h_dec_new[pred != self.blank, ...]
            h_prev[:, pred != self.blank, :] = h_next[:, pred != self.blank, :]
            c_prev[:, pred != self.blank, :] = c_next[:, pred != self.blank, :]
        y_seq = torch.stack(y_seq, dim=1)
        log_p = torch.stack(log_p, dim=1).sum(dim=1)
        y_seq_truncated = []
        for seq, seq_len in zip(y_seq, xlen):
            y_seq_truncated.append(seq[:seq_len].cpu().numpy())
        return y_seq_truncated, -log_p


class CTCEncoder(nn.Module):
    def __init__(self, vocab_size, input_size,
                 enc_hidden_size, enc_layers, enc_dropout,
                 proj_size, blank=NUL):
        super().__init__()
        self.blank = blank
        self.model = Encoder(
            input_size=input_size,
            hidden_size=enc_hidden_size,
            num_layers=enc_layers,
            dropout=enc_dropout,
            proj_size=proj_size)
        self.tovocab = nn.Sequential(
            nn.Linear(proj_size, vocab_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, xs):
        xs, _ = self.model(xs)
        logprobs = self.tovocab(xs)
        return logprobs

    def greedy_decode(self, xs, xlen):
        xs, _ = self.model(xs)
        logprobs = self.tovocab(xs)
        logprob, y_seq = logprobs.max(dim=-1)
        unique = y_seq[:, 1:] != y_seq[:, :-1]
        unique = F.pad(unique, [1, 0, 0, 0], value=True).bool()
        nonblank = y_seq != self.blank
        masks = (nonblank.int() * unique.int()).bool()
        y_seq_truncated = []
        log_p = []
        for seq, logprob, seq_len, mask in zip(y_seq, logprobs, xlen, masks):
            mask = mask[:seq_len]
            seq = seq[:seq_len][mask]
            logprob = logprob[:seq_len][mask]
            y_seq_truncated.append(seq.cpu().numpy())
            log_p.append(logprob.sum())
        return y_seq_truncated, -torch.stack(log_p)


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
