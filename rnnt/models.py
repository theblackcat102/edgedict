import torch
import torch.nn.functional as F
from torch import nn
import math
from dataclasses import dataclass, field
from typing import List, Tuple

try:
    from warprnnt_pytorch import RNNTLoss
except:
    pass

from rnnt.tokenizer import NUL, BOS, PAD
from modules.group_norm import Fp32GroupNorm

class TimeReduction(nn.Module):
    def __init__(self, reduction_factor=2):
        super().__init__()
        self.reduction_factor = reduction_factor

    def forward(self, xs):
        batch_size, xlen, hidden_size = xs.shape
        pad = self.reduction_factor - xlen % self.reduction_factor
        pad = pad % self.reduction_factor
        pad_shape = [0, 0, 0, pad, 0, 0]
        xs = nn.functional.pad(xs, pad_shape)
        xs = xs.reshape(batch_size, -1, self.reduction_factor, hidden_size)
        xs = xs.mean(dim=-2)
        return xs


class ResLayerNormLSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 dropout=0,
                 time_reductions=[1],
                 reduction_factor=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstms = nn.ModuleList()
        self.projs = nn.ModuleList()
        for i in range(num_layers):
            self.lstms.append(
                nn.LSTM(input_size, hidden_size, 1, batch_first=True))
            proj = [nn.LayerNorm(hidden_size)]
            if i in time_reductions:
                proj.append(TimeReduction(reduction_factor))
            if dropout > 0:
                proj.append(nn.Dropout(dropout))
            input_size = hidden_size
            self.projs.append(nn.Sequential(*proj))

    def forward(self, xs, hiddens=None):
        if hiddens is None:
            hs = xs.new_zeros(len(self.lstms), xs.shape[0], self.hidden_size)
            cs = xs.new_zeros(len(self.lstms), xs.shape[0], self.hidden_size)
        else:
            hs, cs = hiddens
        new_hs = []
        new_cs = []
        for i, (lstm, proj) in enumerate(zip(self.lstms, self.projs)):
            lstm.flatten_parameters()
            xs_next, (h, c) = lstm(xs, (hs[i, None], cs[i, None]))
            if i != 0:
                xs = xs + xs_next
            else:
                xs = xs_next
            xs = proj(xs)
            new_hs.append(h)
            new_cs.append(c)
        hs = torch.cat(new_hs, dim=0)
        cs = torch.cat(new_cs, dim=0)
        return xs, (hs, cs)

class ResLayerNormGRU(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 dropout=0,
                 time_reductions=[1],
                 reduction_factor=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstms = nn.ModuleList()
        self.projs = nn.ModuleList()
        for i in range(num_layers):
            self.lstms.append(
                nn.GRU(input_size, hidden_size, 1, batch_first=True))
            proj = [nn.LayerNorm(hidden_size)]
            if i in time_reductions:
                proj.append(TimeReduction(reduction_factor))
            if dropout > 0:
                proj.append(nn.Dropout(dropout))
            input_size = hidden_size
            self.projs.append(nn.Sequential(*proj))

    def forward(self, xs, hiddens=None):
        if hiddens is None:
            hs = xs.new_zeros(len(self.lstms), xs.shape[0], self.hidden_size)
        else:
            hs = hiddens
        new_hs = []
        for i, (lstm, proj) in enumerate(zip(self.lstms, self.projs)):
            lstm.flatten_parameters()
            xs_next, h = lstm(xs, hs[i, None])
            if i != 0:
                xs = xs + xs_next
            else:
                xs = xs_next
            xs = proj(xs)
            new_hs.append(h)
        hs = torch.cat(new_hs, dim=0)
        return xs, hs


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout,
                 proj_size, module=ResLayerNormGRU,
                 time_reductions=[1], has_proj=True):
        super().__init__()
        self.norm = nn.LayerNorm(input_size)
        self.lstm = module(
            input_size, hidden_size, num_layers, dropout=dropout, time_reductions=time_reductions)
        self.has_proj = has_proj
        if has_proj:
            self.proj = nn.Linear(hidden_size, proj_size)

    def forward(self, xs, hiddens=None):
        xs = self.norm(xs)
        xs, hiddens = self.lstm(xs, hiddens)
        if self.has_proj:
            xs = self.proj(xs)
        return xs, hiddens


class Decoder(nn.Module):
    def __init__(self, vocab_embed_size, vocab_size, hidden_size, num_layers,
                 dropout=0, proj_size=None):
        super().__init__()
        self.embed = nn.Embedding(
            vocab_size, vocab_embed_size, padding_idx=PAD)
        self.lstm = nn.LSTM(
            vocab_embed_size, hidden_size, num_layers,
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
                 vocab_embed_size, vocab_size, input_size,
                 enc_hidden_size, enc_layers, enc_dropout, enc_proj_size,
                 dec_hidden_size, dec_layers, dec_dropout, dec_proj_size,
                 joint_size, enc_time_reductions=[1],
                 blank=NUL, module_type='LSTM', output_loss=True):
        super().__init__()
        self.blank = blank
        # Encoder
        if module_type not in ['GRU', 'LSTM']:
            raise ValueError('Unsupported module type')
        if module_type == 'GRU':
            module = ResLayerNormGRU
        else:
            module = ResLayerNormLSTM
        self.encoder = Encoder(
            input_size=input_size,
            hidden_size=enc_hidden_size,
            num_layers=enc_layers,
            dropout=enc_dropout,
            proj_size=enc_proj_size,
            time_reductions=enc_time_reductions,
            module=module)
        # Decoder
        self.decoder = Decoder(
            vocab_embed_size=vocab_embed_size,
            vocab_size=vocab_size,
            hidden_size=dec_hidden_size,
            num_layers=dec_layers,
            dropout=dec_dropout,
            proj_size=dec_proj_size)
        # Joint
        self.joint = Joint(
            input_size=enc_proj_size + dec_proj_size,
            hidden_size=joint_size,
            vocab_size=vocab_size)
        self.output_loss = output_loss
        if output_loss:
            self.loss_fn = RNNTLoss(blank=blank)

    def scale_length(self, logits, xlen):
        scale = (xlen.max().float() / logits.shape[1]).ceil()
        xlen = (xlen / scale).ceil().int()
        return xlen

    def forward(self, xs, ys, xlen, ylen):
        xs = xs[:, :xlen.max()].contiguous()
        ys = ys[:, :ylen.max()].contiguous()

        h_enc, _ = self.encoder(xs)
        h_dec, _ = self.decoder(ys)
        logits = self.joint(h_enc, h_dec)

        if self.output_loss:
            xlen = self.scale_length(logits, xlen)
            loss = self.loss_fn(logits, ys, xlen, ylen)
            return loss

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


def CausalConv1d(in_channels, out_channels, kernel_size, dilation=1, **kwargs):
    pad = (kernel_size - 1) * dilation
    conv  = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation, **kwargs)
    nn.init.kaiming_normal_(conv.weight)
    return conv

class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, group_norm_size=1, **kwargs):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation, **kwargs)

        nn.init.kaiming_normal_(self.conv.weight)
        self.gn = nn.GroupNorm( group_norm_size, in_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(x)
        x = self.gn(x)
        x = self.conv(x)
        x = x[:, :, :-self.conv.padding[0]]
        return x

class FrontEnd(nn.Module):
    def __init__(self, frontend_params = [(10, 5, 16)]+[ (8, 4, 32) ]+[(4,2,128)]*3, bias=True):

        super().__init__()
        kernel_sizes=[ p[0] for p in frontend_params]
        strides=[ p[1] for p in frontend_params] 
        channels=[ p[2] for p in frontend_params]

        assert len(kernel_sizes) == len(strides)

        self.conv1 = CausalConv1d(1, channels[0], kernel_size=kernel_sizes[0], dilation=1, stride=strides[0], bias=bias)

        self.encode = nn.Sequential(*[
            DilatedConvBlock(channels[idx-1], channels[idx], kernel_size=kernel_sizes[idx], dilation=1, stride=strides[idx], group_norm_size=1, bias=bias)
            for idx in range(1, len(strides))
        ])
        self.layer_norm = nn.LayerNorm(frontend_params[-1][-1])

    def forward(self, x):

        if len(x.shape) < 3: # B x T -> B x 1 x T
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = x[:, :, :-self.conv1.padding[0]]
        features = self.encode(x)
        # convert B x C x T -> B x T x C
        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        return features


def convert_lightning2normal(checkpoint):
    keys = checkpoint.keys()

    if 'state_dict' in keys:
        checkpoint = checkpoint['state_dict']
        keys = checkpoint.keys()
        if 'model.' in list(keys)[0]:
            new_checkpoint = {}
            for key, value in checkpoint.items():
                new_checkpoint[key.replace('model.', '')] = value
            checkpoint = {'model': new_checkpoint}

    return checkpoint


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):

        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        return x


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
        input_size=128,
        enc_layers=2,
        dec_layers=1,
        enc_proj_size=256,
        enc_hidden_size=128,
        dec_hidden_size=128,
        dec_proj_size=124,
        dec_dropout=0.1,
        enc_dropout=0.1,
        joint_size=10)
    xs = torch.randn((1, 50, 40)).float()
    xlen = torch.ones(1).int() * 50

    frontend = FrontEnd()
    signals = torch.randn((32, 1, 5000)).float()
    outputs = frontend(signals)
    xs = outputs.permute(0, 2, 1)
    xlen = torch.ones(1).int() * outputs.shape[1]
    print(outputs.shape)
    seqs, prob = model.greedy_decode(xs, xlen)
    print(seqs[0], prob[0])
