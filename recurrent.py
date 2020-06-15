import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from torch import Tensor
from torch import nn, autograd
import torch.nn.functional as F
from torchaudio import functional as F_audio
import math
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torchaudio

from torchaudio.compliance.kaldi import mfcc
from speechpy.processing import cmvn, cmvnw

def fast_tanh(x):
    return x / (1 + x.abs())

class FastTanh(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x / (1 + x.abs())

class TimeReduction(nn.Module):
    def __init__(self, reduction_factor):
        super().__init__()
        self.reduction_factor = reduction_factor

    def forward(self, xs):
        batch_size, xlen, hidden_size = xs.shape
        reduction_factor = xlen % self.reduction_factor
        if reduction_factor != 0:
            reduction_factor =  self.reduction_factor - reduction_factor
        pad_shape = [0, 0, 0, reduction_factor, 0, 0]
        xs = nn.functional.pad(xs, pad_shape)
        xs = xs.view(batch_size, -1, self.reduction_factor, hidden_size)
        xs = xs.mean(dim=2)
        return xs


class MFCC_(torch.nn.Module):
    r"""Create the Mel-frequency cepstrum coefficients from an audio signal.

    By default, this calculates the MFCC on the DB-scaled Mel spectrogram.
    This is not the textbook implementation, but is implemented here to
    give consistency with librosa.

    This output depends on the maximum value in the input spectrogram, and so
    may return different values for an audio clip split into snippets vs. a
    a full clip.

    Args:
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        n_mfcc (int, optional): Number of mfc coefficients to retain. (Default: ``40``)
        dct_type (int, optional): type of DCT (discrete cosine transform) to use. (Default: ``2``)
        norm (str, optional): norm to use. (Default: ``'ortho'``)
        log_mels (bool, optional): whether to use log-mel spectrograms instead of db-scaled. (Default: ``False``)
        melkwargs (dict or None, optional): arguments for MelSpectrogram. (Default: ``None``)
    """
    __constants__ = ['sample_rate', 'n_mfcc', 'dct_type', 'top_db', 'log_mels']

    def __init__(self,
                 sample_rate: int = 16000,
                 n_mfcc: int = 40,
                 dct_type: int = 2,
                 norm: str = 'ortho',
                 log_mels: bool = False,
                 normalize=False,
                 melkwargs= None) -> None:
        super(MFCC_, self).__init__()
        supported_dct_types = [2]
        if dct_type not in supported_dct_types:
            raise ValueError('DCT type not supported'.format(dct_type))
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.dct_type = dct_type
        self.norm = norm
        self.normalize = normalize
        self.top_db = 80.0
        stype = 'power'
        self.multiplier = 10.0 if stype == 'power' else 20.0
        self.amin = 1e-10
        self.ref_value = 1.0
        self.db_multiplier = math.log10(max(self.amin, self.ref_value))
        # self.amplitude_to_DB = AmplitudeToDB('power', self.top_db)

        if melkwargs is not None:
            self.MelSpectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate, **melkwargs)
        else:
            self.MelSpectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate)

        if self.n_mfcc > self.MelSpectrogram.n_mels:
            raise ValueError(
                'Cannot select more MFCC coefficients than # mel bins')
        dct_mat = F_audio.create_dct(
            self.n_mfcc, self.MelSpectrogram.n_mels, self.norm)
        self.register_buffer('dct_mat', dct_mat)
        self.log_mels = log_mels

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: specgram_mel_db of size (..., ``n_mfcc``, time).
        """

        # pack batch
        shape = waveform.size()
        waveform = waveform.view(-1, shape[-1])

        mel_specgram = self.MelSpectrogram(waveform)
        if self.log_mels:
            log_offset = 1e-6
            mel_specgram = torch.log(mel_specgram + log_offset)
        else:
            mel_specgram = F_audio.amplitude_to_DB(mel_specgram, self.multiplier, self.amin, self.db_multiplier, self.top_db)
        # (channel, n_mels, time).tranpose(...) dot (n_mels, n_mfcc)
        # -> (channel, time, n_mfcc).tranpose(...)
        mfcc = torch.matmul(mel_specgram.transpose(1, 2),
                            self.dct_mat).transpose(1, 2)

        # unpack batch
        mfcc = mfcc.view(shape[:-1] + mfcc.shape[-2:])

        if self.normalize:

            mfcc = torch.from_numpy(cmvnw(mfcc.numpy().T, win_size=201)).T
            # mean_vec = mfcc.mean(dim=1)
            # mfcc = torch.sub(mfcc, mean_vec[:, None])

        return mfcc


class ResidualRNNModel(nn.Module):
    def __init__(self, input_size, vocab_size, hidden_size, num_layers, dropout=.2, blank=0, bidirectional=False):
        super(ResidualRNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.blank = blank
        self.layer_norm = nn.LayerNorm(input_size)
        stacked = StackedRecurrent(residual=True, merge_first=True)
        pre_lstm = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
        stacked.add_module('0', pre_lstm)

        for i in range(num_layers-1):
            layer = nn.LSTM(hidden_size, hidden_size, 1, batch_first=True)
            stacked.add_module(str(i+1), layer)
            stacked.add_module(str(i+1)+'1', nn.LayerNorm(hidden_size))

        # normalize spectrum feature
        # lstm hidden vector: (h_0, c_0) num_layers * num_directions, batch, hidden_size
        # nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.lstm = stacked
        # if bidirectional: hidden_size *= 2
        self.linear = None
        if vocab_size == hidden_size:
            self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, xs, hid=None):
        xs = self.layer_norm(xs)
        h, hid = self.lstm(xs, hid)
        if self.linear != None:
            h = self.linear(h)
        return h, hid

    def greedy_decode(self, xs):
        xs = self(xs)[0][0]  # only one sequence
        xs = F.log_softmax(xs, dim=1)
        logp, pred = torch.max(xs, dim=1)
        return pred.data.cpu().numpy(), -float(logp.sum())

    def beam_search(self, xs, W):
        ''' CTC '''
        xs = self(xs)[0][0]  # only one sequence
        logp = F.log_softmax(xs, dim=1)
        return ctc_beam(logp.data.cpu().numpy(), W)


class ResidualProjModel(nn.Module):
    def __init__(self, input_size, vocab_size, hidden_size,num_layers, ff_dim=-1, dropout=.2, blank=0, bidirectional=False):
        super(ResidualProjModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.blank = blank
        if ff_dim == -1:
            ff_dim = hidden_size//2
        stacked = StackedRecurrent(residual=True, merge_first=True)

        pre_lstm = nn.Sequential(
            nn.LSTM(input_size, hidden_size, 1, batch_first=True),
            nn.Linear(hidden_size, ff_dim),
            FastTanh(),
        )
        stacked.add_module('0', pre_lstm)

        for i in range(num_layers-1):
            layer = nn.Sequential(
                nn.Linear(ff_dim, hidden_size),
                nn.LSTM(hidden_size, hidden_size, 1, batch_first=True),
                nn.Linear(hidden_size, ff_dim),
                FastTanh(),
            )
            stacked.add_module(str(i+1), layer)

        # normalize spectrum feature
        # lstm hidden vector: (h_0, c_0) num_layers * num_directions, batch, hidden_size
        # nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.lstm = stacked
        # if bidirectional: hidden_size *= 2
        self.linear = None
        if vocab_size == hidden_size:
            self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, xs, hid=None):
        h, hid = self.lstm(xs, hid)
        if self.linear != None:
            h = self.linear(h)
        return h, hid

    def greedy_decode(self, xs):
        xs = self(xs)[0][0]  # only one sequence
        xs = F.log_softmax(xs, dim=1)
        logp, pred = torch.max(xs, dim=1)
        return pred.data.cpu().numpy(), -float(logp.sum())

    def beam_search(self, xs, W):
        ''' CTC '''
        xs = self(xs)[0][0]  # only one sequence
        logp = F.log_softmax(xs, dim=1)
        return ctc_beam(logp.data.cpu().numpy(), W)

class StackedRecurrent(nn.Sequential):

    def __init__(self, dropout=0, residual=False, normalization=False, merge_first=True, reduction_factor=2):
        super(StackedRecurrent, self).__init__()
        self.residual = residual
        self.dropout = dropout
        self.normalization = normalization
        if merge_first:
            self.concat = TimeReduction(reduction_factor=reduction_factor)
        self.merge_first = merge_first

    def forward(self, inputs, hidden=None):
        hidden = hidden or tuple([None] * len(self))
        next_hidden = []
        hidden_idx = 0

        for i, module in enumerate(self._modules.values()):

            if isinstance(module, TimeReduction):
                continue
            elif i == 4 and self.merge_first:
                inputs = self.concat(inputs)
            elif isinstance(module, nn.LayerNorm):
                inputs = module(inputs)
                continue

            output, h = module(inputs, hidden[hidden_idx])
            hidden_idx += 1
            next_hidden.append(h)
            if self.residual and inputs.size(-1) == output.size(-1):
                inputs = output + inputs
            else:
                inputs = output
            if isinstance(inputs, PackedSequence):
                data = nn.functional.dropout(
                    inputs.data, self.dropout, self.training)
                inputs = PackedSequence(data, inputs.batch_sizes)
            else:
                inputs = nn.functional.dropout(
                    inputs, self.dropout, self.training)

        return output, tuple(next_hidden)


class NormalizationLayer(nn.Module):
    def __init__(self, num_features):
        super(NormalizationLayer, self).__init__()
        self.norm = nn.InstanceNorm1d(num_features)

    def forward(self, inputs, hidden=None):
        x = inputs.permute(0, 2, 1)
        x = self.norm(x)
        return x.permute(0, 2, 1)


class ConcatFeature(torch.nn.Module):

    def __init__(self, merge_size=3):
        super(ConcatFeature, self).__init__()
        self.merge_size = merge_size

    def forward(self, waveform: Tensor) -> Tensor:
        batch_size, waveform_len, feat = waveform.shape
        if waveform_len % self.merge_size != 0:
            pad_wave = torch.zeros(
                (batch_size, self.merge_size - (waveform_len % self.merge_size), feat))
            if waveform.is_cuda:
                pad_wave = pad_wave.to(waveform.get_device())
            waveform = torch.cat([waveform, pad_wave], dim=1)

        return waveform.reshape(batch_size, -1, feat*self.merge_size)


if __name__ == "__main__":
    import numpy as np
    import pickle
    # pickle.dump(MFCC_(), open('test.pt', 'wb'))
    import torchaudio

    trans = MFCC_(normalize=True, log_mels=True)
    data, sr = torchaudio.load('bloom.mp3', normalization=True)
    print(data[0].shape)
    mfcc_f = trans(data[0])
    print(mfcc_f.shape)
    print(torch.mean(mfcc_f), torch.var(mfcc_f))
    print('first feature')
    print(torch.mean(mfcc_f[:, 0]), torch.var(mfcc_f))
    print('feature 0')
    print(mfcc_f[0].shape,torch.mean(mfcc_f[0]), torch.var(mfcc_f))

    # stacked = StackedRecurrent(residual=True, merge_first=True)

    # cell = nn.LSTM(40, 64, batch_first=True)
    # stacked.add_module('0',cell)
    # cell = nn.LSTM(64*3, 64*3, batch_first=True)
    # stacked.add_module('1',cell)
    # cell = nn.LSTM(64*3, 64*3, batch_first=True)
    # stacked.add_module('2',cell)

    # inputs = torch.randn((32, 10, 40))
    # outputs, hid = stacked(inputs)
    # inputs = torch.randn((32, 10, 40))
    # outputs, hid = stacked(inputs)
    inputs = torch.randn((32, 10, 40))
    hidden_size = 128
    num_layers = 4
    input_size = 40
    # stacked = StackedRecurrent(residual=True, merge_first=True)

    # pre_lstm = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
    # stacked.add_module('0', pre_lstm)
    # _lstm = nn.LSTM((hidden_size//3)*3, hidden_size, 1, batch_first=True)
    # stacked.add_module('1', _lstm)

    # for i in range(num_layers):
    #     layer = nn.LSTM(hidden_size, hidden_size, 1, batch_first=True)
    #     stacked.add_module(str(i+1), layer)
    # stacked(inputs)

    import torch
    from torch import nn

    a = torch.randn(32, 100, 1)  
    m = nn.Conv1d(100, 100, 1, stride=2) 
    out = m(a)
    print(out.size())
    print(m)
    # model = ResidualRNNModel(40, 3600, 128, 4)
    # outputs, hid = model(inputs)
    # print(outputs.shape)
    # concat = ConcatFeature()
    # x = torch.from_numpy(np.array([
    #     [
    #         list([1]*4),
    #         list([2]*4),
    #         list([3]*4),
    #         list([4]*4),
    #     ],
    #     [
    #         list([4]*4),
    #         list([5]*4),
    #         list([5]*4),
    #         list([5]*4),
    #     ]
    # ], dtype=float)).float()
    # print(x.shape)
    # x = concat(x)
    # print(x)
    # print(x.shape)
