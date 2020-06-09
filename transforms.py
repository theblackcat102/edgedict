import random

import torch
import torchaudio
import torch.nn as nn

from parts.features import FeatureFactory


class AudioPreprocessing(nn.Module):
    """GPU accelerated audio preprocessing
    """
    __constants__ = ["optim_level"]

    def __init__(self, **kwargs):
        nn.Module.__init__(self)    # For PyTorch API
        self.optim_level = kwargs.get('optimization_level', 0)
        self.featurizer = FeatureFactory.from_config(kwargs)
        self.transpose_out = kwargs.get("transpose_out", False)

    @torch.no_grad()
    def forward(self, input_signal, length):
        processed_signal = self.featurizer(input_signal, length)
        processed_length = self.featurizer.get_seq_len(length)
        if self.transpose_out:
            processed_signal.transpose_(2, 1)
            return processed_signal, processed_length
        else:
            return processed_signal, processed_length


class SpecAugment(nn.Module):
    """Spec augment. refer to https://arxiv.org/abs/1904.08779
    """
    def __init__(self,
                 time_regions=0, time_width=10,
                 freq_regions=0, freq_width=10):
        super(SpecAugment, self).__init__()
        self.time_regions = time_regions
        self.freq_regions = freq_regions
        self.time_width = time_width
        self.freq_width = freq_width

    @torch.no_grad()
    def forward(self, x):
        shape = x.shape
        mask = x.new_zeros(x.shape, dtype=torch.uint8)

        for idx in range(shape[0]):
            for _ in range(self.time_regions):
                time_idx = int(
                    random.uniform(0, shape[1] - self.time_width))
                mask[idx, time_idx:time_idx + self.time_width, :] = 1

            for _ in range(self.freq_regions):
                freq_idx = int(
                    random.uniform(0, shape[2] - self.freq_width))
                mask[idx, :, freq_idx:freq_idx + self.freq_width] = 1

        x = x.masked_fill(mask, 0)

        return x


class SpecCutoutRegions(nn.Module):
    """Cutout. refer to https://arxiv.org/pdf/1708.04552.pdf
    """
    def __init__(self, rect_regions=0, rect_time=5, rect_freq=20):
        super(SpecCutoutRegions, self).__init__()

        self.rect_regions = rect_regions
        self.rect_time = rect_time
        self.rect_freq = rect_freq

    @torch.no_grad()
    def forward(self, x):
        shape = x.shape
        mask = x.new_zeros(x.shape, dtype=torch.uint8)

        for idx in range(shape[0]):
            for i in range(self.rect_regions):
                time_idx = int(random.uniform(0, shape[1] - self.rect_time))
                freq_idx = int(random.uniform(0, shape[2] - self.rect_freq))

                mask[idx,
                     time_idx:time_idx + self.rect_time,
                     freq_idx:freq_idx + self.rect_freq] = 1

        x = x.masked_fill(mask, 0)

        return x


class KaldiMFCC(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    @torch.no_grad()
    def forward(self, wave_form):
        feat = torchaudio.compliance.kaldi.mfcc(
            wave_form, channel=0, **self.kwargs)
        return feat


class CatDeltas(torch.nn.Module):
    def forward(self, feat):
        d1 = torchaudio.functional.compute_deltas(feat.T)
        d2 = torchaudio.functional.compute_deltas(d1)
        feat = torch.cat([feat, d1.T, d2.T], dim=-1)
        return feat


class CMVN(torch.nn.Module):
    eps = 1e-10

    @torch.no_grad()
    def forward(self, feat):
        mean = feat.mean(0, keepdim=True)
        std = feat.std(0, keepdim=True)
        feat = (feat - mean) / (std + CMVN.eps)
        return feat


class Log(torch.nn.Module):
    @torch.no_grad()
    def forward(self, mel_spec):
        log_mel_spec = torch.log(mel_spec + 1e-8)
        return log_mel_spec


class Downsample(torch.nn.Module):
    def __init__(self, n_frame):
        super().__init__()
        self.n_frame = n_frame

    @torch.no_grad()
    def forward(self, feat):
        feat_length, feat_size = feat.shape
        pad = (self.n_frame - feat_length % self.n_frame) % self.n_frame
        pad_shape = [0, 0, 0, pad]
        feat = torch.nn.functional.pad(feat, pad_shape)
        feat = feat.reshape(-1, feat_size * self.n_frame)

        return feat


class Transpose(torch.nn.Module):
    @torch.no_grad()
    def forward(self, feat):
        return feat[0].T
