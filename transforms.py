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


class SpectrogramAugmentation(nn.Module):
    """Spectrogram augmentation
    """
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.spec_cutout_regions = SpecCutoutRegions(kwargs)
        self.spec_augment = SpecAugment(kwargs)

    @torch.no_grad()
    def forward(self, input_spec):
        augmented_spec = self.spec_cutout_regions(input_spec)
        augmented_spec = self.spec_augment(augmented_spec)
        return augmented_spec


class SpecAugment(nn.Module):
    """Spec augment. refer to https://arxiv.org/abs/1904.08779
    """
    def __init__(self, cfg):
        super(SpecAugment, self).__init__()
        self.cutout_x_regions = cfg.get('cutout_x_regions', 0)
        self.cutout_y_regions = cfg.get('cutout_y_regions', 0)

        self.cutout_x_width = cfg.get('cutout_x_width', 10)
        self.cutout_y_width = cfg.get('cutout_y_width', 10)

    @torch.no_grad()
    def forward(self, x):
        sh = x.shape

        mask = torch.zeros(x.shape).byte()
        for idx in range(sh[0]):
            for _ in range(self.cutout_x_regions):
                cutout_x_left = int(
                    random.uniform(0, sh[1] - self.cutout_x_width))

                mask[idx,
                     cutout_x_left:cutout_x_left + self.cutout_x_width, :] = 1

            for _ in range(self.cutout_y_regions):
                cutout_y_left = int(
                    random.uniform(0, sh[2] - self.cutout_y_width))

                mask[idx, :,
                     cutout_y_left:cutout_y_left + self.cutout_y_width] = 1

        x = x.masked_fill(mask.to(device=x.device), 0)

        return x


class SpecCutoutRegions(nn.Module):
    """Cutout. refer to https://arxiv.org/pdf/1708.04552.pdf
    """
    def __init__(self, cfg):
        super(SpecCutoutRegions, self).__init__()

        self.cutout_rect_regions = cfg.get('cutout_rect_regions', 0)
        self.cutout_rect_time = cfg.get('cutout_rect_time', 5)
        self.cutout_rect_freq = cfg.get('cutout_rect_freq', 20)

    @torch.no_grad()
    def forward(self, x):
        sh = x.shape

        mask = torch.zeros(x.shape, dtype=torch.uint8)

        for idx in range(sh[0]):
            for i in range(self.cutout_rect_regions):
                cutout_rect_x = int(random.uniform(
                        0, sh[1] - self.cutout_rect_freq))
                cutout_rect_y = int(random.uniform(
                        0, sh[2] - self.cutout_rect_time))

                mask[idx,
                     cutout_rect_x:cutout_rect_x + self.cutout_rect_freq,
                     cutout_rect_y:cutout_rect_y + self.cutout_rect_time] = 1

        x = x.masked_fill(mask.to(device=x.device), 0)

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
