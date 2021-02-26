import random

import torch
import torchaudio
from torchaudio.transforms import MFCC, MelSpectrogram

from rnnt.features import FilterbankFeatures


class CatDeltas(torch.nn.Module):
    @torch.no_grad()
    def forward(self, feat):
        d1 = torchaudio.functional.compute_deltas(feat)
        d2 = torchaudio.functional.compute_deltas(d1)
        feat = torch.cat([feat, d1, d2], dim=1)
        return feat


class CMVN(torch.nn.Module):
    eps = 1e-5

    @torch.no_grad()
    def forward(self, feat):
        mean = feat.mean(dim=2, keepdim=True)
        std = feat.std(dim=2, keepdim=True)
        feat = (feat - mean) / (std + CMVN.eps)
        return feat


class Downsample(torch.nn.Module):

    def __init__(self, n_frame, pad_to_divisible=True):
        super().__init__()
        self.n_frame = n_frame
        self.pad_to_divisible = pad_to_divisible

    @torch.no_grad()
    def forward(self, feat):
        feat = feat.transpose(1, 2)
        batch_size, feat_length, feat_size = feat.shape
        if self.pad_to_divisible:
            pad = (self.n_frame - feat_length % self.n_frame) % self.n_frame
            pad_shape = [0, 0, 0, pad, 0, 0]
            feat = torch.nn.functional.pad(feat, pad_shape)
        else:
            feat_length = feat_length - feat_length % self.n_frame
            feat = feat[:, :feat_length, :]

        feat = feat.reshape(batch_size, -1, feat_size * self.n_frame)

        return feat.transpose(1, 2)


class FrequencyMasking(torch.nn.Module):
    """
    Implements frequency masking transform from SpecAugment paper
    (https://arxiv.org/abs/1904.08779)

      Example:
        >>> transforms.Compose([
        >>>     transforms.ToTensor(),
        >>>     FrequencyMasking(max_width=10, num_masks=1, use_mean=False),
        >>> ])

    """

    def __init__(self, max_width, num_masks, use_mean=False):
        super().__init__()
        self.max_width = max_width
        self.num_masks = num_masks
        self.use_mean = use_mean

    def forward(self, x):
        """
        Args:
            x (Tensor): Tensor image of size (N, T, H) where the frequency
                mask is to be applied.

        Returns:
            Tensor: Transformed image with Frequency Mask.
        """
        if self.use_mean:
            fill_value = x.mean()
        else:
            fill_value = 0
        mask = x.new_zeros(x.shape).bool()
        for i in range(x.shape[0]):
            for _ in range(self.num_masks):
                start = random.randrange(0, x.shape[1])
                end = start + random.randrange(0, self.max_width)
                mask[i, start:end, :] = 1
        x = x.masked_fill(mask, value=fill_value)
        return x

    def __repr__(self):
        format_string = "%s(max_width=%d,num_masks=%d,use_mean=%s)" % (
            self.__class__.__name__, self.max_width, self.num_masks,
            self.use_mean)
        return format_string


class TimeMasking(torch.nn.Module):
    """
    Implements time masking transform from SpecAugment paper
    (https://arxiv.org/abs/1904.08779)

      Example:
        >>> transforms.Compose([
        >>>     transforms.ToTensor(),
        >>>     TimeMasking(max_width=10, num_masks=2, use_mean=False),
        >>> ])

    """

    def __init__(self, max_width, num_masks, use_mean=False):
        super().__init__()
        self.max_width = max_width
        self.num_masks = num_masks
        self.use_mean = use_mean

    def forward(self, x):
        """
        Args:
            x (Tensor): Tensor image of size (N, T, H) where the time mask is
                to be applied.

        Returns:
            Tensor: Transformed image with Time Mask.
        """
        if self.use_mean:
            fill_value = x.mean()
        else:
            fill_value = 0
        mask = x.new_zeros(x.shape).bool()
        for i in range(x.shape[0]):
            for _ in range(self.num_masks):
                start = random.randrange(0, x.shape[2])
                end = start + random.randrange(0, self.max_width)
                mask[i, :, start:end] = 1
        x = x.masked_fill(mask, value=fill_value)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + "(max_width="
        format_string += str(self.max_width) + ")"
        return format_string


class TrimAudio(torch.nn.Module):
    '''Trim raw audio into the maximum length
        sampling_rate: sampling rate for audio raw signal, default 16800
        max_audio_length: maximum audio length in seconds
    '''
    def __init__(self, sampling_rate=16800, max_audio_length=10, truncate_end=True):
        super().__init__()
        self.truncate_end = truncate_end
        self.max_length = int(sampling_rate*max_audio_length)

    def forward(self, x):

        if self.truncate_end:
            return x[:, :self.max_length]
        return x[:, -self.max_length:]

def build_transform(feature_type, feature_size, n_fft=512, win_length=400,
                    hop_length=200, delta=False, cmvn=False, downsample=1,
                    T_mask=0, T_num_mask=0, F_mask=0, F_num_mask=0,
                    pad_to_divisible=True):
    feature_args = {
        'n_fft': n_fft,
        'win_length': win_length,
        'hop_length': hop_length,
        # 'f_min': 20,
        # 'f_max': 5800,
    }
    transform = []
    input_size = feature_size
    if feature_type == 'mfcc':
        transform.append(MFCC(
            n_mfcc=feature_size, log_mels=True, melkwargs=feature_args))
    if feature_type == 'melspec':
        transform.append(MelSpectrogram(
            n_mels=feature_size, **feature_args))
    if feature_type == 'logfbank':
        transform.append(FilterbankFeatures(
            n_filt=feature_size, **feature_args))
    if delta:
        transform.append(CatDeltas())
        input_size = input_size * 3
    # if cmvn:
    #     transform.append(CMVN())
    if downsample > 1:
        transform.append(Downsample(downsample, pad_to_divisible))
        input_size = input_size * downsample
    transform_test = torch.nn.Sequential(*transform)

    if T_mask > 0 and T_num_mask > 0:
        transform.append(TimeMasking(T_mask, T_num_mask))
    if F_mask > 0 and F_num_mask > 0:
        transform.append(FrequencyMasking(F_mask, F_num_mask))
    transform_train = torch.nn.Sequential(*transform)

    return transform_train, transform_test, input_size
