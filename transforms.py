import torch
import torchaudio


class KaldiMFCC(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, wave_form):
        feat = torchaudio.compliance.kaldi.mfcc(
            wave_form[None], channel=0, **self.kwargs)
        return feat


class CatDeltas(torch.nn.Module):
    def forward(self, feat):
        d1 = torchaudio.functional.compute_deltas(feat)
        d2 = torchaudio.functional.compute_deltas(d1)
        feat = torch.cat([feat, d1, d2], dim=-1)
        return feat


class CMVN(torch.nn.Module):
    eps = 1e-10

    def forward(self, feat):
        mean = feat.mean(0, keepdim=True)
        std = feat.std(0, keepdim=True)
        feat = (feat - mean) / (std + CMVN.eps)
        return feat


class Log(torch.nn.Module):
    def forward(self, mel_spec):
        log_mel_spec = torch.log(mel_spec + 1e-8)
        return log_mel_spec


class Downsample(torch.nn.Module):
    def __init__(self, n_frame):
        super().__init__()
        self.n_frame = n_frame

    def forward(self, feat):
        feat_length, feat_size = feat.shape
        feat_length = (feat_length // self.n_frame) * self.n_frame
        feat_sampled = feat[:feat_length, :]
        feat_sampled = feat_sampled.reshape(feat_size * self.n_frame, -1)
        return feat_sampled


class Transpose(torch.nn.Module):
    def forward(self, feat):
        return feat.T
