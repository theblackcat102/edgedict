import torch


class FreqNormalize(torch.nn.Module):
    def forward(self, mel_spec):
        mel_spec = mel_spec - mel_spec.mean(dim=-1, keepdim=True)
        return mel_spec


class Log(torch.nn.Module):
    """
    ref: https://github.com/noahchalifour/rnnt-speech-recognition/blob/a0d972f5e407e465ad784c682fa4e72e33d8eefe/utils/preprocessing.py#L48
    """
    def forward(self, mel_spec):
        log_mel_spec = torch.log(mel_spec + 1e-8)
        return log_mel_spec


class Downsample(torch.nn.Module):
    def __init__(self, n_frame):
        self.n_frame = n_frame

    def __call__(self, spec):
        feat_size, spec_length = spec.shape
        spec_length = (spec_length // self.n_frame) * self.n_frame
        spec_sampled = spec[:, :spec_length]
        spec_sampled = spec_sampled.reshape(feat_size * self.n_frame, -1)
        return spec_sampled
