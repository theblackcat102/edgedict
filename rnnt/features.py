import torch
import torch.nn as nn
import math
import librosa


def normalize_batch(x, seq_len, normalize_type: str):
    if normalize_type == "per_feature":
        assert not torch.isnan(x).any(), x
        x_mean = torch.zeros(
            (seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
        x_std = torch.zeros(
            (seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            x_mean[i, :] = x[i, :, :seq_len[i]].mean(dim=1)
            x_std[i, :] = x[i, :, :seq_len[i]].std(dim=1)
        # make sure x_std is not zero
        x_std += 1e-5
        return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)
    elif normalize_type == "all_features":
        x_mean = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        x_std = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            x_mean[i] = x[i, :, :int(seq_len[i])].mean()
            x_std[i] = x[i, :, :int(seq_len[i])].std()
        # make sure x_std is not zero
        x_std += 1e-5
        return (x - x_mean.view(-1, 1, 1)) / x_std.view(-1, 1, 1)
    else:
        return x


class FilterbankFeatures(nn.Module):
    # For JIT. See
    # https://pytorch.org/docs/stable/jit.html#python-defined-constants
    __constants__ = ["dither", "preemph", "n_fft", "hop_length", "win_length",
                     "center", "log", "frame_splicing", "window", "normalize",
                     "pad_to", "max_duration", "max_length"]

    def __init__(self,
                 sample_rate=16000,
                 win_length=320,
                 hop_length=160,
                 n_fft=512,
                 window="hann",
                 normalize="none",
                 log=True,
                 dither=1e-5,
                 pad_to=0,
                 max_duration=16.7,
                 preemph=0.97,
                 n_filt=64,
                 f_min=0,
                 f_max=None):
        super(FilterbankFeatures, self).__init__()

        torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'none': None,
        }

        self.win_length = win_length    # frame size
        self.hop_length = hop_length
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))

        self.normalize = normalize
        self.log = log
        # TORCHSCRIPT: Check whether or not we need this
        self.dither = dither
        self.n_filt = n_filt
        self.preemph = preemph
        self.pad_to = pad_to
        f_max = f_max or sample_rate / 2
        window_fn = torch_windows.get(window, None)
        window_tensor = window_fn(self.win_length,
                                  periodic=False) if window_fn else None
        filterbanks = torch.tensor(
            librosa.filters.mel(
                sample_rate, self.n_fft, n_mels=n_filt,
                fmin=f_min, fmax=f_max),
            dtype=torch.float).unsqueeze(0)
        # self.fb = filterbanks
        # self.window = window_tensor
        self.register_buffer("fb", filterbanks)
        self.register_buffer("window", window_tensor)
        # Calculate maximum sequence length (# frames)
        max_length = 1 + math.ceil(
            (max_duration * sample_rate - self.win_length) / self.hop_length
        )
        max_pad = 16 - (max_length % 16)
        self.max_length = max_length + max_pad

    def get_seq_len(self, seq_len):
        return torch.ceil(seq_len.float() / self.hop_length).int()

    # do stft
    # TORCHSCRIPT: center removed due to bug
    def stft(self, x):
        return torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                          win_length=self.win_length,
                          window=self.window.to(dtype=torch.float))

    def forward(self, x):
        # dtype = x.dtype
        seq_len = self.get_seq_len(torch.tensor([x.shape[1]]))

        # dither
        if self.dither > 0:
            x += self.dither * torch.randn_like(x)

        # do preemphasis
        if self.preemph is not None:
            x = torch.cat(
                [x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]],
                dim=1)

        x = self.stft(x)

        # get power spectrum
        x = x.pow(2).sum(-1)

        # dot with filterbank energies
        x = torch.matmul(self.fb.to(x.dtype), x)

        # log features if required
        if self.log:
            x = torch.log(x + 1e-20)

        # normalize if required
        x = normalize_batch(x, seq_len, normalize_type=self.normalize)

        # mask to zero any values beyond seq_len in batch, pad to multiple of
        # `pad_to` (for efficiency)
        max_len = x.size(-1)
        mask = torch.arange(max_len, dtype=seq_len.dtype).to(x.device).expand(
            x.size(0), max_len) >= seq_len.unsqueeze(1)

        x = x.masked_fill(mask.unsqueeze(1), 0)
        # TORCHSCRIPT: Is this del important? It breaks scripting
        # del mask
        # TORCHSCRIPT: Cant have mixed types. Using pad_to < 0 for "max"
        if self.pad_to < 0:
            x = nn.functional.pad(x, (0, self.max_length - x.size(-1)))
        elif self.pad_to > 0:
            pad_amt = x.size(-1) % self.pad_to
            #            if pad_amt != 0:
            x = nn.functional.pad(x, (0, self.pad_to - pad_amt))

        return x
