import torch
import torchaudio


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

    def __init__(self, n_frame):
        super().__init__()
        self.n_frame = n_frame

    @torch.no_grad()
    def forward(self, feat):
        batch_size, feat_size, feat_length = feat.shape
        pad = (self.n_frame - feat_length % self.n_frame) % self.n_frame
        pad_shape = [0, pad, 0, 0, 0, 0]
        feat = torch.nn.functional.pad(feat, pad_shape)
        feat = feat.reshape(batch_size, feat_size * self.n_frame, -1)

        return feat
