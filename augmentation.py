import string
import torch
from torch import Tensor
from sparse_img_wrap import sparse_image_warp
import random

class ConcatFeature(torch.nn.Module):

    def __init__(self,merge_size=3):
        super(ConcatFeature, self).__init__()
        self.merge_size = merge_size
        

    def forward(self, waveform:Tensor) -> Tensor:
        feat, waveform_len = waveform.shape
        waveform = waveform.T
        if waveform_len % self.merge_size != 0:
            pad_wave = torch.zeros((self.merge_size - (waveform_len % self.merge_size), feat))
            waveform = torch.cat([waveform, pad_wave], dim=0)

        return waveform.reshape(-1, feat*self.merge_size).T


class TimeWrap(torch.nn.Module):


    def __init__(self, W=5):
        super(TimeWrap, self).__init__()
        self.W = W
    
    def forward(self, waveform:Tensor) -> Tensor:
        waveform = waveform.T
        feat, waveform_len = waveform.shape
        device = waveform.device

        waveform = waveform.unsqueeze(0)
    
        y = feat//2
        horizontal_line_at_ctr = waveform[0][y]
        assert len(horizontal_line_at_ctr) == waveform_len

        point_to_warp = horizontal_line_at_ctr[random.randrange(self.W, waveform_len - self.W)]
        assert isinstance(point_to_warp, torch.Tensor)

        # Uniform distribution from (0,W) with chance to be up to W negative
        dist_to_warp = random.randrange(-self.W, self.W)
        src_pts, dest_pts = (torch.tensor([[[y, point_to_warp]]], device=device), 
                                torch.tensor([[[y, point_to_warp + dist_to_warp]]], device=device))
        warped_waveform, dense_flows = sparse_image_warp(waveform, src_pts, dest_pts)
        warped_waveform = warped_waveform.squeeze(3).T.squeeze(-1)
        return warped_waveform


class TimeMask(torch.nn.Module):
    def __init__(self, T=40, num_masks=1, replace_with_zero=False):
        super(TimeMask, self).__init__()
        '''
            uniform distribution from 0 to the time mask parameter T
        '''
        self.T = T
        self.num_masks = num_masks
        self.replace_with_zero = replace_with_zero
    
    def forward(self, waveform):
        cloned = waveform.clone()
        len_spectro = cloned.shape[1]
        
        for i in range(0, self.num_masks):
            t = random.randrange(0, self.T)
            t_zero = random.randrange(0, len_spectro - t)

            # avoids randrange error if values are equal and range is empty
            if (t_zero == t_zero + t): return cloned

            mask_end = random.randrange(t_zero, t_zero + t)
            if (self.replace_with_zero): cloned[:,t_zero:mask_end] = 0
            else: cloned[:,t_zero:mask_end] = cloned.mean()
        return cloned

class FreqMask(torch.nn.Module):
    def __init__(self, F=40, num_masks=1, replace_with_zero=False):
        super(FreqMask, self).__init__()
        '''
            F : frequency mask parameter F,
        '''
        self.F = F
        self.num_masks = num_masks
        self.replace_with_zero = replace_with_zero
    
    def forward(self, waveform):
        cloned = waveform.clone()
        num_mel_channels = cloned.shape[0]
        
        for i in range(0, self.num_masks):
            f = random.randrange(0, self.F)
            f_zero = random.randrange(0, num_mel_channels - f)

            # avoids randrange error if values are equal and range is empty
            if (f_zero == f_zero + f): return cloned

            mask_end = random.randrange(f_zero, f_zero + f)
            if (self.replace_with_zero): cloned[f_zero:mask_end, :] = 0
            else: cloned[f_zero:mask_end, :] = cloned.mean()
        return cloned

if __name__ == "__main__":
    import torchaudio
    transforms_piplines = [
            torchaudio.transforms.MelSpectrogram(
                # n_mfcc=args.audio_feat, 
                n_fft=512, n_mels=40,
                # melkwargs={'n_fft':1024, 'win_length': 1024}
            ),
            TimeWrap(W=5),
            TimeMask(T=100),
            FreqMask(F=40),
            ConcatFeature()
        ]
    transforms = torch.nn.Sequential(*transforms_piplines)
    data, sr = torchaudio.load_wav('../common_voice/clips/common_voice_en_19664034.wav')
    print(data[0].shape)
    data = transforms(data[0])
    print(data.shape)