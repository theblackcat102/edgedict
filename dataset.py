import glob
import os

import numpy as np
import pandas as pd
import torchaudio
import torch
import torchaudio.transforms as transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm

from tokenizer import CharTokenizer, zero_pad_concat, end_pad_concat


class MergedDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        self.total_length = 0
        for d in datasets:
            self.total_length += len(d)
        self.vocab_size = self.datasets[0].vocab_size
        self.tokenizer = datasets[0].tokenizer
        self.vocab_size = self.tokenizer.vocab_size


class AudioDataset(Dataset):
    def __init__(self, root, session='', desc='AudioDataset',
                 audio_max_length=99, audio_min_length=1, sampling_rate=16000,
                 transforms=None, tokenizer=CharTokenizer()):
        self.root = root
        processed_labels = os.path.join(root, 'preprocessed_' + session)

        if os.path.exists(processed_labels):
            self.data = list(
                pd.read_csv(processed_labels).T.to_dict().values())
        else:
            self.data = []
            total_secs = 0
            paths, texts = self.build()
            pairs = list(zip(paths, texts))
            for path, text in tqdm(pairs, dynamic_ncols=True, desc=desc):
                try:
                    if os.path.exists(path):
                        wave, sr = torchaudio.load(path, normalization=False)
                        if sr == sampling_rate:
                            audio_length = len(wave[0]) // sr
                            if audio_min_length < audio_length < audio_max_length:
                                total_secs += audio_length
                                self.data.append({'path': path, 'text': text})
                except RuntimeError:
                    continue
            pd.DataFrame(self.data).to_csv(processed_labels)
            print('size {}'.format(len(self.data)))
            print('hrs  {}'.format(total_secs / 3600))

        self.transforms = transforms
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size

    def build(self):
        raise NotImplementedError()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, sr = torchaudio.load(self.data[idx]['path'], normalization=True)

        if len(data.shape) > 0:
            data = data[0]  # take left channel?

        if isinstance(self.transforms, list):
            for trans in self.transforms:
                data = trans(data)
        else:
            data = self.transforms(data)

        texts = self.data[idx]['text']
        tokens = torch.from_numpy(np.array(self.tokenizer.encode(texts)))

        return data.T, tokens


class YoutubeCaption(AudioDataset):
    def __init__(self, root, labels='english_meta.csv', *args, **kwargs):
        self.labels = labels
        session = labels
        desc = "YoutubeCaption"
        super(YoutubeCaption, self).__init__(
            root, session, desc, *args, **kwargs)

    def build(self):
        paths = []
        texts = []
        wav_path = os.path.join(self.root, self.labels.split('_')[0])
        df = pd.read_csv(os.path.join(self.root, self.labels))
        df = df.T.to_dict().values()
        for voice in df:
            filename = voice['ID']
            path = os.path.join(wav_path, filename)
            text = str(voice['Transcription'])
            if ' ' in text:
                paths.append(path)
                texts.append(text)
        return paths, texts


class CommonVoice(AudioDataset):
    def __init__(self, root, labels='train.tsv', *args, **kwargs):
        self.labels = labels
        session = labels.replace('.tsv', '.csv')
        desc = "CommonVoice"
        super(CommonVoice, self).__init__(root, session, desc, *args, **kwargs)

    def build(self):
        paths = []
        texts = []
        df = pd.read_csv(os.path.join(self.root, self.labels), sep='\t')
        df = df.T.to_dict().values()
        for voice in df:
            filename = voice['path'].replace('.mp3', '.wav')
            path = os.path.join(self.root, 'clips', filename)
            paths.append(path)
            texts.append(voice['sentence'])
        return paths, texts


class Librispeech(AudioDataset):
    def __init__(self, root, *args, **kwargs):
        session = 'label.csv'
        desc = "Librispeech"
        super(Librispeech, self).__init__(root, session, desc, *args, **kwargs)

    def build(self):
        paths = []
        texts = []
        trans_files = list(glob.glob(os.path.join(self.root, '*/*/*.txt')))
        for trans_file in tqdm(trans_files):
            with open(trans_file, 'r') as f:
                dir_path = os.path.dirname(os.path.realpath(trans_file))
                for line in f.readlines():
                    filename, text = line.split(maxsplit=1)
                    path = os.path.join(dir_path, filename + '.flac')
                    paths.append(path)
                    texts.append(text)
        return paths, texts


class TEDLIUM(AudioDataset):
    def __init__(self, root, *args, **kwargs):
        session = 'label.csv'
        desc = "TEDLIUM"

        super(TEDLIUM, self).__init__(root, session, desc, *args, **kwargs)

    def build(self):
        paths = []
        texts = []
        with open(os.path.join(self.root, 'wav', 'labels.txt')) as f:
            for line in f:
                filename, text = line.split(maxsplit=1)
                path = os.path.join(self.root, 'wav', filename)
                paths.append(path)
                texts.append(text)
        return paths, texts


def seq_collate(results):
    xs = []
    ys = []
    xlen = []
    ylen = []
    for (audio_feat, tokens) in results:
        xs.append(audio_feat)
        ys.append(tokens)
        xlen.append(len(audio_feat))
        ylen.append(len(tokens))

    xs = zero_pad_concat(xs)
    ys = end_pad_concat(ys)
    xlen = torch.from_numpy(np.array(xlen)).int()
    ylen = torch.from_numpy(np.array(ylen)).int()
    return xs, ys, xlen, ylen


class Compose:
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, x):
        for transform in self.transform_list:
            x = transform(x)
        return x


class LogMelSpectrogram(transforms.MelSpectrogram):
    """
    ref: https://github.com/noahchalifour/rnnt-speech-recognition/blob/a0d972f5e407e465ad784c682fa4e72e33d8eefe/utils/preprocessing.py#L48
    """
    def forward(self, waveform):
        mel_specs = super().forward(waveform)
        log_mel_specs = torch.log(mel_specs + 1e-6)
        log_mel_specs -= torch.mean(log_mel_specs, dim=0, keepdim=True)
        return log_mel_specs


class DownsampleSpectrogram:
    def __init__(self, n_frame):
        self.n_frame = n_frame

    def __call__(self, spec):
        feat_size, spec_length = spec.shape
        spec_length = (spec_length // self.n_frame) * self.n_frame
        spec_sampled = spec[:, :spec_length]
        spec_sampled = spec_sampled.reshape(feat_size * self.n_frame, -1)
        return spec_sampled


if __name__ == "__main__":
    sr = 16000
    transform = Compose([
        LogMelSpectrogram(
            sample_rate=sr, win_length=int(0.025 * sr),
            hop_length=int(0.01 * sr), n_fft=512, f_min=125, f_max=7600,
            n_mels=80),
        DownsampleSpectrogram(n_frame=3)
    ])
    # Test
    librispeech = Librispeech(
        '../LibriSpeech/test-clean/', transforms=[transform])
    tedlium = TEDLIUM(
        '../TEDLIUM_release1/test/', transforms=[transform])
    commonvoice = CommonVoice(
        '../common_voice', labels='test.tsv', transforms=[transform])
    dataset = MergedDataset([librispeech, tedlium, commonvoice])
    dataloader = DataLoader(
        dataset, collate_fn=seq_collate, batch_size=8, num_workers=4,
        shuffle=True)
    for i, (xs, ys, xlen, ylen) in enumerate(dataloader):
        print(xs.shape)
        if i == 3:
            break

    # Train
    librispeech = Librispeech(
        '../LibriSpeech/train-clean-100/', transforms=[transform])
    tedlium = TEDLIUM(
        '../TEDLIUM_release1/train/', transforms=[transform])
    commonvoice = CommonVoice(
        '../common_voice', labels='train.tsv', transforms=[transform])
    # youtubecaption = YoutubeCaption(
    #     '../youtube-speech-text/', transforms=[transform])
    dataset = MergedDataset([librispeech, tedlium, commonvoice])
    dataloader = DataLoader(
        dataset, collate_fn=seq_collate, batch_size=8, num_workers=4,
        shuffle=True)
    for i, (xs, ys, xlen, ylen) in enumerate(dataloader):
        print(xs.shape)
        if i == 3:
            break
