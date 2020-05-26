import glob
import numpy as np
import os
import pandas as pd
import torchaudio
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tokenizer import CharTokenizer, zero_pad_concat, end_pad_concat


class MergedDataset(Dataset):
    def __init__(self, datasets):
        total_length = 0
        self.offsets = []
        for d in datasets:
            total_length += len(d)
            self.offsets.append(total_length)

        self.total_length = total_length
        self.datasets = datasets
        self.vocab_size = self.datasets[0].vocab_size

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        prev = 0
        for d_idx, offset in enumerate(self.offsets):
            if idx < offset:
                assert (idx-prev) >= 0
                return self.datasets[d_idx][idx - prev]
            prev = offset


class YoutubeCaption(Dataset):
    def __init__(self, path, labels='english_meta.csv',
                 audio_max_length=18, audio_min_length=1, sampling_rate=16000, transforms=None, tokenizer=CharTokenizer()):
        self.data = []
        processed_labels = 'preprocessed_' + labels
        wav_path = labels.split('_')[0]
        self.wav_path = wav_path

        if os.path.exists(os.path.join(path, processed_labels)):
            self.data = list(pd.read_csv(os.path.join(path, processed_labels)).T.to_dict().values())
        else:
            total_secs = 0
            df = pd.read_csv(os.path.join(path, labels)).T.to_dict().values()
            for voice in tqdm(df, dynamic_ncols=True):
                filename = voice['ID']
                file_path = os.path.join(path, wav_path, filename)
                try:
                    if os.path.exists(file_path):
                        data, sr = torchaudio.load(file_path)
                        if sr == sampling_rate:
                            audio_length = len(data[0])//sr
                            voice['Transcription'] = str(voice['Transcription'])
                            if audio_length < audio_max_length and audio_length > audio_min_length and ' ' in voice['Transcription']:
                                total_secs += audio_length
                                self.data.append(voice)
                except RuntimeError:
                    continue
            pd.DataFrame(self.data).to_csv(os.path.join(path, processed_labels))
            print('size {}'.format(len(self.data)))
            print('hrs {}'.format(total_secs/3600))

        self.tokenizer = tokenizer
        self.transforms = transforms
        self.path = path
        self.vocab_size = self.tokenizer.vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        voice = self.data[idx]
        filename = voice['ID']
        file_path = os.path.join(self.path, self.wav_path, filename)

        data, sr = torchaudio.load(file_path, normalization=True)

        if len(data.shape) > 0:
            data = data[0]  # take left channel?

        if isinstance(self.transforms, list):
            for trans in self.transforms:
                data = trans(data)

        texts = str(voice['Normalized Transcription'])
        tokens = torch.from_numpy(np.array(self.tokenizer.encode(texts))).long()

        return data.T, tokens


class CommonVoice(Dataset):
    def __init__(self, path, labels='train.tsv', audio_max_length=18, sampling_rate=16000, transforms=None, tokenizer=CharTokenizer()):
        self.data = []
        processed_labels = 'preprocessed_' + labels.replace('.tsv', '.csv')

        # validate audio quality and sample rate
        if os.path.exists(os.path.join(path, processed_labels)):
            self.data = list(pd.read_csv(os.path.join(path, processed_labels)).T.to_dict().values())
        else:
            df = pd.read_csv(os.path.join(path, labels), sep='\t').T.to_dict().values()
            total_secs = 0
            for voice in tqdm(df, dynamic_ncols=True):
                filename = voice['path'].replace('.mp3', '.wav')
                file_path = os.path.join(path, 'clips', filename)

                if os.path.exists(file_path):
                    data, sr = torchaudio.load(file_path)
                    if sr == sampling_rate:
                        audio_length = len(data[0]) // sr
                        if audio_length < audio_max_length:
                            voice['path'] = filename
                            total_secs += audio_length
                            self.data.append(voice)
            pd.DataFrame(self.data).to_csv(os.path.join(path, processed_labels))
            print('size {}'.format(len(self.data)))
            print('hrs {}'.format(total_secs / 3600))

        self.tokenizer = tokenizer
        self.transforms = transforms
        self.path = path
        self.vocab_size = self.tokenizer.vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        voice = self.data[idx]
        filename = voice['path']
        file_path = os.path.join(self.path, 'clips', filename)

        data, sr = torchaudio.load(file_path, normalization=True)

        if len(data.shape) > 0:
            data = data[0]  # take left channel?

        if isinstance(self.transforms, list):
            for trans in self.transforms:
                data = trans(data)

        texts = voice['sentence']
        tokens = torch.from_numpy(np.array(self.tokenizer.encode(texts))).long()

        return data.T, tokens


class VoxCeleb(Dataset):
    def __init__(self, path, labels='train.tsv', audio_max_length=5, sampling_rate=16000, transforms=None, tokenizer=CharTokenizer()):
        self.path = path


class LibreSpeech(Dataset):
    def __init__(self, path, audio_max_length=18, sampling_rate=16000, transforms=None, tokenizer=CharTokenizer()):
        self.sampling_rate = sampling_rate
        self.transforms = transforms
        self.tokenizer = tokenizer
        all_voice = glob.glob(os.path.join(path, '*/*/*.wav'))
        all_trans = glob.glob(os.path.join(path, '*/*/*.trans.txt'))

        self.labels = {}
        for trans_file in all_trans:
            with open(trans_file) as f:
                for line in f:
                    name, text = line.split(maxsplit=1)
                    self.labels[name] = text

        self.voice = []
        for voice_file in all_voice:
            if os.path.splitext(os.path.basename(voice_file))[0] in self.labels:
                self.voice.append(voice_file)
        print(len(list(self.voice)))

        self.vocab_size = tokenizer.vocab_size

    def __len__(self):
        return len(self.voice)

    def __getitem__(self, idx):
        data, sr = torchaudio.load(self.voice[idx], normalization=True)
        assert(sr == self.sampling_rate)

        if len(data.shape) > 0:
            data = data[0]  # take left channel?

        if isinstance(self.transforms, list):
            for trans in self.transforms:
                data = trans(data)

        filename = os.path.splitext(os.path.basename(self.voice[idx]))[0]
        texts = self.labels[filename]
        tokens = torch.from_numpy(np.array(self.tokenizer.encode(texts))).long()

        return data.T, tokens


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


if __name__ == "__main__":
    transforms = torchaudio.transforms.MFCC(n_mfcc=40)
    # cv_dataset = CommonVoice('../common_voice', labels='test.tsv', transforms=[transforms])
    # print(cv_dataset.vocab_size)

    # cv_dataset = CommonVoice('../common_voice', transforms=[transforms])
    # print(cv_dataset.vocab_size)

    # yt_dataset = YoutubeCaption('../youtube-speech-text/', transforms=[transforms])

    ls_dataset = LibreSpeech('/nfs/lab2/dataset/libri_speech/LibriSpeech/train-clean-100', transforms=[transforms])
    data, tokens = ls_dataset[0]
    print(data.shape)
    print(tokens.shape)

    ls_dataset = LibreSpeech('/nfs/lab2/dataset/libri_speech/LibriSpeech/test-clean', transforms=[transforms])
    data, tokens = ls_dataset[0]
    print(data.shape)
    print(tokens.shape)

    # dataset = MergedDataset([cv_dataset, yt_dataset, ls_dataset])
    dataset = MergedDataset([ls_dataset])
    data, tokens = dataset[0]
    print(data.shape)
    print(tokens.shape)

    dataloader = DataLoader(dataset, collate_fn=seq_collate, batch_size=10, num_workers=4)
    xs, ys, xlen, ylen = next(iter(dataloader))
    print(xs.shape)
