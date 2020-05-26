import glob
import os
import re

import numpy as np
import pandas as pd
import torchaudio
import torch
from sphfile import SPHFile
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


class YoutubeCaption(Dataset):
    def __init__(self, path, labels='english_meta.csv',
                 audio_max_length=18, audio_min_length=1, sampling_rate=16000,
                 transforms=None, tokenizer=CharTokenizer()):
        self.data = []
        processed_labels = 'preprocessed_' + labels
        wav_path = labels.split('_')[0]
        self.wav_path = wav_path

        if os.path.exists(os.path.join(path, processed_labels)):
            self.data = list(pd.read_csv(
                os.path.join(path, processed_labels)).T.to_dict().values())
        else:
            total_secs = 0
            df = pd.read_csv(os.path.join(path, labels)).T.to_dict().values()
            for voice in tqdm(df, dynamic_ncols=True, desc="YoutubeCaption"):
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
            print('hrs  {}'.format(total_secs / 3600))

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
        tokens = torch.from_numpy(np.array(self.tokenizer.encode(texts)))

        return data.T, tokens


class CommonVoice(Dataset):
    def __init__(self, path, labels='train.tsv',
                 audio_max_length=18, sampling_rate=16000,
                 transforms=None, tokenizer=CharTokenizer()):
        self.data = []
        processed_labels = 'preprocessed_' + labels.replace('.tsv', '.csv')

        # validate audio quality and sample rate
        if os.path.exists(os.path.join(path, processed_labels)):
            self.data = list(pd.read_csv(
                os.path.join(path, processed_labels)).T.to_dict().values())
        else:
            df = pd.read_csv(os.path.join(path, labels), sep='\t').T.to_dict().values()
            total_secs = 0
            for voice in tqdm(df, dynamic_ncols=True, desc="CommonVoice"):
                filename = voice['path'].replace('.mp3', '.wav')
                file_path = os.path.join(path, 'clips', filename)
                try:
                    if os.path.exists(file_path):
                        data, sr = torchaudio.load(file_path)
                        if sr == sampling_rate:
                            audio_length = len(data[0])//sr
                            if audio_length < audio_max_length:
                                voice['path'] = filename
                                total_secs += audio_length
                                self.data.append(voice)
                except RuntimeError:
                    continue

            pd.DataFrame(self.data).to_csv(
                os.path.join(path, processed_labels))
            print('size {}'.format(len(self.data)))
            print('hrs  {}'.format(total_secs / 3600))

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
        tokens = torch.from_numpy(np.array(self.tokenizer.encode(texts)))

        return data.T, tokens


class Librispeech(Dataset):
    def __init__(self, path='../LibriSpeech/train-clean-360/',
                 audio_max_length=18, transforms=None,
                 sampling_rate=16000, tokenizer=CharTokenizer()):
        self.data = []
        processed_labels = 'preprocessed_label.csv'

        # validate audio quality and sample rate
        if os.path.exists(os.path.join(path, processed_labels)):
            self.data = list(pd.read_csv(
                os.path.join(path, processed_labels)).T.to_dict().values())
        else:
            self.data = []
            total_secs = 0
            paths = list(glob.glob(os.path.join(path, '*/*/*.txt')))
            for texts in tqdm(paths, dynamic_ncols=True, desc="Librispeech"):
                with open(texts, 'r') as f:
                    dir_path = os.path.dirname(os.path.realpath(texts))
                    for line in f.readlines():
                        filename = line.split(' ')[0]
                        wav_path = os.path.join(dir_path, filename + '.flac')
                        data, sr = torchaudio.load(wav_path)
                        if sr == sampling_rate:
                            text = line.replace(filename, '').strip().lower()
                            audio_length = len(data[0])//sr
                            if audio_length < audio_max_length:
                                total_secs += audio_length
                                self.data.append({
                                    'path': wav_path,
                                    'text': text
                                })
                        else:
                            print('Warning')
            pd.DataFrame(self.data).to_csv(
                os.path.join(path, processed_labels))
            print('size {}'.format(len(self.data)))
            print('hrs  {}'.format(total_secs / 3600))

        self.tokenizer = tokenizer
        self.transforms = transforms
        self.path = path
        self.vocab_size = self.tokenizer.vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        voice = self.data[idx]
        file_path = voice['path']

        data, sr = torchaudio.load(file_path, normalization=True)

        if len(data.shape) > 0:
            data = data[0]  # take left channel?

        if isinstance(self.transforms, list):
            for trans in self.transforms:
                data = trans(data)

        texts = voice['text']
        tokens = torch.from_numpy(np.array(self.tokenizer.encode(texts)))

        return data.T, tokens


class TEDLIUM(Dataset):
    PAUSE_MATCH = re.compile(r'\([0-9]\)')
    NOTATION = re.compile(r'\{[A-Z]*\}')

    def __init__(self, path='../TEDLIUM/TEDLIUM_release1/train/',
                 audio_max_length=18, transforms=None,
                 sampling_rate=16000, tokenizer=CharTokenizer()):
        self.data = []
        processed_labels = 'preprocessed_label.csv'
        wav_dir = os.path.join(path, 'wav')

        # validate audio quality and sample rate
        if os.path.exists(os.path.join(path, processed_labels)):
            self.data = list(pd.read_csv(
                os.path.join(path, processed_labels)).T.to_dict().values())
        else:
            self.data = []
            total_secs = 0
            os.makedirs(wav_dir, exist_ok=True)

            for sph_audio in tqdm(glob.glob(os.path.join(path, 'sph/*.sph'))):
                sph = SPHFile(sph_audio)
                stm_file = sph_audio.replace('sph', 'stm')
                with open(stm_file, 'r') as f:
                    for idx, line in enumerate(f.readlines()):
                        tokens = line.split(' ')
                        start, end = float(tokens[3]), float(tokens[4])
                        if (end-start) <= audio_max_length:
                            name = tokens[0]
                            total_secs += (end-start)
                            text = line.split('male> ')[-1].split('('+name)[0]
                            text = text.replace('<sil>', '')

                            text = TEDLIUM.PAUSE_MATCH.sub('', text)
                            text = TEDLIUM.NOTATION.sub('', text)

                            wav_filename = name+'_'+str(idx)+'.wav'
                            sph.write_wav(os.path.join(wav_dir, wav_filename),
                                          start, end)
                            self.data.append({
                                'path': wav_filename,
                                'text': text.strip()
                            })

            pd.DataFrame(self.data).to_csv(
                os.path.join(path, processed_labels))
            print('size {}'.format(len(self.data)))
            print('hrs  {}'.format(total_secs / 3600))
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.path = wav_dir
        self.vocab_size = self.tokenizer.vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        voice = self.data[idx]
        file_path = voice['path']

        data, sr = torchaudio.load(file_path, normalization=True)

        if len(data.shape) > 0:
            data = data[0]  # take left channel?

        if isinstance(self.transforms, list):
            for trans in self.transforms:
                data = trans(data)

        texts = voice['text']
        tokens = torch.from_numpy(np.array(self.tokenizer.encode(texts)))

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
    # Test
    librispeech = Librispeech(
        '../LibriSpeech/test-clean/', transforms=[transforms])
    commonvoice = CommonVoice(
        '../common_voice', labels='test.tsv', transforms=[transforms])
    dataset = MergedDataset([librispeech, commonvoice])
    dataloader = DataLoader(
        dataset, collate_fn=seq_collate, batch_size=10, num_workers=4)
    for i, (xs, ys, xlen, ylen) in enumerate(dataloader):
        print(xs.shape)
        if i == 3:
            break

    # Train
    # youtubecaption = YoutubeCaption(
    #     '../youtube-speech-text/', transforms=[transforms])
    # tedlium = TEDLIUM(
    #     '../TEDLIUM/TEDLIUM_release1/train/', transforms=[transforms])
    librispeech = Librispeech(
        '../LibriSpeech/train-clean-100/', transforms=[transforms])
    commonvoice = CommonVoice(
        '../common_voice', labels='train.tsv', transforms=[transforms])

    dataset = MergedDataset([librispeech, commonvoice])
    dataloader = DataLoader(
        dataset, collate_fn=seq_collate, batch_size=10, num_workers=4)
    for i, (xs, ys, xlen, ylen) in enumerate(dataloader):
        print(xs.shape)
        if i == 3:
            break
