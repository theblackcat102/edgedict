import glob
import os
import pickle
import string
import re

import numpy as np
import pandas as pd
import torchaudio
import torch
from unidecode import unidecode
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm

from tokenizer import PAD
from parts.text.numbers import normalize_numbers


class TextCleaner:
    def __init__(self):
        self.standard_chars = [
            " ", "'", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
            "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v",
            "w", "x", "y", "z"
        ]
        self.standard_chars_set = set(self.standard_chars)

        punctuation = string.punctuation
        punctuation = punctuation.replace("+", "")
        punctuation = punctuation.replace("&", "")
        punctuation = punctuation.replace("@", "")
        punctuation = punctuation.replace("%", "")
        for char in self.standard_chars:
            punctuation = punctuation.replace(char, "")
        self.punctuation = str.maketrans(punctuation, " " * len(punctuation))

        self.abbreviations = [
            (re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
                ('mrs', 'misess'),
                ('mr', 'mister'),
                ('dr', 'doctor'),
                ('st', 'saint'),
                ('co', 'company'),
                ('jr', 'junior'),
                ('maj', 'major'),
                ('gen', 'general'),
                ('drs', 'doctors'),
                ('rev', 'reverend'),
                ('lt', 'lieutenant'),
                ('hon', 'honorable'),
                ('sgt', 'sergeant'),
                ('capt', 'captain'),
                ('esq', 'esquire'),
                ('ltd', 'limited'),
                ('col', 'colonel'),
                ('ft', 'fort'),
            ]]
        self.whitespace_re = re.compile(r'\s+')

    def expand_abbreviations(self, text):
        for regex, replacement in self._abbreviations:
            text = re.sub(regex, replacement, text)
        return text

    def expand_numbers(self, text):
        return normalize_numbers(text)

    def lowercase(text):
        return text.lower()

    def collapse_whitespace(self, text):
        return re.sub(self.whitespace_re, ' ', text)

    def convert_to_ascii(self, text):
        return unidecode(text)

    def remove_punctuation(self, text):
        text = text.translate(self.punctuation)
        text = re.sub(r'&', " and ", text)
        text = re.sub(r'\+', " plus ", text)
        text = re.sub(r'@', " at ", text)
        text = re.sub(r'%', " percent ", text)
        return text

    def english_cleaners(self, text):
        text = self.convert_to_ascii(text)
        text = self.lowercase(text)
        text = self.expand_numbers(text)
        text = self.expand_abbreviations(text)
        text = self.remove_punctuation(text)
        text = self.collapse_whitespace(text)
        return text

    def normalize(self, text):
        def good_token(token):
            for t in token:
                if t not in self.standard_chars_set:
                    return False
            return True

        text = self.english_cleaners(text).strip()
        text = ''.join([token for token in text if good_token(token)])
        return text


class MergedDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        self.datasets = datasets
        self.total_length = 0
        for dataset in datasets:
            self.total_length += len(dataset)
        self.tokenizer = datasets[0].tokenizer

    def texts(self):
        texts = []
        for dataset in self.datasets:
            texts.extend(dataset.texts())
        return texts


class AudioDataset(Dataset):
    def __init__(self, root, tokenizer, session='', desc='AudioDataset',
                 transform=None, audio_min_length=0, audio_max_length=999,
                 sampling_rate=16000, reverse_sorted_by_length=False):
        self.root = root
        processed_labels = os.path.join(
            root, 'preprocessed_v3_%s.pkl' % session)

        if os.path.exists(processed_labels):
            data = pickle.load(open(processed_labels, 'rb'))
        else:
            data = []
            paths, texts = self.build()
            pairs = list(zip(paths, texts))
            with tqdm(pairs, dynamic_ncols=True, desc=desc) as pbar:
                for path, text in pbar:
                    full_path = os.path.join(root, path)
                    try:
                        if os.path.exists(full_path):
                            wave, sr = torchaudio.load(
                                full_path, normalization=False)
                            if sr == sampling_rate:
                                audio_length = len(wave[0]) // sr
                                data.append({
                                    'path': path,
                                    'text': text,
                                    'audio_length': audio_length})
                    except RuntimeError:
                        pbar.write('Fail to load %s' % full_path)
            pickle.dump(data, open(processed_labels, 'wb'))

        # length limits and text cleaning
        # text_cleaner = TextCleaner()
        total_secs = 0
        filtered_secs = 0
        self.data = []
        for x in data:
            if audio_min_length <= x['audio_length'] <= audio_max_length:
                # if normalize_text:
                #     x['text'] = text_cleaner.normalize(x['text'])
                self.data.append(x)
                total_secs += x['audio_length']
            else:
                filtered_secs += x['audio_length']
        print('Dataset : %s' % desc)
        print('size    : %d' % len(self.data))
        print('Time    : %.2f hours' % (total_secs / 3600))
        print('Filtered: %.2f hours' % (filtered_secs / 3600))
        print('=' * 40)

        if reverse_sorted_by_length:
            self.data = sorted(
                self.data, key=lambda x: x['audio_length'], reverse=True)
        # print(root, data[0]['path'])
        self.transform = transform
        self.tokenizer = tokenizer

    def texts(self):
        return [x['text'] for x in self.data]

    def build(self):
        # return paths, texts, all path in paths is relative to self.root
        raise NotImplementedError()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.data[idx]['path'])
        try:
            data, sr = torchaudio.load(path, normalization=True)
        except Exception:
            print("Failt to load %s, closed" % path)
            exit(0)
        data = self.transform(data[:1])

        texts = self.data[idx]['text']
        tokens = torch.from_numpy(np.array(self.tokenizer.encode(texts)))

        return data[0].T, tokens


class YoutubeCaption(AudioDataset):
    def __init__(self, root, labels, tokenizer, *args, **kwargs):
        self.labels = labels
        session = labels.replace('.csv', '')
        desc = "YoutubeCaption"
        super(YoutubeCaption, self).__init__(
            root, tokenizer, session, desc, *args, **kwargs)

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
    def __init__(self, root, labels, tokenizer, *args, **kwargs):
        self.labels = labels
        session = labels.replace('.tsv', '')
        desc = "CommonVoice"
        super(CommonVoice, self).__init__(
            root, tokenizer, session, desc, *args, **kwargs)

    def build(self):
        paths = []
        texts = []
        df = pd.read_csv(os.path.join(self.root, self.labels), sep='\t')
        df = df.T.to_dict().values()
        for voice in df:
            filename = voice['path'].replace('.mp3', '.wav')
            path = os.path.join('clips', filename)
            paths.append(path)
            texts.append(voice['sentence'])
        return paths, texts


class Librispeech(AudioDataset):
    def __init__(self, root, tokenizer, *args, **kwargs):
        session = 'label'
        desc = "Librispeech"
        super(Librispeech, self).__init__(
            root, tokenizer, session, desc, *args, **kwargs)

    def build(self):
        paths = []
        texts = []
        trans_files = list(glob.glob(os.path.join(self.root, '*/*/*.txt')))
        for trans_file in trans_files:
            dir2 = os.path.dirname(trans_file)
            dir1 = os.path.dirname(dir2)
            dir_path = os.path.join(
                os.path.basename(dir1), os.path.basename(dir2))
            with open(trans_file, 'r') as f:
                for line in f.readlines():
                    filename, text = line.split(maxsplit=1)
                    path = os.path.join(dir_path, filename + '.flac')
                    paths.append(path)
                    texts.append(text)
        return paths, texts


class TEDLIUM(AudioDataset):
    def __init__(self, root, tokenizer, *args, **kwargs):
        session = 'label'
        desc = "TEDLIUM"

        super(TEDLIUM, self).__init__(
            root, tokenizer, session, desc, *args, **kwargs)

    def build(self):
        paths = []
        texts = []
        with open(os.path.join(self.root, 'wav', 'labels.txt')) as f:
            for line in f:
                filename, text = line.split(maxsplit=1)
                path = os.path.join('wav', filename)
                paths.append(path)
                texts.append(text)
        return paths, texts


def zero_pad_concat(feats):
    # Pad audio feature sets
    max_t = max(len(feat) for feat in feats)
    shape = (len(feats), max_t) + feats[0].shape[1:]

    input_mat = torch.zeros(shape)
    for e, feat in enumerate(feats):
        input_mat[e, :len(feat)] = feat

    return input_mat


def end_pad_concat(texts):
    # Pad text token sets
    max_t = max(len(text) for text in texts)
    shape = (len(texts), max_t)

    labels = torch.full(shape, fill_value=PAD, dtype=torch.long)
    for e, l in enumerate(texts):
        labels[e, :len(l)] = l
    return labels


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
    ys = end_pad_concat(ys).int()
    xlen = torch.from_numpy(np.array(xlen)).int()
    ylen = torch.from_numpy(np.array(ylen)).int()
    return xs, ys, xlen, ylen


if __name__ == "__main__":
    from tokenizer import CharTokenizer
    from torchaudio.transforms import MFCC

    from transforms import CatDeltas, CMVN, Downsample

    transform = torch.nn.Sequential(
        MFCC(
            n_mfcc=80,
            melkwargs={
                'n_fft': 400,
                'win_length': 400,
                'hop_length': 200,
                'f_min': 20,
                'f_max': 5800
            }),
        CatDeltas(),
        CMVN(),
        Downsample(3)
    )
    tokenizer = CharTokenizer(cache_dir='/tmp')
    train_dataloader = DataLoader(
        dataset=MergedDataset([
            Librispeech(
                root='./data/LibriSpeech/train-other-500',
                tokenizer=tokenizer,
                transforms=transform,
                audio_max_length=14),
            Librispeech(
                root='./data/LibriSpeech/train-clean-360',
                tokenizer=tokenizer,
                transforms=transform,
                audio_max_length=14),
            Librispeech(
                root='./data/LibriSpeech/train-clean-100',
                tokenizer=tokenizer,
                transforms=transform,
                audio_max_length=14),
            # TEDLIUM(
            #     root="./data/TEDLIUM_release-3/data",
            #     tokenizer=tokenizer,
            #     transforms=transform,
            #     audio_max_length=14),
            # TEDLIUM(
            #     root="./data/TEDLIUM_release1/train",
            #     tokenizer=tokenizer,
            #     transforms=transform,
            #     audio_max_length=14),
            # CommonVoice(
            #     root='./data/common_voice', labels='train.tsv',
            #     tokenizer=tokenizer,
            #     transforms=transform,
            #     audio_max_length=14)
        ]),
        batch_size=4, shuffle=True, num_workers=4,
        collate_fn=seq_collate, drop_last=True)

    val_dataloader = DataLoader(
        dataset=MergedDataset([
            Librispeech(
                './data/LibriSpeech/test-clean',
                tokenizer=tokenizer,
                transforms=transform),
            # Librispeech(
            #     './data/LibriSpeech/dev-clean',
            #     tokenizer=tokenizer,
            #     transforms=transform),
            # Librispeech(
            #     './data/LibriSpeech/test-other',
            #     tokenizer=tokenizer,
            #     transforms=transform),
            # Librispeech(
            #     './data/LibriSpeech/dev-other',
            #     tokenizer=tokenizer,
            #     transforms=transform),
            # TEDLIUM(
            #     root='./data/TEDLIUM_release1/test',
            #     tokenizer=tokenizer,
            #     transforms=transform),
            # CommonVoice(
            #     root='./data/common_voice', labels='test.tsv',
            #     tokenizer=tokenizer,
            #     transforms=transform)
        ]),
        batch_size=4, shuffle=False, num_workers=4,
        collate_fn=seq_collate)

    tokenizer.build(train_dataloader.dataset.texts())
    print("==========================")
    for xs, ys, xlen, ylen in tqdm(train_dataloader):
        pass
        print(xs.shape, ys.shape, xlen.shape, ylen.shape)

    for xs, ys, xlen, ylen in tqdm(val_dataloader):
        pass
        # print(xs.shape, ys.shape, xlen.shape, ylen.shape)
