from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tokenizer import CharTokenizer, zero_pad_concat, end_pad_concat
import os
import torchaudio
from tqdm import tqdm
import numpy as np
import torch


class MergedDataset(Dataset):
    def __init__(self, datasets):
        total_length = 0
        self.offsets = [  ]
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
    def __init__(self, path, labels='english_meta.csv', audio_max_length=15,sampling_rate=16000, transforms=None, tokenizer=CharTokenizer()):
        self.data = []
        processed_labels = 'preprocessed_'+ labels
        wav_path = labels.split('_')[0]
        self.wav_path = wav_path

        if os.path.exists(os.path.join(path, processed_labels)):
            self.data = list(pd.read_csv(os.path.join(path, processed_labels)).T.to_dict().values())
        else:
            df = pd.read_csv(os.path.join(path, labels)).T.to_dict().values()
            for voice in tqdm(df, dynamic_ncols=True):
                filename = voice['ID']
                file_path = os.path.join(path, wav_path,filename)
                if os.path.exists(file_path):
                    data, sr = torchaudio.load(file_path)
                    if sr == sampling_rate:
                        audio_length = len(data[0])//sr
                        if audio_length < audio_max_length:
                            voice['path'] = filename
                            self.data.append(voice)
            pd.DataFrame(self.data).to_csv(os.path.join(path, processed_labels))
            print('size {}'.format(len(self.data)))

        self.tokenizer = tokenizer
        self.transforms = transforms
        self.path = path
        self.vocab_size = self.tokenizer.vocab_size    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        voice = self.data[idx]
        filename = voice['ID']
        file_path = os.path.join(self.path, self.wav_path ,filename)

        data, sr = torchaudio.load(file_path, normalization=True)

        if len(data.shape) > 0:
            data = data[0] # take left channel?

        if isinstance(self.transforms, list):
            for trans in self.transforms:
                data = trans(data)

        texts = str(voice['Normalized Transcription'])
        tokens = torch.from_numpy(np.array(self.tokenizer.encode(texts))).long()

        return data.T, tokens

class CommonVoice(Dataset):

    def __init__(self, path, labels='train.tsv', audio_max_length=15,sampling_rate=16000, transforms=None, tokenizer=CharTokenizer()):
        self.data = []
        processed_labels = 'preprocessed_'+ labels.replace('.tsv', '.csv')

        # validate audio quality and sample rate
        if os.path.exists(os.path.join(path, processed_labels)):
            self.data = list(pd.read_csv(os.path.join(path, processed_labels)).T.to_dict().values())
        else:
            df = pd.read_csv(os.path.join(path, labels), sep='\t').T.to_dict().values()

            for voice in tqdm(df, dynamic_ncols=True):
                filename = voice['path'].replace('.mp3', '.wav')
                file_path = os.path.join(path, 'clips',filename)

                if os.path.exists(file_path):
                    data, sr = torchaudio.load(file_path)
                    if sr == sampling_rate:
                        audio_length = len(data[0])//sr
                        if audio_length < audio_max_length:
                            voice['path'] = filename
                            self.data.append(voice)
            pd.DataFrame(self.data).to_csv(os.path.join(path, processed_labels))
            print('size {}'.format(len(self.data)))

        self.tokenizer = tokenizer
        self.transforms = transforms
        self.path = path
        self.vocab_size = self.tokenizer.vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        voice = self.data[idx]
        filename = voice['path']
        file_path = os.path.join(self.path,'clips' ,filename)

        data, sr = torchaudio.load(file_path, normalization=True)

        if len(data.shape) > 0:
            data = data[0] # take left channel?

        if isinstance(self.transforms, list):
            for trans in self.transforms:
                data = trans(data)

        texts = voice['sentence']
        tokens = torch.from_numpy(np.array(self.tokenizer.encode(texts))).long()

        return data.T, tokens

class VoxCeleb(Dataset):
    def __init__(self, path, labels='train.tsv', audio_max_length=5,sampling_rate=16000, transforms=None, tokenizer=CharTokenizer()):
        self.path = path






def seq_collate(results):
    xs = []
    ys = []
    xlen = []
    ylen = []
    for (audio_feat, tokens) in results:
        xs.append( audio_feat )
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
    dataset = CommonVoice('../common_voice', transforms=[transforms])
    yt_dataset = YoutubeCaption('../youtube-speech-text/', transforms=[transforms])
    dataset = MergedDataset([dataset, yt_dataset])
    dataloader = DataLoader(dataset, collate_fn=seq_collate, batch_size=10, num_workers=4)
    for batch in dataloader:
        xs, ys, xlen, ylen = batch
        # print(xs.shape)
        # break

    data, tokens = dataset[0]
    print(data.shape)
    print(tokens.shape)

