import string
import torch
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from parts.text.cleaners import english_cleaners

'''
Not to be confused with tokenizers package
'''

BOS = 1

def zero_pad_concat(inputs):
    # Pad audio feature sets
    max_t = max(inp.shape[0] for inp in inputs)
    shape = (len(inputs), max_t) + inputs[0].shape[1:]

    input_mat = torch.zeros(shape)
    for e, inp in enumerate(inputs):
        try:
            input_mat[e, :inp.shape[0]] = inp
        except:
            print(input_mat.shape, inp.shape, e)
    # return pad_sequence(inputs, batch_first=True, padding_value=0)
    return input_mat

def end_pad_concat(inputs, pad_idx=0):
    # Pad text token sets
    return pad_sequence(inputs, batch_first=True, padding_value=pad_idx).long()



class CharTokenizer():


    def __init__(self):
        valid_tokens = string.ascii_lowercase + string.punctuation + ' 0123456789ÂÂ°Â´ÂˇÂ˝Ă°Ă¸ÄĹĹÉÉÉÉĘĘĘĘťÎĎĐĐľĐˇĐ¸Đ˝ŃŃŃŘ¨Ř°Ř˛Ř´ŮŮŮŮŰŕ˛ááˇáźâââââ˘âĺ˝ĺćˇç'

        self.token2id = {
            '<blank>': 0,
            '<bos>': 1,
            '<unk>': 2,
            
        }

        self.id2token = {}
        for idx, token in enumerate(valid_tokens):
            self.token2id[token] = idx+4

        for token, idx in self.token2id.items():
            self.id2token[idx] = token

        self.vocab_size = len(self.id2token)
    def __str__(self):
        return 'CharTokenizer'

    def encode(self, text, max_length=-1):
        text = str(text).lower()
        if max_length > 1:
            text = text[:max_length]
        return [1]+[ self.token2id[char]  if char in self.token2id else 1 for char in text]

    def decode(self, tokens):
        text = ''.join([ self.id2token[t] if t in self.id2token else '' for t in tokens ])
        text = text.replace('<pad>', '').replace('<blank>', '')
        text = text.replace('<eos>', '')
        return text

    
    def decode_plus(self, token_batch):
        sentences = []
        for tokens in token_batch:
            sentences.append(self.decode(tokens))
        return sentences

from modules.tokenizers import CharBPETokenizer


class HuggingFaceTokenizer():

    def __init__(self, tokenizers=None, cleaner=english_cleaners):
        if tokenizers == None:
            tokenizers = CharBPETokenizer(
                './BPE-1024/-vocab.json',
                './BPE-1024/-merges.txt',
                lowercase=True,
            )
        punctuation = string.punctuation
        punctuation = punctuation.replace("+", "")
        punctuation = punctuation.replace("&", "")
        table = str.maketrans(punctuation, " " * len(punctuation))
        if cleaner!= None:
            print('Use cleaner !')
        self.table = table
        self.cleaner = cleaner
        self.token = tokenizers
        self.vocab_size = self.token.get_vocab_size()

    def __str__(self): # Zzzz 
        return 'HuggingFaceTokenizer-{}'.format(self.vocab_size)


    def encode(self, text, max_length=-1):
        if self.cleaner != None:
            text = self.cleaner(text, table=self.table)
        token_ids = self.token.encode(text).ids
        if max_length > 0:
            token_ids = token_ids[:max_length]

        # Add <eos> and <bos> to front and end of sentence
        return token_ids
        
    def decode(self, tokens, skip_special_tokens=False):
        text = self.token.decode(
            list(tokens),
            skip_special_tokens=skip_special_tokens,
        )
        return text.replace('<pad>', '').replace('<blank>', '').replace('<bos>', '')

    def decode_plus(self, token_batch):
        sentences = []
        for tokens in token_batch:
            sentences.append(self.decode(tokens))
        return sentences

if __name__ == "__main__":
    import pandas as pd
    import os
    import pickle
    caption_texts = [
        ('../TEDLIUM/TEDLIUM_release1/train/preprocessed_label.csv', 'text'),
        ('../LibriSpeech/train-clean-360/preprocessed_label.csv', 'text'),
        ('../common_voice/preprocessed_train.csv', 'sentence'),
        # ('../youtube-speech-text/preprocessed_english_meta.csv', 'Normalized Transcription')
    ]
    if not os.path.exists('raw_corpus.txt'):
        with open('raw_corpus.txt', 'w') as f:
            for csv_filename, col_name in caption_texts:
                texts = list(pd.read_csv(csv_filename)[col_name])
                for t in texts:
                    t.replace('<eos>', '')
                    f.write(t+'\n')
    tokenizer = CharBPETokenizer(lowercase=True)

    tokenizer.train(["raw_corpus.txt"], vocab_size=1000,
        min_frequency=2,
        special_tokens=[
            "<blank>",
            "<bos>",
            "<unk>",
        ],
    )

    # os.makedirs('./BPE-1000', exist_ok=True)
    tokenizer.save(f'./BPE-1000','')

    tokenizer = CharBPETokenizer(
        './BPE-1000/-vocab.json',
        './BPE-1000/-merges.txt'
    )    
    # with open('.test.pkl', 'w') as f:
    #     pickle.dump(tokenizer, f)

    tokenizer = HuggingFaceTokenizer()
    print(tokenizer.encode('might have a solution it might take a long time nobody'))

    print(tokenizer.decode(
        tokenizer.encode('might have a solution it might take a long time nobody'),
    ))

    # transforms = torchaudio.transforms.MFCC(n_mfcc=40)
    # concat = ConcatFeature()
    # waveform = transforms(data)
    # print(waveform.shape)
    # waveform = concat(waveform)
    # print(waveform[:, -1])