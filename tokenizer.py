import os
import tempfile
import pickle

import torch
from tokenizers import CharBPETokenizer

NUL = 0
PAD = 1
BOS = 2
EOS = 3
UNK = 4
NUL_token = '<nul>'
PAD_token = '<pad>'
BOS_token = '<bos>'
EOS_token = '<eos>'
UNK_token = '<unk>'
DEFAULT_TOKEN2ID = {
    NUL_token: NUL,
    PAD_token: PAD,
    BOS_token: BOS,
    EOS_token: EOS,
    UNK_token: UNK,
}
DEFAULT_ID2TOKEN = {v: k for k, v in DEFAULT_TOKEN2ID.items()}


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


class CharTokenizer:
    def __init__(self, cache_dir, max_length=None):
        self.cache_dir = cache_dir

    def load(self):
        self.token2id = pickle.load(
            open(os.path.join(self.cache_dir, "token2id.pkl"), "rb"))
        self.id2token = [None for _ in range(len(self.token2id))]
        for token, idx in self.token2id.items():
            self.id2token[idx] = token
        self.vocab_size = len(self.token2id)

    def build(self, texts):
        self.token2id = dict(DEFAULT_TOKEN2ID)
        chars = sorted(list(set(''.join(texts).lower())))
        for char in chars:
            idx = len(self.token2id)
            self.token2id[char] = idx
        self.id2token = [None for _ in range(len(self.token2id))]
        for token, idx in self.token2id.items():
            self.id2token[idx] = token
        self.vocab_size = len(self.token2id)
        pickle.dump(self.token2id,
                    open(os.path.join(self.cache_dir, "token2id.pkl"), "wb"))

    def encode(self, text, max_length=None):
        text = str(text).lower()
        text = text[:max_length]
        text = [self.token2id.get(char, UNK) for char in text]
        return [BOS] + text + [EOS]

    def decode(self, tokens):
        text = ''.join([self.id2token[token] for token in tokens])
        for token in DEFAULT_TOKEN2ID.keys():
            text = text.replace(token, '')
        return text

    def decode_plus(self, token_batch):
        sentences = []
        for tokens in token_batch:
            sentences.append(self.decode(tokens))
        return sentences


class HuggingFaceTokenizer:
    def __init__(self, cache_dir, max_length=None, vocab_size=400):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.name = "%d-%s" % (vocab_size, max_length)
        self.tokenizer = None

    def build(self, texts):
        tmp_file = tempfile.NamedTemporaryFile()

        with open(tmp_file.name, "w") as f:
            f.write(' '.join(texts).lower())

        self.tokenizer = CharBPETokenizer()
        self.tokenizer.train(
            [tmp_file.name],
            vocab_size=self.vocab_size,
            special_tokens=[
                NUL_token,
                PAD_token,
                BOS_token,
                EOS_token,
            ],
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        self.tokenizer.save(self.cache_dir, self.name)

    def encode(self, text):
        text = "%s %s %s" % (BOS_token, text.lower(), EOS_token)
        token_ids = self.tokenizer.encode(text).ids
        token_ids = token_ids[:self.max_length]

        return token_ids

    def decode(self, tokens, skip_special_tokens=True):
        text = self.tokenizer.decode(
            list(tokens),
            skip_special_tokens=skip_special_tokens,
        )
        return text

    def decode_plus(self, token_batch):
        sentences = []
        for tokens in token_batch:
            sentences.append(self.decode(tokens))
        return sentences


if __name__ == "__main__":
    texts = ["ab asd fbdbfd ff", "sdca a dsa  ads  a"]
    tokenizer = HuggingFaceTokenizer(vocab_size=20)
    tokenizer.build(texts=texts)

    for text in texts:
        encoded = tokenizer.encode("%s" % text)
        decoded = tokenizer.decode(encoded)
        print(text)
        print(decoded)
