import os
import tempfile
import pickle

from tokenizers import CharBPETokenizer

NUL = 0
PAD = 1
BOS = 2
UNK = 3
NUL_token = '<nul>'
PAD_token = '<pad>'
BOS_token = '<bos>'
UNK_token = '<unk>'
DEFAULT_TOKEN2ID = {
    NUL_token: NUL,
    PAD_token: PAD,
    BOS_token: BOS,
    UNK_token: UNK,
}
DEFAULT_ID2TOKEN = {v: k for k, v in DEFAULT_TOKEN2ID.items()}


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
        os.makedirs(self.cache_dir)
        pickle.dump(self.token2id,
                    open(os.path.join(self.cache_dir, "token2id.pkl"), "wb"))

    def encode(self, text, max_length=None):
        text = str(text).lower()
        text = text[:max_length]
        text = [self.token2id.get(char, UNK) for char in text]
        return text

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

        vocab = os.path.join(self.cache_dir, self.name + '-vocab.json')
        merges = os.path.join(self.cache_dir, self.name + '-merges.txt')
        if os.path.exists(vocab) and os.path.exists(merges):
            self.tokenizer = CharBPETokenizer(vocab, merges, lowercase=True)
            print('Using cached HuggingFaceTokenizer')

    def build(self, texts):
        if self.tokenizer is not None:
            return

        tmp_file = tempfile.NamedTemporaryFile()

        with open(tmp_file.name, "w") as f:
            f.write(' '.join(texts).lower())

        self.tokenizer = CharBPETokenizer(lowercase=True)
        self.tokenizer.train(
            [tmp_file.name],
            vocab_size=self.vocab_size,
            special_tokens=[
                NUL_token,
                PAD_token,
                BOS_token,
                UNK_token,
            ],
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        self.tokenizer.save(self.cache_dir, self.name)

    def encode(self, text):
        token_ids = self.tokenizer.encode(text.lower()).ids
        token_ids = token_ids[:self.max_length]

        return token_ids

    def decode(self, tokens, skip_special_tokens=True):
        text = self.tokenizer.decode(                   # My special tokens
            [token for token in tokens if token > 3],   # aren't skipped
            skip_special_tokens=skip_special_tokens,    # even I set fucking
        )                                               # skip_special_tokens
        return text                                     # to True

    def decode_plus(self, token_batch):
        sentences = []
        for tokens in token_batch:
            sentences.append(self.decode(tokens))
        return sentences


if __name__ == "__main__":
    
    tokenizer = HuggingFaceTokenizer('BPE-2048',vocab_size=2048)

    texts = [
        'might have a solution it might take a long time nobody  wrote down the rules clearly  who  designed this'
    ]
    for text in texts:
        encoded = tokenizer.encode("%s" % text)
        decoded = tokenizer.decode(encoded)
        print(text)
        print(decoded)
