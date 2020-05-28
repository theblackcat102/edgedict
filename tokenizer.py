import string
import torch


NUL = 0
PAD = 1
BOS = 2
EOS = 3
UNK = 4
DEFAULT_TOKEN2ID = {
    '<nul>': NUL,
    '<pad>': PAD,
    '<bos>': BOS,
    '<eos>': EOS,
    '<unk>': UNK,
}


def zero_pad_concat(inputs):
    # Pad audio feature sets
    max_t = max(inp.shape[0] for inp in inputs)
    shape = (len(inputs), max_t) + inputs[0].shape[1:]

    input_mat = torch.zeros(shape)
    for e, inp in enumerate(inputs):
        input_mat[e, :inp.shape[0]] = inp

    return input_mat


def end_pad_concat(inputs):
    # Pad text token sets
    max_t = max(i.shape[0] for i in inputs)
    shape = (len(inputs), max_t)
    labels = torch.full(shape, fill_value=PAD).long()
    for e, l in enumerate(inputs):
        labels[e, :len(l)] = l
    return labels


class CharTokenizer():
    def __init__(self):
        valid_tokens = string.ascii_lowercase + string.punctuation + ' '

        self.token2id = dict(DEFAULT_TOKEN2ID)

        self.id2token = {}
        for idx, token in enumerate(valid_tokens):
            self.token2id[token] = idx + 4

        for token, idx in self.token2id.items():
            self.id2token[idx] = token

        self.vocab_size = len(self.id2token)

    def encode(self, text, max_length=None):
        text = str(text).lower()
        text = text[:max_length]
        text = [self.token2id.get(char, UNK) for char in text]
        return text

    def decode(self, tokens):
        text = ''.join([self.id2token.get(token, '') for token in tokens])
        text = text.replace('<pad>', '')
        text = text.replace('<eos>', '')
        return text

    def decode_plus(self, token_batch):
        sentences = []
        for tokens in token_batch:
            sentences.append(self.decode(tokens))
        return sentences
