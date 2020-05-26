import string
import torch


def zero_pad_concat(inputs):
    # Pad audio feature sets
    max_t = max(inp.shape[0] for inp in inputs)
    shape = (len(inputs), max_t) + inputs[0].shape[1:]

    input_mat = torch.zeros(shape)
    for e, inp in enumerate(inputs):
        input_mat[e, :inp.shape[0]] = inp

    return input_mat


def end_pad_concat(inputs, pad_idx=0):
    # Pad text token sets
    max_t = max(i.shape[0] for i in inputs)
    shape = (len(inputs), max_t)
    labels = torch.full(shape, fill_value=pad_idx).long()
    for e, l in enumerate(inputs):
        labels[e, :len(l)] = l
    return labels


class CharTokenizer():

    def __init__(self):
        valid_tokens = string.ascii_lowercase + string.punctuation + ' '

        self.token2id = {
            '<pad>': 0,
            '<bos>': 1,
            '<eos>': 2,
            '<unk>': 3,
        }

        self.id2token = {}
        for idx, token in enumerate(valid_tokens):
            self.token2id[token] = idx+4

        for token, idx in self.token2id.items():
            self.id2token[idx] = token

        self.vocab_size = len(self.id2token)

    def encode(self, text, max_length=-1):
        text = text.lower()
        if max_length > 1:
            text = text[:max_length]
        return [self.token2id[char] if char in self.token2id else 3 for char in text]

    def decode(self, tokens):
        text = ''.join([self.id2token[t] if t in self.id2token else '' for t in tokens ])
        text = text.replace('<pad>', '')
        text = text.replace('<eos>', '')
        return text

    def decode_plus(self, token_batch):
        sentences = []
        for tokens in token_batch:
            sentences.append(self.decode(tokens))
        return sentences
