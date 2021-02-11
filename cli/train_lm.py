import os
import time
import random
import argparse
import logging
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from modules.tokenizer import HuggingFaceTokenizer, CharTokenizer
from tensorboardX import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
import json
import jiwer
from torch.nn.utils.rnn import pack_padded_sequence
from models import LMModel



class TextDataset(Dataset):

    def __init__(self, txt_file, tokenizer):
        self.tokenizer = tokenizer
        self.corpus = []
        with open(txt_file, 'r') as f:
            for line in f.readlines():
                self.corpus.append(line.strip())

    def __len__(self):
        return len(self.corpus)

        
    def __getitem__(self, idx):
        line = self.corpus[idx]
        return torch.from_numpy(np.array(self.tokenizer.encode(line))).long()

def seq_collate(batches):
    inputs = []
    outputs = []
    outputs = pad_sequence(batches, batch_first=True, padding_value=0)
    # append <bos> to input
    inputs = torch.cat([torch.ones(outputs.shape[0], 1).long(), outputs], dim=1)
    return inputs[:, :-1], outputs

if __name__ == "__main__":

    model = LMModel(1024, 64, 1024, 2, tie_weights=False)
    model = model.cuda()
    txt_file = 'librispeech_corpus.txt'
    val_txt_file  = 'librispeech_corpus-dev.txt'
    # input_tokens = torch.randint(0, 1000, (10, 10))
    # hidden = model.init_hidden(10)
    # logits, hidden = model(input_tokens, hidden)
    from datetime import datetime

    tokenizer = HuggingFaceTokenizer()
    dataset = TextDataset(txt_file, tokenizer)
    val_dataset = TextDataset(val_txt_file, tokenizer)

    criterion = nn.NLLLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    bs = 256
    dataloader = DataLoader(dataset, batch_size=bs, collate_fn=seq_collate)
    eval_bs = 128
    val_dataloader = DataLoader(val_dataset, batch_size=eval_bs, collate_fn=seq_collate)

    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch in val_dataloader:
            inputs, targets = batch
            inputs, targets = inputs.cuda(), targets.cuda()
            hidden = model.init_hidden(inputs.shape[0])
            output, _ = model(inputs, hidden)
            total_loss += criterion(output, targets.flatten()).item()
    best_loss = total_loss / (len(val_dataloader) - 1)


    for epoch in range(100):
        print('[epoch %d]' % epoch)
        model.train()
        for batch in tqdm(dataloader, dynamic_ncols=True):
            inputs, targets = batch
            inputs, targets = inputs.cuda(), targets.cuda()
            hidden = model.init_hidden(inputs.shape[0])
            model.zero_grad()
            logits, _ = model(inputs, hidden)

            loss = criterion(logits, targets.flatten())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        model.eval()
        total_loss = 0.
        with torch.no_grad():
            for batch in val_dataloader:
                inputs, targets = batch
                inputs, targets = inputs.cuda(), targets.cuda()
                hidden = model.init_hidden(inputs.shape[0])
                output, _ = model(inputs, hidden)
                total_loss += criterion(output, targets.flatten()).item()
        eval_loss = total_loss / (len(val_dataloader) - 1)
        print(eval_loss)
        if eval_loss < best_loss:
            best_loss = eval_loss
            print('save checkpoint')
            torch.save(model.state_dict(), 'librispeech_lm_model.pt')