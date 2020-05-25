import os
import time
import random
import argparse
import logging
import numpy as np
import torch
from torch import nn, autograd
from torch.autograd import Variable
import torch.nn.functional as F
import torchaudio
from models import Transducer
from dataset import CommonVoice, seq_collate, YoutubeCaption, MergedDataset
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='RNN-T')
parser.add_argument('--name', type=str, default='rnn-t')

parser.add_argument('--lr', type=float, default=2e-4,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('-b', '--batch_size', type=int, default=2, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--layers', type=int, default=2,
                    help='number rnn layers')
parser.add_argument('--audio-feat', default=40, type=int, 
                    help='audio feature dimension size')
parser.add_argument('--bi', default=False, action='store_true', 
                    help='whether use bidirectional lstm')
parser.add_argument('--apex', default=False, action='store_true', 
                    help='use mix precision')
parser.add_argument('--opt_level', default='O1', type=str, 
                    help='operation level')
parser.add_argument('--noise', default=False, action='store_true',
                    help='add Gaussian weigth noise')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval')
parser.add_argument('--stdout', default=False, action='store_true', help='log in terminal')
parser.add_argument('--out', type=str, default='exp/rnnt_lr1e-3',
                    help='path to save the final model')
parser.add_argument('--cuda', default=True, action='store_false')
parser.add_argument('--init', type=str, default='',
                    help='Initial am & pm parameters')

parser.add_argument('--initam', type=str, default='',
                    help='Initial am parameters')
parser.add_argument('--gradclip', default=False, action='store_true')
parser.add_argument('--schedule', default=False, action='store_true')

args = parser.parse_args()
if args.apex:
    from apex import amp


class Trainer():

    def __init__(self, args):
        transforms = torchaudio.transforms.MFCC(n_mfcc=args.audio_feat)
        dataset = CommonVoice(
         '../common_voice', transforms=[transforms]   
        )
        yt_dataset = YoutubeCaption('../youtube-speech-text/', transforms=[transforms])
        dataset = MergedDataset([dataset, yt_dataset])

        self.dataloader = DataLoader(dataset, collate_fn=seq_collate, batch_size=args.batch_size, num_workers=4)

        val_dataset = CommonVoice(
             '../common_voice', labels='test.tsv',transforms=[transforms]   
        )
        self.tokenizer = val_dataset.tokenizer
        self.val_dataloader = DataLoader(val_dataset, collate_fn=seq_collate, batch_size=1, num_workers=4)

        self.args = args
        self.model = Transducer(args.audio_feat, dataset.vocab_size,
            32,
            64, 
            args.layers).cuda()
        self.gradclip = args.gradclip
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        if args.apex:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                opt_level=args.opt_level)


    def test(self):
        for batch in self.dataloader:
            xs, ys, xlen, ylen = batch
            xs, ys, xlen, ylen = xs.cuda(), ys.cuda(), xlen.cuda(), ylen.cuda()
            loss = self.model(xs[:5], ys[:5], xlen[:5], ylen[:5])
            break

    def evaluate(self, evaluate_size=1000):
        from jiwer import wer
        wers = []
        with torch.no_grad():
            for batch in self.val_dataloader:
                xs, ys, xlen, ylen = batch
                xs, ys = xs.cuda(), ys
                y, nll = self.model.greedy_decode(xs)
                hypothesis = self.tokenizer.decode(y)
                ground_truth = self.tokenizer.decode(ys[0].numpy())
                error = wer(ground_truth, hypothesis)
                wers.append(error)
                if len(wers) > evaluate_size:
                    break

        return np.mean(wers), np.std(wers)

    def train(self):

        from datetime import datetime
        cur_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        # writer = SummaryWriter('logs/{}-{}'.format(self.args.name, cur_time))

        for epoch in range(self.args.epochs):
            print(f'[epoch: {epoch+1}]')

            with tqdm(total=len(self.dataloader), dynamic_ncols=True) as pbar:

                for batch in self.dataloader:
                    xs, ys, xlen, ylen = batch
                    xs, ys, xlen, ylen = xs.cuda(), ys.cuda(), xlen.cuda(), ylen.cuda()
                    loss = self.model(xs, ys, xlen, ylen)
                    self.optimizer.zero_grad()

                    if self.args.apex:
                        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    if self.gradclip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    pbar.update(1)

                    pbar.set_description(
                        'loss: %.4f' % (loss.item()))

            if epoch % 10 == 0 and epoch > 100:
                self.model.eval()
                print(self.evaluate())
                self.model.train()

if __name__ == "__main__":
    trainer = Trainer(args)
    trainer.train()