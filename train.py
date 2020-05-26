import argparse
import textwrap
from datetime import datetime

import numpy as np
import torch
import torchaudio
from tensorboardX import SummaryWriter
import jiwer

from models import Transducer
from dataset import (
    seq_collate, MergedDataset,
    Librispeech,
    # CommonVoice,
    # YoutubeCaption,
    # TEDLIUM,
)
from tqdm import tqdm
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='RNN-T')
parser.add_argument('--name', type=str, default='rnn-t')

parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--layers', type=int, default=2,
                    help='number rnn layers')
parser.add_argument('--audio_feat', default=40, type=int,
                    help='audio feature dimension size')
parser.add_argument('--bi', default=False, action='store_true',
                    help='whether use bidirectional lstm')
parser.add_argument('--apex', default=False, action='store_true',
                    help='use mix precision')
parser.add_argument('--opt_level', default='O1', type=str, 
                    help='operation level')
parser.add_argument('--gradclip', default=False, action='store_true')

args = parser.parse_args()
if args.apex:
    from apex import amp


class Trainer():
    def __init__(self, args):
        transforms = torchaudio.transforms.MFCC(
            n_mfcc=args.audio_feat,
            melkwargs={'n_fft': 1024, 'win_length': 1024})

        # common_voice = CommonVoice(
        #     '../common_voice', labels='train.tsv', transforms=[transforms])
        librispeech = Librispeech(
            '../LibriSpeech/train-clean-360/', transforms=[transforms])
        # yt_dataset = YoutubeCaption(
        #     '../youtube-speech-text/', transforms=[transforms])
        # tedlium = TEDLIUM(
        #     '../TEDLIUM/TEDLIUM_release1/train/', transforms=[transforms])

        dataset = MergedDataset([librispeech])
        self.dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=4, collate_fn=seq_collate)

        # val_dataset = CommonVoice(
        #      '../common_voice', labels='test.tsv', transforms=[transforms])
        val_dataset = Librispeech(
            '../LibriSpeech/test-clean/', transforms=[transforms])
        self.val_dataloader = DataLoader(
            val_dataset, collate_fn=seq_collate, batch_size=16, num_workers=4)
        self.tokenizer = val_dataset.tokenizer

        self.args = args
        self.model = Transducer(
            args.audio_feat, dataset.vocab_size,
            16,     # vocab embedding dim
            128,    # hidden dim
            args.layers).cuda()
        self.gradclip = args.gradclip
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        if args.apex:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=args.opt_level)

        cur_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.writer = SummaryWriter(
            'logs/{}-{}'.format(self.args.name, cur_time))
        self.log_pattern = textwrap.dedent(
            '''
            `%s`
            `%s`
            ---
            ''')

    def evaluate(self, epoch, evaluate_size=1000, write_size=100):
        self.model.eval()
        write_count = 0
        wers = []
        with torch.no_grad():
            with tqdm(self.val_dataloader, dynamic_ncols=True) as pbar:
                pbar.set_description('evaluate')
                for batch in pbar:
                    xs, ys, xlen, ylen = batch
                    xs = xs.cuda()
                    ys_hat, nll = self.model.greedy_decode(xs, xlen)
                    for seq, seq_gt in zip(ys_hat, ys):
                        pred_seq = self.tokenizer.decode(seq)
                        true_seq = self.tokenizer.decode(seq_gt.numpy())
                        measures = jiwer.compute_measures(true_seq, pred_seq)
                        wers.append(measures['wer'])
                        if write_count < write_size:
                            self.writer.add_text(
                                "visualize/%d" % write_count,
                                self.log_pattern % (true_seq, pred_seq),
                                epoch)
                            write_count += 1
                # can measure others: mer, wil
                # if len(wers) > evaluate_size:
                #     break
        self.model.train()

        return np.mean(wers), np.std(wers)

    def train(self):
        print(self.evaluate(0))
        for epoch in range(self.args.epochs):
            print(f'[epoch: {epoch + 1}]')
            with tqdm(total=len(self.dataloader), dynamic_ncols=True) as pbar:
                for batch in self.dataloader:
                    xs, ys, xlen, ylen = [x.cuda() for x in batch]
                    loss = self.model(xs, ys, xlen, ylen)
                    self.optimizer.zero_grad()

                    if self.args.apex:
                        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    if self.gradclip:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 10.0)
                    self.optimizer.step()

                    pbar.update(1)
                    pbar.set_description('loss: %.4f' % (loss.item()))
            print(self.evaluate(epoch + 1))


if __name__ == "__main__":
    trainer = Trainer(args)
    trainer.train()
