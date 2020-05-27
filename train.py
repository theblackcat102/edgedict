import argparse
import os
import textwrap
from datetime import datetime

import numpy as np
import torch
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
from warprnnt_pytorch import RNNTLoss


parser = argparse.ArgumentParser(description='RNN-T')
parser.add_argument('--name', type=str, default='rnn-t')
# learning
parser.add_argument('--lr', type=float, default=4e-4,
                    help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=16,
                    help='batch size')
parser.add_argument('--gradclip', default=None, type=float,
                    help='clip norm value')
# model
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--num_layers', type=int, default=3,
                    help='number rnn layers')
parser.add_argument('--vocab_embed_size', type=int, default=16,
                    help='vocab embedding dim')
parser.add_argument('--hidden_size', type=int, default=256,
                    help='RNN hidden dimension')
parser.add_argument('--audio_feat', default=40, type=int,
                    help='audio feature dimension size')
parser.add_argument('--bidirectional', default=False, action='store_true',
                    help='whether use bidirectional lstm')
# apex
parser.add_argument('--apex', default=False, action='store_true',
                    help='use mix precision')
parser.add_argument('--opt_level', default='O1', type=str,
                    help='operation level')
# parallel
parser.add_argument('--multi_gpu', action='store_true',
                    help='DataParallel')
device = torch.device('cuda:0')

args = parser.parse_args()
if args.apex:
    from apex import amp


class Trainer():
    def __init__(self, args):
        self.args = args
        # transforms = transforms.MFCC(
        #     n_mfcc=args.audio_feat,
        #     melkwargs={'n_fft': 1024, 'win_length': 1024})
        sr = 16000
        transforms = Compose([
            LogMelSpectrogram(
                sample_rate=sr, win_length=0.025 * sr, hop_length=0.01 * sr,
                n_fft=512, f_min=125, f_max=7600, n_mels=80),
            DownsampleSpectrogram(n_frame=3)
        ])

        train_librispeech = Librispeech(
            '../LibriSpeech/train-clean-360/', transforms=[transforms])
        # train_common_voice = CommonVoice(
        #     '../common_voice', labels='train.tsv', transforms=[transforms])
        # train_yt_dataset = YoutubeCaption(
        #     '../youtube-speech-text/', transforms=[transforms])
        # train_tedlium = TEDLIUM(
        #     '../TEDLIUM/TEDLIUM_release1/train/', transforms=[transforms])

        train_dataset = MergedDataset([train_librispeech])
        self.dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=4, collate_fn=seq_collate)

        val_librispeech = Librispeech(
            '../LibriSpeech/test-clean/', transforms=[transforms])
        # val_common_voice = CommonVoice(
        #      '../common_voice', labels='test.tsv', transforms=[transforms])
        val_dataset = MergedDataset([val_librispeech])
        self.val_dataloader = DataLoader(
            val_dataset, batch_size=args.eval_batch_size, shuffle=False,
            num_workers=4, collate_fn=seq_collate)
        self.tokenizer = val_librispeech.tokenizer

        self.model = Transducer(
            args.audio_feat, train_dataset.vocab_size, args.vocab_embed_size,
            args.hidden_size, args.num_layers, args.dropout,
            args.bidirectional).to(device)
        if args.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=args.lr)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, mode='min', factor=0.5, patience=10, verbose=1,
            min_lr=1e-6)
        self.loss_fn = RNNTLoss()

        if args.apex:
            self.model, self.optim = amp.initialize(
                self.model, self.optim, opt_level=args.opt_level)

        current = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.logdir = os.path.join('logs', '%s-%s' % (self.args.name, current))
        self.writer = SummaryWriter(self.logdir)
        self.log_pattern = textwrap.dedent(
            '''
            `True: %s`

            `Pred: %s`
            ''')

    def save_model(self, epoch):
        if not os.path.exists(os.path.join(self.logdir, 'models')):
            os.mkdir(os.path.join(self.logdir, 'models'))
        if self.args.multi_gpu:
            ckpt = {'model': self.model.module.state_dict()}
        else:
            ckpt = {'model': self.model.state_dict()}
        ckpt.update({'optim': self.optim})
        torch.save(
            ckpt, os.path.join(self.logdir, 'models', 'epoch-%d' % epoch))

    def evaluate(self, epoch, evaluate_size=1000, write_size=100):
        self.model.eval()
        write_count = 0
        wers = []
        losses = []
        with torch.no_grad():
            with tqdm(self.val_dataloader, dynamic_ncols=True) as pbar:
                pbar.set_description('evaluate')
                for batch in pbar:
                    xs, ys, xlen, ylen = [x.to(device) for x in batch]

                    prob = self.model(xs, ys)
                    loss = self.loss_fn(prob, ys.int(), xlen, ylen)
                    losses.append(loss)

                    xs = xs.to(device)
                    if args.multi_gpu:
                        ys_hat, nll = self.model.module.greedy_decode(xs, xlen)
                    else:
                        ys_hat, nll = self.model.greedy_decode(xs, xlen)
                    for seq, seq_gt in zip(ys_hat, ys.cpu()):
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
        self.model.train()
        wer_mean, wer_std = np.mean(wers), np.std(wers)
        loss = torch.stack(losses).mean()
        self.writer.add_scalar('WER', wer_mean, epoch)
        self.writer.add_scalar('WER_STD', wer_std, epoch)
        self.writer.add_scalar('val_loss', loss)
        print('WER: %.5f(%.5f)' % (wer_mean, wer_std))
        return loss

    def train(self):
        self.evaluate(epoch=0)
        for epoch in range(1, self.args.epochs + 1):
            print(f'[epoch: {epoch}]')
            with tqdm(total=len(self.dataloader), dynamic_ncols=True) as pbar:
                for batch in self.dataloader:
                    xs, ys, xlen, ylen = [x.to(device) for x in batch]
                    prob = self.model(xs, ys)
                    loss = self.loss_fn(prob, ys.int(), xlen, ylen)
                    self.optim.zero_grad()
                    if self.args.apex:
                        with amp.scale_loss(loss, self.optim) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    if self.args.gradclip:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.gradclip)
                    self.optim.step()

                    self.writer.add_scalar('loss', loss)
                    pbar.update(1)
                    pbar.set_description('loss: %.4f' % (loss.item()))
            val_loss = self.evaluate(epoch)
            self.sched.step(val_loss)


if __name__ == "__main__":
    trainer = Trainer(args)
    trainer.train()
