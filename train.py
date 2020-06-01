import argparse
import os
import json
from datetime import datetime

import jiwer
import torch
import torchaudio.transforms as transforms
import numpy as np
import torch.optim as optim
from apex import amp
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import trange, tqdm
from warprnnt_pytorch import RNNTLoss

from models import Transducer
from tokenizer import NUL, HuggingFaceTokenizer, CharTokenizer
from dataset import seq_collate, MergedDataset, Librispeech


parser = argparse.ArgumentParser(description='RNN-T')
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--name_pattern', type=str, default=(
                    "E{enc_layers}D{dec_layers}H{hidden_size}-"
                    "F{n_fft}W{win_length}H{hop_length}"),
                    help="if --name is None, name_pattern is used")
parser.add_argument('--eval_model', type=str, default=None,
                    help='path to model, only evaluate and exit')
# learning
parser.add_argument('--optimizer', default="adam", choices=['adam', 'sgd'],
                    help='initial learning rate')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='initial learning rate')
parser.add_argument('--scheduler', action='store_true',
                    help='reduce lr on plateau')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch size')
parser.add_argument('--sub_batch_size', type=int, default=8,
                    help='sub batch size')
parser.add_argument('--eval_batch_size', type=int, default=8,
                    help='batch size')
parser.add_argument('--gradclip', default=None, type=float,
                    help='clip norm value')
# model
parser.add_argument('--vocab_embed_size', type=int, default=16,
                    help='vocab embedding dim')
parser.add_argument('--hidden_size', type=int, default=256,
                    help='RNN hidden dimension')
parser.add_argument('--enc_layers', type=int, default=4,
                    help='number rnn layers')
parser.add_argument('--enc_dropout', type=float, default=0.,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dec_layers', type=int, default=2,
                    help='number rnn layers')
parser.add_argument('--dec_dropout', type=float, default=0.,
                    help='dropout applied to layers (0 = no dropout)')
# data preprocess
parser.add_argument('--audio_max_length', type=int, default=14,
                    help='audio max length')
parser.add_argument('--audio_feat_size', type=int, default=40,
                    help='audio feature dimension size')
parser.add_argument('--n_fft', type=int, default=1024,
                    help='window size of fft')
parser.add_argument('--win_length', type=int, default=1024,
                    help='window length of frame')
parser.add_argument('--hop_length', type=int, default=512,
                    help='hop length between frame')
parser.add_argument('--sample_frame', type=int, default=1,
                    help='downsample audio feature by concatenating')
parser.add_argument('--tokenizer', default='char', choices=['char', 'bpe'],
                    help='tokenizer')
parser.add_argument('--bpe_size', type=int, default=256,
                    help='BPE vocabulary size')
# apex
parser.add_argument('--apex', default=False, action='store_true',
                    help='use mix precision')
parser.add_argument('--opt_level', default='O1', type=str,
                    help='operation level')
# parallel
parser.add_argument('--multi_gpu', action='store_true',
                    help='DataParallel')
# log
parser.add_argument('--save_step', type=int, default=50000,
                    help='frequency to save model')
parser.add_argument('--eval_step', type=int, default=10000,
                    help='frequency to save model')
parser.add_argument('--sample_size', type=int, default=20,
                    help='size of visualized examples')
args = parser.parse_args()
device = torch.device('cuda:0')


def infloop(dataloader):
    epoch = 1
    while True:
        for batch in dataloader:
            yield batch, epoch
        epoch += 1


class Trainer:
    def __init__(self, args):
        self.args = args
        self.args.name = self.args.name_pattern.format(**vars(self.args))
        current = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.args.logdir = os.path.join('logs', '%s-%s' % (args.name, current))
        self.args.model_dir = os.path.join(args.logdir, 'models')
        self.writer = SummaryWriter(args.logdir)
        self.writer.add_text('args', '`%s`' % json.dumps(vars(args)))
        print(json.dumps(vars(args)))

        transform = torch.nn.Sequential(
            transforms.MFCC(
                n_mfcc=args.audio_feat_size,
                melkwargs={
                    'n_fft': args.n_fft,
                    'win_length': args.win_length,
                    'hop_length': args.hop_length}))

        if args.tokenizer == 'bpe':
            self.tokenizer = HuggingFaceTokenizer(
                cache_dir=args.logdir, vocab_size=args.bpe_size)
        else:
            self.tokenizer = CharTokenizer(cache_dir=args.logdir)

        self.dataloader_train = DataLoader(
            dataset=MergedDataset([
                Librispeech(
                    '../LibriSpeech/train-clean-360/',
                    tokenizer=self.tokenizer,
                    transforms=transform,
                    audio_max_length=args.audio_max_length)]),
            batch_size=args.batch_size, shuffle=True, num_workers=4,
            collate_fn=seq_collate, drop_last=True)

        self.dataloader_val = DataLoader(
            dataset=MergedDataset([
                Librispeech(
                    '../LibriSpeech/test-clean/',
                    tokenizer=self.tokenizer,
                    transforms=transform,
                    reverse_sorted_by_length=True)]),
            batch_size=args.eval_batch_size, shuffle=False, num_workers=4,
            collate_fn=seq_collate)

        self.tokenizer.build(self.dataloader_train.dataset.texts())

        self.model = Transducer(
            vocab_size=self.dataloader_train.dataset.tokenizer.vocab_size,
            vocab_embed_size=args.vocab_embed_size,
            audio_feat_size=args.audio_feat_size * args.sample_frame,
            hidden_size=args.hidden_size,
            enc_layers=args.enc_layers,
            enc_dropout=args.enc_dropout,
            dec_layers=args.dec_layers,
            dec_dropout=args.dec_dropout,
            proj_size=args.hidden_size,
        ).to(device)

        if args.optimizer == 'adam':
            self.optim = optim.Adam(
                self.model.parameters(), lr=args.lr)
        else:
            self.optim = optim.SGD(
                self.model.parameters(), lr=args.lr, momentum=0.9)
        if args.scheduler:
            self.sched = optim.lr_scheduler.ReduceLROnPlateau(
                self.optim, patience=2, factor=0.5)
        else:
            self.sched = None
        self.loss_fn = RNNTLoss(blank=NUL)

        if args.apex:
            self.model, self.optim = amp.initialize(
                self.model, self.optim, opt_level=args.opt_level)
        if args.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)

        self.tokenizer = self.dataloader_train.dataset.tokenizer

    def train(self):
        looper = infloop(self.dataloader_train)
        losses = []
        steps = len(self.dataloader_train) * self.args.epochs
        with trange(steps, dynamic_ncols=True) as pbar:
            for step in pbar:
                batch, epoch = next(looper)
                loss = self.train_step(batch)
                losses.append(loss)
                pbar.set_description('Epoch %d, loss: %.4f' % (epoch, loss))

                if step > 0 and step % 5 == 0:
                    train_loss = torch.stack(losses).mean()
                    self.writer.add_scalar('train_loss', train_loss, step)
                    losses = []

                if step > 0 and step % self.args.save_step == 0:
                    self.save(step)

                if step > 0 and step % self.args.eval_step == 0:
                    pbar.set_description('Evaluating ...')
                    val_loss, wer, pred_seqs, true_seqs = self.evaluate()
                    self.writer.add_scalar('WER', wer, step)
                    self.writer.add_scalar('val_loss', val_loss, step)
                    for i in range(self.args.sample_size):
                        log = "`%s`\n\n`%s`" % (true_seqs[i], pred_seqs[i])
                        self.writer.add_text('val/%d' % i, log, step)
                    pbar.write(
                        'Epoch %d, step %d, loss: %.4f, WER: %.4f' % (
                            epoch, step, val_loss, wer))

    def train_step(self, batch):
        batch = [x.to(device) for x in batch]
        sub_losses = []
        start_idxs = range(0, self.args.batch_size, self.args.sub_batch_size)
        self.optim.zero_grad()
        for sub_batch_idx, start_idx in enumerate(start_idxs):
            sub_slice = slice(start_idx, start_idx + self.args.sub_batch_size)
            xs, ys, xlen, ylen = [x[sub_slice] for x in batch]
            xs = xs[:, :xlen.max()]
            ys = ys[:, :ylen.max()].contiguous()
            prob = self.model(xs, ys)
            loss = self.loss_fn(prob, ys, xlen, ylen) / len(start_idxs)
            if self.args.apex:
                delay_unscale = sub_batch_idx < len(start_idxs) - 1
                with amp.scale_loss(
                        loss,
                        self.optim,
                        delay_unscale=delay_unscale) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            sub_losses.append(loss.detach())

        if self.args.gradclip is not None:
            if self.args.apex:
                parameters = amp.master_params(self.optim)
            else:
                parameters = self.model.parameters()
            torch.nn.utils.clip_grad_norm_(parameters, self.args.gradclip)
        self.optim.step()

        loss = torch.stack(sub_losses).sum()
        return loss

    def evaluate(self):
        self.model.eval()
        wer = []
        losses = []
        pred_seqs = []
        true_seqs = []
        with torch.no_grad():
            with tqdm(self.dataloader_val, dynamic_ncols=True) as pbar:
                for batch in pbar:
                    xs, ys, xlen, ylen = [x.to(device) for x in batch]
                    prob = self.model(xs, ys)
                    loss = self.loss_fn(prob, ys, xlen, ylen)
                    losses.append(loss.item())

                    xs = xs.to(device)
                    if args.multi_gpu:
                        ys_hat, nll = self.model.module.greedy_decode(xs, xlen)
                    else:
                        ys_hat, nll = self.model.greedy_decode(xs, xlen)
                    pred_seq = self.tokenizer.decode_plus(ys_hat)
                    true_seq = self.tokenizer.decode_plus(ys.cpu().numpy())
                    wer.append(jiwer.wer(true_seq, pred_seq))
                    pbar.set_description('wer: %.4f' % wer[-1])
                    sample_nums = self.args.sample_size - len(pred_seqs)
                    pred_seqs.extend(pred_seq[:sample_nums])
                    true_seqs.extend(true_seq[:sample_nums])
        loss = np.mean(losses)
        wer = np.mean(wer)
        self.model.train()
        return loss, wer, pred_seqs, true_seqs

    def save(self, step):
        if not os.path.exists(self.args.model_dir):
            os.mkdir(self.args.model_dir)
        checkpoint = {'optim': self.optim.state_dict()}

        if self.sched is not None:
            checkpoint.update({'sched': self.sched.state_dict()})

        if isinstance(self.model, torch.nn.DataParallel):
            checkpoint = {'model': self.model.module.state_dict()}
        else:
            checkpoint = {'model': self.model.state_dict()}

        if self.args.apex:
            checkpoint.update({'amp': amp.state_dict()})
        path = os.path.join(self.args.model_dir, 'epoch-%d' % step)
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(args.eval_model)
        # self.optim.load_state_dict(checkpoint['optim'])

        if self.sched is not None:
            self.sched.load_state_dict(checkpoint['sched'])

        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])

        # if self.args.apex:
        #     amp.load_state_dict(checkpoint['amp'])


if __name__ == "__main__":

    trainer = Trainer(args)

    if args.eval_model:
        trainer.load(args.eval_model)
        val_loss, wer, pred_seqs, true_seqs = trainer.evaluate()
        for pred_seq, true_seq in zip(pred_seqs, true_seqs):
            print('True: %s\n\nPred:%s\n')
            print('=' * 20)
        print('Evaluate, loss: %.4f, WER: %.4f' % (val_loss, wer))
    else:
        trainer.train()

    # model.eval()
    # with torch.no_grad():
    #     batch = next(iter(val_dataloader))
    #     xs, ys, xlen, ylen = [x.to(device) for x in batch]
    #     prob = model(xs, ys)
    #     loss = loss_fn(prob, ys, xlen, ylen)
    # model.train()
