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
from warprnnt_pytorch import RNNTLoss
from models import Transducer
from dataset import (
    CommonVoice, 
    YoutubeCaption,
    Librispeech,
    Synthetic,
    TEDLIUM,
    seq_collate, MergedDataset
)
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from tokenizer import HuggingFaceTokenizer, CharTokenizer
from augmentation import ConcatFeature, TimeMask, FreqMask
from tensorboardX import SummaryWriter
import json
import jiwer
from parts.features import AudioPreprocessing
from parts.text.cleaners import english_cleaners
from plot_utils import plot_alignment_to_numpy
from recurrent import MFCC_
from tokenizers import CharBPETokenizer

from torch.nn.utils.rnn import pack_padded_sequence

parser = argparse.ArgumentParser(description='RNN-T')
parser.add_argument('--name', type=str, default='rnn-t')

parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('-b', '--batch_size', type=int, default=2, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--layers', type=int, default=2,
                    help='number rnn layers')
parser.add_argument('--pred-layers', type=int, default=1,
                    help='number rnn layers')

parser.add_argument('--tokenizer', type=str, default='bpe',
                    help='type of tokenizer: bpe, char', choices=['char', 'bpe'])

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

parser.add_argument('--data-path', type=str, default='../',
                    help='Path to all datasets')

parser.add_argument('--n-fft', type=int, default=768,
                    help='SFTT size')
parser.add_argument('--h-dim', type=int, default=128,
                    help='SFTT size')
parser.add_argument('--pred-dim', type=int, default=128,
                    help='prediction hidden dimension')
parser.add_argument('--vocab-dim', type=int, default=16,
                    help='vocab dim size')
parser.add_argument('--audio-feat', default=128, type=int, 
                    help='audio feature dimension size')
parser.add_argument('--warmup', default=10000, type=int, 
                    help='learning rate warmup')

parser.add_argument('--initam', type=str, default='',
                    help='Initial am parameters')
parser.add_argument('--gradclip', default=False, action='store_true')
parser.add_argument('--schedule', default=False, action='store_true')
parser.add_argument('--concat', default=False, action='store_true')
parser.add_argument('-a', '--accumulation-steps', default=1, type=int, help='gradient accumulation')

parser.add_argument('--log-alignment', default=10000, type=int, 
                    help='steps to log alignment')
parser.add_argument('--log-eval', default=60000, type=int, 
                    help='steps to log alignment')
parser.add_argument('--ckpt', default=None, type=str, 
                    help='steps to log alignment')


args = parser.parse_args()
if args.apex:
    from apex import amp


class Trainer():

    def __init__(self, args):
        cleaner = english_cleaners
        tokenizers = CharBPETokenizer(
            './BPE-1024/-vocab.json',
            './BPE-1024/-merges.txt',
            lowercase=True,
        )
        transforms_piplines = [
            # torchaudio.transforms.MelSpectrogram(
            #     # n_mfcc=args.audio_feat, 
            #     n_fft=args.n_fft, n_mels=args.audio_feat,
            #     # melkwargs={'n_fft':1024, 'win_length': 1024}
            # ),
            AudioPreprocessing(
                normalize='none', sample_rate=16000, window_size=0.02, 
                window_stride=0.01, features=args.audio_feat, n_fft=512, log=True,
                feat_type='logfbank', trim_silence=True, window='hann',dither=0.00001, frame_splicing=1, transpose_out=False
            ),
            TimeMask(T=40, num_masks=5, replace_with_zero=False),
            FreqMask(F=5, num_masks=5, replace_with_zero=False),
        ]
        val_pipeline = [
            # torchaudio.transforms.MelSpectrogram(
            #     # n_mfcc=args.audio_feat, 
            #     n_fft=args.n_fft, n_mels=args.audio_feat,
            #     # melkwargs={'n_fft':1024, 'win_length': 1024}
            # ),
            AudioPreprocessing(
                normalize='none', sample_rate=16000, window_size=0.02, 
                window_stride=0.01, features=args.audio_feat, n_fft=512, log=True,
                feat_type='logfbank', trim_silence=True, window='hann',dither=0.00001, frame_splicing=1, transpose_out=False
            ),
        ]
        if args.concat:
            val_pipeline.append(
                ConcatFeature(merge_size=3)
            )
        val_transform = torch.nn.Sequential(*val_pipeline)
        if len(val_pipeline) == 1:
            val_transform = val_pipeline[0]

        if args.concat:
            args.audio_feat *= 3
            transforms_piplines.append(
                 ConcatFeature(merge_size=3)
            )
        transforms = torch.nn.Sequential(*transforms_piplines)

        if args.tokenizer == 'char':
            _tokenizer = CharTokenizer()
        else:
            _tokenizer = HuggingFaceTokenizer(tokenizers=tokenizers, cleaner=cleaner) # use BPE-400

        common_voice = CommonVoice(f'{args.data_path}common_voice',
            audio_max_length=13,
            transforms=transforms, tokenizer=_tokenizer)
        synthetic = Synthetic(f'{args.data_path}synthetic',
            audio_max_length=13,
            transforms=transforms, tokenizer=_tokenizer)

        yt_dataset = YoutubeCaption(f'{args.data_path}youtube-speech-text/',
            audio_max_length=13,
            transforms=transforms, tokenizer=_tokenizer)
        yt2_dataset = YoutubeCaption(f'{args.data_path}youtube-speech-text/',
            labels='english2_meta.csv',
            audio_max_length=13,
            transforms=transforms, tokenizer=_tokenizer)
        yt3_dataset = YoutubeCaption(f'{args.data_path}youtube-speech-text/',
            labels='bloomberg_meta.csv',
            audio_max_length=13,
            transforms=transforms, tokenizer=_tokenizer)

        librispeech2 = Librispeech(f'{args.data_path}LibriSpeech/train-other-500/',
            audio_max_length=13,
            transforms=transforms, tokenizer=_tokenizer)
        librispeech = Librispeech(f'{args.data_path}LibriSpeech/train-clean-360/',
            audio_max_length=13,
            transforms=transforms, tokenizer=_tokenizer)
        tedlium = TEDLIUM(f'{args.data_path}TEDLIUM/TEDLIUM_release1/train/',
            audio_max_length=13,
            transforms=transforms, tokenizer=_tokenizer)
        tedlium2 = TEDLIUM(f'{args.data_path}TEDLIUM/TEDLIUM_release-3/data/',
            audio_max_length=13,
            transforms=transforms, tokenizer=_tokenizer)

        dataset = MergedDataset([common_voice, yt2_dataset, librispeech, tedlium, synthetic, tedlium2, librispeech2, yt_dataset])
        # dataset = MergedDataset([librispeech, librispeech2])
        
        # self.libri_dataloader = DataLoader(librispeech, collate_fn=seq_collate, batch_size=args.batch_size, 
        #     num_workers=4, shuffle=True, drop_last=True)

        self.dataloader = DataLoader(dataset, collate_fn=seq_collate, batch_size=args.batch_size, 
            num_workers=4, shuffle=True, drop_last=True)

        print(val_transform)
        # val_dataset = CommonVoice(
        #      f'{args.data_path}common_voice', labels='test.tsv',transforms=val_transform, tokenizer= _tokenizer  
        # )
        val_dataset = Librispeech(f'{args.data_path}LibriSpeech/test-clean/',
            audio_max_length=13,
            transforms=transforms, tokenizer=_tokenizer)
        self.loss_fn = RNNTLoss(blank=0)
        self.tokenizer = _tokenizer
        self.val_dataloader = DataLoader(val_dataset, collate_fn=seq_collate, batch_size=64, num_workers=4)

        self.args = args
        self.model = Transducer(args.audio_feat, dataset.vocab_size,
            args.vocab_dim, # vocab embedding dim
            args.h_dim, # hidden dim
            args.layers, pred_num_layers=args.pred_layers, dropout=args.dropout).cuda()
        self.gradclip = args.gradclip
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        if args.apex:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                opt_level=args.opt_level)

        if args.ckpt != None:
            print('load previous checkpoint')

            checkpoint = torch.load(args.ckpt)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'amp' in checkpoint:
                print('load previous amp')
                amp.load_state_dict(checkpoint['amp'])

        self.accu_steps = float(args.accumulation_steps)


    def test(self):
        for batch in self.dataloader:
            xs, ys, xlen, ylen = batch
            xs, ys, xlen, ylen = xs.cuda(), ys.cuda(), xlen.cuda(), ylen.cuda()
            loss = self.model(xs[:5], ys[:5], xlen[:5], ylen[:5])
            break

    def evaluate(self, evaluate_size=1000, steps=-1, writer=None):
        wers = []
        texts = []
        null_avg = []
        with torch.no_grad():
            for batch in self.val_dataloader:
                xs, ys, xlen, ylen = batch
                xs, ys, xlen = xs.cuda(), ys, xlen.cuda()
                y, nll = self.model.greedy_decode(xs, xlen)

                null_avg.append(nll.mean().item())

                hypothesis = self.tokenizer.decode_plus(y)
                ground_truth = self.tokenizer.decode_plus(ys.numpy())
                measures = jiwer.compute_measures(ground_truth, hypothesis)
                if len(texts) < 10:
                    texts.append((ground_truth[0], hypothesis[0]))
                wers.append(measures['wer'])

                # can measure others: mer, wil
                if len(wers) > evaluate_size:
                    break

        if writer != None:
            hypothesis, ground_truth = '', ''
            for (g_, h_) in texts:
                hypothesis += h_+'\n\n'
                ground_truth += g_ + '\n\n'
            writer.add_text('generated', hypothesis, steps)
            writer.add_text('grouth_truth', ground_truth, steps)
            writer.add_scalar('val/perplexity',  np.mean(null_avg), steps)

        return np.mean(wers), np.std(wers), np.mean(null_avg)


    def optimizer_step(self, steps):
        if steps < self.args.warmup:
            lr_scale = min(1., float(steps + 1) / self.args.warmup*1.0)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr_scale * self.args.lr

    def train(self):

        from datetime import datetime
        cur_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        log_name = '{}-{}-{}'.format(self.args.name, self.args.tokenizer,cur_time)
        log_path = 'logs/{}'.format(log_name)
        writer = SummaryWriter(log_path)
        with open('logs/{}/vars.json'.format(log_name), 'w') as f:
            json.dump(vars(self.args), f)

        if self.args.tokenizer == 'bpe':
            self.tokenizer.token.save(f'logs/{log_name}/BPE')
        else:
            with open('logs/{}/vocab.json'.format(log_name), 'w') as f:
                json.dump(self.tokenizer.token2id, f)


        best_wer = 1.0
        log_avg_steps = 5
        steps, avg_loss = 0, 0
        lmbda = lambda epoch: 0.9
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.optimizer, lr_lambda=lmbda)
        avg_wer, std_wer, avg_ppl = self.evaluate(steps=steps, writer=writer, evaluate_size=100)
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=4, factor=0.9)


        if self.accu_steps > 1:
            print('Accumulation steps %d' % self.accu_steps)
            self.model.zero_grad()

        if self.args.ckpt != None:
            print('change lr')
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = args.lr
        self.optimizer_step(steps)
        for epoch in range(self.args.epochs):
            print(f'[epoch: {epoch+1}]')
            lr_val = 0
            for param_group in self.optimizer.param_groups:
                lr_val = param_group['lr']
            writer.add_scalar('lr', lr_val, steps)

            with tqdm(total=len(self.dataloader), dynamic_ncols=True) as pbar:
                for batch in self.dataloader:
                    xs, ys, xlen, ylen = batch
                    xs, ys, xlen, ylen = xs.cuda(), ys.cuda(), xlen.cuda(), ylen.cuda()
                    # xs = pack_padded_sequence(xs, lengths=xlen, batch_first=True, enforce_sorted=False)
                    alignment = self.model(xs, ys, xlen, ylen)
                    if alignment.shape[1] != xs.shape[1]:
                        reduction_ratio = (xs.shape[1]/alignment.shape[1])
                        xlen = torch.round(xlen/reduction_ratio).int()
                    loss = self.loss_fn(alignment, ys.int(), xlen, ylen)

                    if self.accu_steps <= 1:
                        self.model.zero_grad()

                        if self.gradclip:
                            if self.args.apex:
                                torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), 200.0)
                            else:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 200.0)

                        if self.args.apex:
                            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()

                        if self.gradclip:
                            if self.args.apex:
                                torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), 200.0)
                            else:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 200.0)

                        self.optimizer.step()
                        self.optimizer_step(steps)

                    elif (steps+1) % int(self.accu_steps) == 0:

                        if self.gradclip:
                            if self.args.apex:
                                torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), 20.0)
                            else:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 20.0)

                        if self.args.apex:
                            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()

                        self.optimizer.step()
                        self.optimizer_step(steps)

                        self.model.zero_grad()

                    elif (steps+1) % int(self.accu_steps) != 0:

                        with amp.scale_loss(loss, self.optimizer, delay_unscale=True) as scaled_loss:
                            scaled_loss.backward()

                    pbar.update(1)

                    pbar.set_description(
                        'loss: %.4f' % (loss.item()))

                    avg_loss += loss.item()

                    if (steps+1) % log_avg_steps == 0:
                        avg_loss /= log_avg_steps
                        writer.add_scalar('loss', avg_loss, steps)
                        avg_loss = 0

                    if (steps+1) % self.args.log_alignment == 0:
                        idx = random.randint(0, alignment.size(0) - 1)
                        alignment = torch.softmax(alignment[idx], dim=-1)
                        alignment[:, :, 0] = 0 # ignore blank token
                        alignment = alignment.mean(dim=-1)

                        writer.add_image(
                                "alignment",
                                plot_alignment_to_numpy(alignment.data.cpu().numpy().T),
                                steps, dataformats='HWC')

                    if (steps+1) % self.args.log_eval == 0:
                        self.model.eval()
                        avg_wer, std_wer, avg_ppl = self.evaluate(steps=steps, writer=writer)
                        plateau_scheduler.step(avg_wer)
                        writer.add_scalar('wer', avg_wer, steps)

                        for param_group in self.optimizer.param_groups:
                            lr_val = param_group['lr']
                        writer.add_scalar('lr', lr_val, steps)

                        if best_wer > avg_wer:
                            print('best checkpoint found!')
                            checkpoint = {
                                'model': self.model.state_dict(),
                                'optimizer': self.optimizer.state_dict(),
                                'epoch': epoch
                            }
                            if self.args.apex:
                                checkpoint['amp'] = amp.state_dict()
                            torch.save(checkpoint, os.path.join(log_path, str(epoch)+'amp_checkpoint.pt'))
                            best_wer = avg_wer
                        self.model.train()
                    steps += 1

            
            scheduler.step()

            if epoch > -1:
                self.model.eval()
                avg_wer, std_wer, avg_ppl = self.evaluate(steps=steps, writer=writer)
                plateau_scheduler.step(avg_wer)
                writer.add_scalar('wer', avg_wer, steps)
                if best_wer > avg_wer:
                    print('best checkpoint found!')
                    checkpoint = {
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'epoch': epoch
                    }
                    if self.args.apex:
                        checkpoint['amp'] = amp.state_dict()
                    torch.save(checkpoint,os.path.join(log_path, str(epoch)+'amp_checkpoint.pt'))
                    best_wer = avg_wer
                self.model.train()
                writer.flush()

if __name__ == "__main__":
    trainer = Trainer(args)
    trainer.train()