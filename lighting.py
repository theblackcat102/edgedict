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
from recurrent import MFCC_
from dataset import (
    CommonVoice, 
    YoutubeCaption,
    Synthetic,
    Librispeech,
    TEDLIUM,
    seq_collate, MergedDataset
)
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from tokenizer import HuggingFaceTokenizer, CharTokenizer
from augmentation import ConcatFeature, TimeMask, FreqMask, TimeWrap
from tensorboardX import SummaryWriter
import json
import jiwer
from plot_utils import plot_alignment_to_numpy
from torch.nn.utils.rnn import pack_padded_sequence
from apex import amp
import pytorch_lightning as pl
from train import args
from parts.features import AudioPreprocessing


class ParallelTraining(pl.LightningModule):
    def __init__(self, args):
        super(ParallelTraining, self).__init__()
        if args.tokenizer == 'char':
            _tokenizer = CharTokenizer()
        else:
            print('use BPE 1000')
            _tokenizer = HuggingFaceTokenizer() # use BPE-1000
        audio_feature = args.audio_feat
        if args.concat:
            audio_feature *= 3

        self.tokenizer = _tokenizer
        self.loss_fn = RNNTLoss(blank=0)
        self.model = Transducer(audio_feature, _tokenizer.vocab_size,
            args.vocab_dim, # vocab embedding dim
            args.h_dim, # hidden dim
            args.layers, pred_num_layers=args.pred_layers, dropout=args.dropout
        )
        self.latest_alignment = None
        self.steps = 0
        self.epoch = 0
        self.args = args
        self.best_wer = 1000

    def warmup_optimizer_step(self, steps):
        if steps < self.args.warmup:
            lr_scale = min(1., float(steps + 1) / self.args.warmup*1.0)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr_scale * self.args.lr
    
    def forward(self, batch):
        xs, ys, xlen, ylen = batch
        # xs, ys, xlen = xs.cuda(), ys, xlen.cuda()
        self.model.flatten_parameters()
        alignment = self.model(xs, ys, xlen, ylen)
        return alignment

    def training_step(self, batch, batch_nb):
        xs, ys, xlen, ylen = batch
        # xs, ys, xlen = xs.cuda(), ys, xlen.cuda()
        if xs.shape[1] != xlen.max():
            xs = xs[:, :xlen.max()]
            ys = ys[:, :ylen.max()]
        self.model.flatten_parameters()
        alignment = self.model(xs, ys, xlen, ylen)
        if batch_nb % 100 == 0:
            self.latest_alignment = alignment.cpu()
        if alignment.shape[1] != xs.shape[1]:
            reduction_ratio = (xs.shape[1]/alignment.shape[1])
            xlen = torch.round(xlen/reduction_ratio).int()
        loss = self.loss_fn(alignment, ys.int(), xlen, ylen)

        if batch_nb % 100 == 0:
            lr_val = 0
            for param_group in self.optimizer.param_groups:
                lr_val = param_group['lr']
            self.logger.experiment.add_scalar('lr', lr_val, self.steps)

        self.steps += 1

        if self.steps < self.args.warmup:
            self.warmup_optimizer_step(self.steps)
        else:
            self.cosine_schedule.step()

        return {'loss': loss, 'log': {
            'loss': loss.item()
        }}

    def validation_step(self, batch, batch_nb):
        xs, ys, xlen, ylen = batch
        self.model.flatten_parameters()
        y, nll = self.model.greedy_decode(xs, xlen)

        hypothesis = self.tokenizer.decode_plus(y)
        ground_truth = self.tokenizer.decode_plus(ys.cpu().numpy())
        measures = jiwer.compute_measures(ground_truth, hypothesis)

        return {'val_loss': nll.mean().item(), 'wer': measures['wer'], 'ground_truth': ground_truth[0], 'hypothesis': hypothesis[0]}

    def validation_end(self, outputs):
        # OPTIONAL
        self.logger.experiment.add_text('test', 'This is test', 0)

        avg_wer = np.mean([x['wer'] for x in outputs])
        ppl = np.mean([x['val_loss'] for x in outputs])
        self.logger.experiment.add_scalar('val/WER', avg_wer, self.steps)
        self.logger.experiment.add_scalar('val/perplexity', ppl, self.steps)

        hypothesis, ground_truth = '', ''
        for idx in range(min(5, len(outputs))):
            hypothesis += outputs[idx]['hypothesis']+'\n\n'
            ground_truth += outputs[idx]['ground_truth'] + '\n\n'

        self.logger.experiment.add_text('generated', hypothesis, self.steps)
        self.logger.experiment.add_text('grouth_truth', ground_truth, self.steps)
        if self.latest_alignment != None:
            alignment = self.latest_alignment
            idx = random.randint(0, alignment.size(0) - 1)
            alignment = torch.softmax(alignment[idx], dim=-1)
            alignment[:, :, 0] = 0 # ignore blank token
            alignment = alignment.mean(dim=-1)

            self.logger.experiment.add_image(
                    "alignment",
                    plot_alignment_to_numpy(alignment.data.numpy().T),
                    self.steps, dataformats='HWC')
        self.logger.experiment.flush()

        if self.best_wer > avg_wer:
            print('best checkpoint found!')
            checkpoint = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': self.epoch
            }
            if self.args.apex:
                checkpoint['amp'] = amp.state_dict()
            torch.save(checkpoint, os.path.join(self.args.log_path, str(self.epoch)+'amp_checkpoint.pt'))
            self.best_wer = avg_wer


        self.plateau_scheduler.step(avg_wer)
        self.epoch += 1

        return {'val/WER': torch.tensor(avg_wer),
            'wer': torch.tensor(avg_wer),
            'val/perplexity': torch.tensor(ppl) }
    
    def validation_epoch_end(self, outputs):
        avg_wer = np.mean([x['wer'] for x in outputs])
        ppl = np.mean([x['val_loss'] for x in outputs])

        hypothesis, ground_truth = '', ''
        for idx in range(5):
            hypothesis += outputs[idx]['hypothesis']+'\n\n'
            ground_truth += outputs[idx]['ground_truth'] + '\n\n'

        writer.add_text('generated', hypothesis, self.steps)
        writer.add_text('grouth_truth', ground_truth, self.steps)

        if self.latest_alignment != None:
            alignment = self.latest_alignment
            idx = random.randint(0, alignment.size(0) - 1)
            alignment = torch.softmax(alignment[idx], dim=-1)
            alignment[:, :, 0] = 0 # ignore blank token
            alignment = alignment.mean(dim=-1)

            writer.add_image(
                    "alignment",
                    plot_alignment_to_numpy(alignment.data.numpy().T),
                    self.steps, dataformats='HWC')

        self.logger.experiment.add_scalar('val/WER', avg_wer, self.steps)
        self.logger.experiment.add_scalar('val/perplexity', ppl, self.steps)
        self.logger.experiment.flush()

        self.plateau_scheduler.step(avg_wer)

        self.epoch += 1
        return {'val/WER': torch.tensor(avg_wer),
         'val/perplexity': torch.tensor(ppl) }

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9)
        lmbda = lambda epoch: 0.97
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.optimizer, lr_lambda=lmbda)

        self.plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=2, factor=0.9)
        self.cosine_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 250000 , eta_min=1e-8)
        self.warmup_optimizer_step(0)
        return [self.optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):

        args = self.args
        transforms_piplines = [
            # torchaudio.transforms.MelSpectrogram(
            #     # n_mfcc=args.audio_feat, 
            #     n_fft=args.n_fft, n_mels=args.audio_feat,
            #     # melkwargs={'n_fft':1024, 'win_length': 1024}
            # ),
            # MFCC_(
            #     n_mfcc=args.audio_feat, log_mels=True,
            #     melkwargs={'n_fft':args.n_fft, 'f_max': 5800, 'f_min': 20}
            # ),
            AudioPreprocessing(
                normalize='none', sample_rate=16000, window_size=0.02, 
                window_stride=0.015, features=args.audio_feat, n_fft=512, log=True,
                feat_type='logfbank', trim_silence=True, window='hann',dither=0.00001, frame_splicing=1, transpose_out=False
            ),
            TimeWrap(),
            TimeMask(T=40, num_masks=5, replace_with_zero=False),
            FreqMask(F=5, num_masks=5, replace_with_zero=False),
        ]
        if args.concat:
            transforms_piplines.append(
                 ConcatFeature(merge_size=3)
            )
        transforms = torch.nn.Sequential(*transforms_piplines)

        common_voice = CommonVoice(f'{args.data_path}common_voice',
            audio_max_length=13,
            transforms=transforms, tokenizer=self.tokenizer)
        synthetic = Synthetic(f'{args.data_path}synthetic',
            audio_max_length=13,
            transforms=transforms, tokenizer=self.tokenizer)
        yt3_dataset = YoutubeCaption(f'{args.data_path}youtube-speech-text/',
            labels='news_meta.csv',
            audio_max_length=13,
            transforms=transforms, tokenizer=self.tokenizer)
        yt_dataset = YoutubeCaption(f'{args.data_path}youtube-speech-text/',
            labels='bloomberg2_meta.csv',
            audio_max_length=13,
            transforms=transforms, tokenizer=self.tokenizer)
        yt2_dataset = YoutubeCaption(f'{args.data_path}youtube-speech-text/',
            labels='english2_meta.csv',
            audio_max_length=13,
            transforms=transforms, tokenizer=self.tokenizer)
        yt3_dataset = YoutubeCaption(f'{args.data_path}youtube-speech-text/',
            labels='life_meta.csv',
            audio_max_length=13,
            transforms=transforms, tokenizer=self.tokenizer)
        librispeech2 = Librispeech(f'{args.data_path}LibriSpeech/train-other-500/',
            audio_max_length=13,
            transforms=transforms, tokenizer=self.tokenizer)
        librispeech = Librispeech(f'{args.data_path}LibriSpeech/train-clean-360/',
            audio_max_length=13,
            transforms=transforms, tokenizer=self.tokenizer)
        tedlium = TEDLIUM(f'{args.data_path}TEDLIUM/TEDLIUM_release1/train/',
            audio_max_length=13,
            transforms=transforms, tokenizer=self.tokenizer)
        # tedlium2 = TEDLIUM(f'{args.data_path}TEDLIUM/TEDLIUM_release-3/data/',
        #     audio_max_length=12,
        #     transforms=transforms, tokenizer=self.tokenizer)
        dataset = MergedDataset([common_voice, yt_dataset, librispeech, yt3_dataset, tedlium, yt3_dataset, synthetic, librispeech2, yt2_dataset])        
        return DataLoader(dataset, collate_fn=seq_collate, batch_size=args.batch_size, 
            num_workers=4, shuffle=True, drop_last=True)

    @pl.data_loader
    def val_dataloader(self):

        args = self.args
        val_pipeline = [
            # torchaudio.transforms.MelSpectrogram(
            #     # n_mfcc=args.audio_feat, 
            #     n_fft=args.n_fft, n_mels=args.audio_feat,
            #     # melkwargs={'n_fft':1024, 'win_length': 1024}
            # ),
            # MFCC_(
            #     n_mfcc=args.audio_feat, log_mels=True,
            #     melkwargs={'n_fft':args.n_fft, 'f_max': 5800, 'f_min': 20}
            # )
            AudioPreprocessing(
                normalize='none', sample_rate=16000, window_size=0.02, 
                window_stride=0.015, features=args.audio_feat, n_fft=512, log=True,
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
        _tokenizer = self.tokenizer
        val_dataset = Librispeech(f'{args.data_path}LibriSpeech/test-clean/',
            audio_max_length=14,
            transforms=val_transform, tokenizer=self.tokenizer)
        return DataLoader(val_dataset, collate_fn=seq_collate, batch_size=64, num_workers=4, shuffle=False)



if __name__ == "__main__":
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    import pickle
    model = ParallelTraining(args)
    # with open('test.pt', 'wb') as f:
    #     pickle.dump(model, f)
    params = {
        'gpus': [0, 1, 2],
        'distributed_backend': 'ddp',
        'gradient_clip_val': 10,
        'accumulate_grad_batches': args.accumulation_steps
    }
    if args.apex:
        print('use apex')
        params['amp_level'] = args.opt_level
        params['precision'] = 16

    from datetime import datetime
    cur_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_name = '{}-{}'.format(args.name, args.tokenizer)
    log_path = 'logs/{}'.format(log_name)
    os.makedirs(log_path, exist_ok=True)
    with open('logs/{}/vars.json'.format(log_name), 'w') as f:
        json.dump(vars(args), f)
    if args.tokenizer == 'bpe':
        model.tokenizer.token.save(f'logs/{log_name}/BPE')
    else:
        with open('logs/{}/vocab.json'.format(log_name), 'w') as f:
            json.dump(model.tokenizer.token2id, f)
    model.args.log_path = log_path
    logger = pl.loggers.tensorboard.TensorBoardLogger('logs', name=args.name)
    params['logger'] = logger

    checkpoint_callback = ModelCheckpoint(
        filepath=log_path,
        save_top_k=True,
        verbose=True,
        monitor='val/perplexity',
        mode='min',
        prefix=''
    )
    params['checkpoint_callback'] = checkpoint_callback
    print(params)
    trainer = Trainer(**params)
    trainer.fit(model)
