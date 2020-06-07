import os
from datetime import datetime

import jiwer
import torch
import torchaudio.transforms as transforms
import numpy as np
import torch.optim as optim
from absl import app, flags
from apex import amp
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import trange, tqdm
from warprnnt_pytorch import RNNTLoss

from dataset import (
    seq_collate, MergedDataset, Librispeech, TEDLIUM, CommonVoice)
from models import Transducer
from transforms import CatDeltas, CMVN, Downsample, Transpose
from tokenizer import NUL, HuggingFaceTokenizer, CharTokenizer


FLAGS = flags.FLAGS
flags.DEFINE_string('name', 'rnn-t-v3', help='session name')
flags.DEFINE_string('eval_model', None, help='evaluate and exit')
flags.DEFINE_string('resume_from', None, help='resume from checkpoint')
# dataset
flags.DEFINE_string('LibriSpeech_train', "./data/LibriSpeech/train-clean-360",
                    help='LibriSpeech train')
flags.DEFINE_string('LibriSpeech_test', "./data/LibriSpeech/test-clean",
                    help='LibriSpeech test')
flags.DEFINE_string('TEDLIUM_train', "./data/TEDLIUM_release-3/data",
                    help='TEDLIUM 3 train')
flags.DEFINE_string('TEDLIUM_test', "./data/TEDLIUM_release1/test",
                    help='TEDLIUM 1 test')
flags.DEFINE_string('CommonVoice', "./data/common_voice",
                    help='common voice')
# learning
flags.DEFINE_enum('optim', "adam", ['adam', 'sgd'], help='optimizer')
flags.DEFINE_float('lr', 1e-4, help='initial lr')
flags.DEFINE_bool('sched', True, help='lr reduce rate on plateau')
flags.DEFINE_integer('sched_patience', 1, help='lr reduce rate on plateau')
flags.DEFINE_float('sched_factor', 0.5, help='lr reduce rate on plateau')
flags.DEFINE_float('sched_min_lr', 1e-6, help='lr reduce rate on plateau')
flags.DEFINE_integer('warmup_step', 10000, help='linearly warmup lr')
flags.DEFINE_integer('epochs', 30, help='epoch')
flags.DEFINE_integer('batch_size', 8, help='batch size')
flags.DEFINE_integer('sub_batch_size', 8, help='accumulate batch size')
flags.DEFINE_integer('eval_batch_size', 4, help='evaluation batch size')
flags.DEFINE_float('gradclip', None, help='clip norm value')
# encoder
flags.DEFINE_integer('enc_hidden_size', 600, help='encoder hidden dimension')
flags.DEFINE_integer('enc_layers', 4, help='encoder layers')
flags.DEFINE_float('enc_dropout', 0, help='encoder dropout')
flags.DEFINE_integer('enc_proj_size', 600, help='encoder layers')
# decoder
flags.DEFINE_integer('dec_hidden_size', 150, help='decoder hidden dimension')
flags.DEFINE_integer('dec_layers', 2, help='decoder layers')
flags.DEFINE_float('dec_dropout', 0., help='decoder dropout')
flags.DEFINE_integer('dec_proj_size', 150, help='encoder layers')
# joint
flags.DEFINE_integer('proj_size', 320, help='RNN hidden dimension')
flags.DEFINE_integer('joint_size', 512, help='RNN hidden dimension')
# data preprocess
flags.DEFINE_integer('audio_max_length', 14, help='max length in seconds')
flags.DEFINE_enum('feature', 'mfcc', ['mfcc', 'melspec'], help='audio feature')
flags.DEFINE_integer('feature_size', 80, help='mel_bins')
flags.DEFINE_integer('n_fft', 400, help='spectrogram')
flags.DEFINE_integer('win_length', 400, help='spectrogram')
flags.DEFINE_integer('hop_length', 200, help='spectrogram')
flags.DEFINE_bool('delta', False, help='concat delta and detal of dealt')
flags.DEFINE_bool('cmvn', False, help='normalize spectrogram')
flags.DEFINE_integer('downsample', 3, help='downsample audio feature')
flags.DEFINE_enum('tokenizer', 'char', ['char', 'bpe'], help='tokenizer')
flags.DEFINE_integer('bpe_size', 256, help='BPE vocabulary size')
flags.DEFINE_integer('vocab_embed_size', 16, help='vocabulary embedding size')
# apex
flags.DEFINE_bool('apex', default=True, help='fp16 training')
flags.DEFINE_string('opt_level', 'O1', help='use mix precision')
# parallel
flags.DEFINE_bool('multi_gpu', False, help='DataParallel')
# log
flags.DEFINE_integer('loss_step', 5, help='frequency to show loss in pbar')
flags.DEFINE_integer('save_step', 10000, help='frequency to save model')
flags.DEFINE_integer('eval_step', 10000, help='frequency to save model')
flags.DEFINE_integer('sample_size', 20, help='size of visualized examples')
device = torch.device('cuda:0')


def infloop(dataloader):
    epoch = 1
    while True:
        for batch in dataloader:
            yield batch, epoch
        epoch += 1


def get_transform():
    # Transform
    feature_args = {
        'n_fft': FLAGS.n_fft,
        'win_length': FLAGS.win_length,
        'hop_length': FLAGS.hop_length,
        'f_min': 20,
        'f_max': 5800,
    }
    if FLAGS.feature == 'mfcc':
        transform = [
            transforms.MFCC(
                log_mels=True,
                n_mfcc=FLAGS.feature_size,
                melkwargs=feature_args),
            Transpose()]
    elif FLAGS.feature == 'melspec':
        transform = [
            transforms.MelSpectrogram(
                n_mels=FLAGS.feature_size, **feature_args),
            Transpose()]
    input_size = FLAGS.feature_size
    if FLAGS.delta:
        transform.append(CatDeltas())
        input_size = input_size * 3
    if FLAGS.cmvn:
        transform.append(CMVN())
    if FLAGS.downsample > 1:
        transform.append(Downsample(FLAGS.downsample))
        input_size = input_size * FLAGS.downsample
    transform = torch.nn.Sequential(*transform)
    return transform, input_size


def get_lambda_lr(warmup_step):
    def get_lr(step):
        return min((step + 1) / warmup_step, 1.)
    return get_lr


class Trainer:
    def __init__(self):
        self.name = FLAGS.name
        current = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.logdir = os.path.join('logs', '%s-%s' % (FLAGS.name, current))
        self.model_dir = os.path.join(self.logdir, 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        self.writer = SummaryWriter(self.logdir)
        self.writer.add_text(
            'flagfile', FLAGS.flags_into_string().replace('\n', '\n\n'))
        FLAGS.append_flags_into_file(os.path.join(self.logdir, 'flagfile.txt'))

        # Transform
        transform, input_size = get_transform()

        # Tokenizer
        if FLAGS.tokenizer == 'char':
            self.tokenizer = CharTokenizer(cache_dir=self.logdir)
        else:
            self.tokenizer = HuggingFaceTokenizer(
                cache_dir=self.logdir, vocab_size=FLAGS.bpe_size)

        # Dataloader
        self.dataloader_train = DataLoader(
            dataset=MergedDataset([
                Librispeech(
                    root=FLAGS.LibriSpeech_train,
                    tokenizer=self.tokenizer,
                    transforms=transform,
                    audio_max_length=FLAGS.audio_max_length),
                # TEDLIUM(
                #     root=FLAGS.TEDLIUM_train1,
                #     tokenizer=self.tokenizer,
                #     transforms=transform,
                #     audio_max_length=FLAGS.audio_max_length),
                # CommonVoice(
                #     root=FLAGS.CommonVoice, labels='train.tsv',
                #     tokenizer=self.tokenizer,
                #     transforms=transform,
                #     audio_max_length=FLAGS.audio_max_length)
            ]),
            batch_size=FLAGS.batch_size, shuffle=True, num_workers=4,
            collate_fn=seq_collate, drop_last=True)

        self.dataloader_val = DataLoader(
            dataset=MergedDataset([
                Librispeech(
                    root=FLAGS.LibriSpeech_test,
                    tokenizer=self.tokenizer,
                    transforms=transform,
                    reverse_sorted_by_length=True)]),
            batch_size=FLAGS.eval_batch_size, shuffle=False, num_workers=4,
            collate_fn=seq_collate)

        self.tokenizer.build(self.dataloader_train.dataset.texts())
        self.vocab_size = self.dataloader_train.dataset.tokenizer.vocab_size

        # Model
        self.model = Transducer(
            vocab_embed_size=FLAGS.vocab_embed_size,
            vocab_size=self.vocab_size,
            input_size=input_size,
            enc_hidden_size=FLAGS.enc_hidden_size,
            enc_layers=FLAGS.enc_layers,
            enc_dropout=FLAGS.enc_dropout,
            enc_proj_size=FLAGS.enc_proj_size,
            dec_hidden_size=FLAGS.dec_hidden_size,
            dec_layers=FLAGS.dec_layers,
            dec_dropout=FLAGS.dec_dropout,
            dec_proj_size=FLAGS.dec_proj_size,
            joint_size=FLAGS.joint_size,
        ).to(device)

        # Optimizer
        if FLAGS.optim == 'adam':
            self.optim = optim.Adam(
                self.model.parameters(), lr=FLAGS.lr)
        else:
            self.optim = optim.SGD(
                self.model.parameters(), lr=FLAGS.lr, momentum=0.9)
        # Scheduler
        if FLAGS.sched:
            self.sched = optim.lr_scheduler.ReduceLROnPlateau(
                self.optim, patience=FLAGS.sched_patience,
                factor=FLAGS.sched_factor, min_lr=FLAGS.sched_min_lr,
                verbose=1)
        # Loss
        self.loss_fn = RNNTLoss(blank=NUL)
        # Apex
        if FLAGS.apex:
            self.model, self.optim = amp.initialize(
                self.model, self.optim, opt_level=FLAGS.opt_level)
        # Multi GPU
        if FLAGS.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)

    def scale_length(self, prob, xlen):
        scale = (xlen.max().float() / prob.shape[1]).ceil()
        xlen = (xlen / scale).ceil().int()
        return xlen

    def train(self):
        looper = infloop(self.dataloader_train)
        losses = []
        steps = len(self.dataloader_train) * FLAGS.epochs
        with trange(1, steps + 1, dynamic_ncols=True) as pbar:
            for step in pbar:
                if step <= FLAGS.warmup_step:
                    scale = step / FLAGS.warmup_step
                    self.optim.param_groups[0]['lr'] = FLAGS.lr * scale
                batch, epoch = next(looper)
                loss = self.train_step(batch)
                losses.append(loss)
                lr = self.optim.param_groups[0]['lr']
                pbar.set_description(
                    'Epoch %d, loss: %.4f, lr: %.3E' % (epoch, loss, lr))

                if step % FLAGS.loss_step == 0:
                    train_loss = torch.stack(losses).mean()
                    losses = []
                    self.writer.add_scalar('train_loss', train_loss, step)

                if step % FLAGS.save_step == 0:
                    self.save(step)

                if step % FLAGS.eval_step == 0:
                    pbar.set_description('Evaluating ...')
                    val_loss, wer, pred_seqs, true_seqs = self.evaluate()
                    if FLAGS.sched:
                        self.sched.step(val_loss)
                    self.writer.add_scalar('WER', wer, step)
                    self.writer.add_scalar('val_loss', val_loss, step)
                    for i in range(FLAGS.sample_size):
                        log = "`%s`\n\n`%s`" % (true_seqs[i], pred_seqs[i])
                        self.writer.add_text('val/%d' % i, log, step)
                    pbar.write(
                        'Epoch %d, step %d, loss: %.4f, WER: %.4f' % (
                            epoch, step, val_loss, wer))

    def train_step(self, batch):
        batch = [x.to(device) for x in batch]
        sub_losses = []
        start_idxs = range(0, FLAGS.batch_size, FLAGS.sub_batch_size)
        self.optim.zero_grad()
        for sub_batch_idx, start_idx in enumerate(start_idxs):
            sub_slice = slice(start_idx, start_idx + FLAGS.sub_batch_size)
            xs, ys, xlen, ylen = [x[sub_slice] for x in batch]
            xs = xs[:, :xlen.max()].contiguous()
            ys = ys[:, :ylen.max()].contiguous()
            # prob = self.model(xs, ys)
            # xlen = self.scale_length(prob, xlen)
            # loss = self.loss_fn(prob, ys, xlen, ylen) / len(start_idxs)
            loss = self.model(xs, ys, xlen, ylen)
            if FLAGS.multi_gpu:
                loss = loss.mean() / len(start_idxs)
            if FLAGS.apex:
                delay_unscale = sub_batch_idx < len(start_idxs) - 1
                with amp.scale_loss(
                        loss,
                        self.optim,
                        delay_unscale=delay_unscale) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            sub_losses.append(loss.detach())

        if FLAGS.gradclip is not None:
            if FLAGS.apex:
                parameters = amp.master_params(self.optim)
            else:
                parameters = self.model.parameters()
            torch.nn.utils.clip_grad_norm_(parameters, FLAGS.gradclip)
        self.optim.step()

        loss = torch.stack(sub_losses).sum()
        return loss

    def evaluate(self):
        self.model.eval()
        wers = []
        losses = []
        pred_seqs = []
        true_seqs = []
        with torch.no_grad():
            with tqdm(self.dataloader_val, dynamic_ncols=True) as pbar:
                for batch in pbar:
                    loss, wer, pred_seq, true_seq = self.evaluate_step(batch)
                    wers.append(wer)
                    losses.append(loss)
                    sample_nums = FLAGS.sample_size - len(pred_seqs)
                    pred_seqs.extend(pred_seq[:sample_nums])
                    true_seqs.extend(true_seq[:sample_nums])
                    pbar.set_description('wer: %.4f' % wer)
        loss = np.mean(losses)
        wer = np.mean(wers)
        self.model.train()
        return loss, wer, pred_seqs, true_seqs

    def evaluate_step(self, batch):
        xs, ys, xlen, ylen = [x.to(device) for x in batch]
        xs = xs[:, :xlen.max()]
        ys = ys[:, :ylen.max()].contiguous()
        loss = self.model(xs, ys, xlen, ylen)
        if FLAGS.multi_gpu:
            loss = loss.mean()
        if FLAGS.multi_gpu:
            ys_hat, nll = self.model.module.greedy_decode(xs, xlen)
        else:
            ys_hat, nll = self.model.greedy_decode(xs, xlen)
        pred_seq = self.tokenizer.decode_plus(ys_hat)
        true_seq = self.tokenizer.decode_plus(ys.cpu().numpy())
        wer = jiwer.wer(true_seq, pred_seq)
        return loss.item(), wer, pred_seq, true_seq

    def save(self, step):
        checkpoint = {'optim': self.optim.state_dict()}

        if FLAGS.multi_gpu:
            checkpoint = {'model': self.model.module.state_dict()}
        else:
            checkpoint = {'model': self.model.state_dict()}

        if self.sched is not None:
            checkpoint.update({'sched': self.sched.state_dict()})

        if FLAGS.apex:
            checkpoint.update({'amp': amp.state_dict()})

        path = os.path.join(self.model_dir, 'epoch-%d' % step)
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(FLAGS.eval_model)
        self.optim.load_state_dict(checkpoint['optim'])

        if FLAGS.multi_gpu:
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])

        if self.sched is not None:
            self.sched.load_state_dict(checkpoint['sched'])

        if FLAGS.apex:
            amp.load_state_dict(checkpoint['amp'])

    def sanity_check(self):
        self.model.eval()
        batch = next(iter(self.dataloader_val))
        self.evaluate_step(batch)
        self.model.train()


def main(argv):
    trainer = Trainer()

    if FLAGS.eval_model:
        trainer.load(FLAGS.eval_model)
        val_loss, wer, pred_seqs, true_seqs = trainer.evaluate()
        for pred_seq, true_seq in zip(pred_seqs, true_seqs):
            print('True: %s\n\nPred:%s' % (pred_seq, true_seq))
            print('=' * 20)
        print('Evaluate, loss: %.4f, WER: %.4f' % (val_loss, wer))
    else:
        if FLAGS.resume_from:
            trainer.load(FLAGS.resume_from)
        trainer.sanity_check()
        trainer.train()


if __name__ == "__main__":
    app.run(main)
