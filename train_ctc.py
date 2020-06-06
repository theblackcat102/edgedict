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
from warpctc_pytorch import CTCLoss

from dataset import seq_collate, MergedDataset, Librispeech
from models import CTCEncoder
from transforms import CatDeltas, CMVN, Downsample, Transpose, KaldiMFCC
from tokenizer import NUL, HuggingFaceTokenizer, CharTokenizer


FLAGS = flags.FLAGS
flags.DEFINE_string('name', 'ctc-v1', help='session name')
flags.DEFINE_string('eval_model', None, help='evaluate and exit')
flags.DEFINE_string('resume_from', None, help='evaluate and exit')
# learning
flags.DEFINE_multi_enum('optim', "adam", ['adam', 'sgd'], help='optimizer')
flags.DEFINE_float('lr', 5e-4, help='initial learning rate')
flags.DEFINE_bool('sched', True, help='reduce lr on plateau')
flags.DEFINE_integer('epochs', 30, help='epoch')
flags.DEFINE_integer('batch_size', 32, help='batch size')
flags.DEFINE_integer('sub_batch_size', 32, help='accumulate batch size')
flags.DEFINE_integer('eval_batch_size', 64, help='evaluation batch size')
flags.DEFINE_float('gradclip', None, help='clip norm value')
# encoder
flags.DEFINE_integer('enc_hidden_size', 320, help='encoder hidden dimension')
flags.DEFINE_integer('enc_layers', 4, help='encoder layers')
flags.DEFINE_float('enc_dropout', 0.3, help='encoder dropout')
flags.DEFINE_integer('proj_size', 320, help='RNN hidden dimension')
# data preprocess
flags.DEFINE_integer('audio_max_length', 14, help='max length in seconds')
flags.DEFINE_enum('feature', 'mfcc', ['mfcc', 'melspec', 'kaldi'],
                  help='audio feature')
flags.DEFINE_integer('feature_size', 40, help='mel_bins')
flags.DEFINE_integer('n_fft', 512, help='spectrogram')
flags.DEFINE_integer('win_length', 400, help='spectrogram')
flags.DEFINE_integer('hop_length', 160, help='spectrogram')
flags.DEFINE_bool('delta', False, help='concat delta and detal of dealt')
flags.DEFINE_bool('cmvn', False, help='normalize spectrogram')
flags.DEFINE_integer('downsample', 3, help='downsample audio feature')
flags.DEFINE_multi_enum('tokenizer', 'char', ['char', 'bpe'], help='tokenizer')
flags.DEFINE_integer('bpe_size', 256, help='BPE vocabulary size')
# apex
flags.DEFINE_bool('apex', default=True, help='use mix precision')
flags.DEFINE_string('opt_level', 'O1', help='operation level')
# parallel
flags.DEFINE_bool('multi_gpu', False, help='DataParallel')
# log
flags.DEFINE_integer('save_step', 10000, help='frequency to save model')
flags.DEFINE_integer('eval_step', 5000, help='frequency to save model')
flags.DEFINE_integer('sample_size', 20, help='size of visualized examples')
device = torch.device('cuda:0')


def infloop(dataloader):
    epoch = 1
    while True:
        for batch in dataloader:
            yield batch, epoch
        epoch += 1


def build_transform():
    if FLAGS.feature == 'mfcc':
        transform = [
            transforms.MFCC(
                n_mfcc=FLAGS.feature_size,
                log_mels=True,
                melkwargs={
                    'n_fft': FLAGS.n_fft,
                    'win_length': FLAGS.win_length,
                    'hop_length': FLAGS.hop_length}),
            Transpose()
        ]
        input_size = FLAGS.feature_size
    elif FLAGS.feature == 'melspec':
        transform = [
            transforms.MelSpectrogram(
                n_mels=FLAGS.feature_size,
                n_fft=FLAGS.n_fft,
                win_length=FLAGS.win_length,
                hop_length=FLAGS.hop_length),
            Transpose()
        ]
        input_size = FLAGS.feature_size
    elif FLAGS.feature == 'kaldi':
        transform = [KaldiMFCC()]
        input_size = 13
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


class CTCTrainer:
    def __init__(self):
        self.name = FLAGS.name
        current = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.logdir = os.path.join('logs', '%s-%s' % (FLAGS.name, current))
        self.model_dir = os.path.join(self.logdir, 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        self.writer = SummaryWriter(self.logdir)
        self.writer.add_text(
            'flagfile', FLAGS.flags_into_string().replace('\n', '\n\n'))
        FLAGS.append_flags_into_file(
            os.path.join(self.logdir, 'flagfile.txt'))

        # Transform
        transform, input_size = build_transform()

        # Tokenizer
        if FLAGS.tokenizer == 'bpe':
            self.tokenizer = HuggingFaceTokenizer(
                cache_dir=self.logdir, vocab_size=FLAGS.bpe_size)
        else:
            self.tokenizer = CharTokenizer(cache_dir=self.logdir)

        # Dataloader
        self.dataloader_train = DataLoader(
            dataset=MergedDataset([
                Librispeech(
                    '../LibriSpeech/train-clean-360/',
                    tokenizer=self.tokenizer,
                    transforms=transform,
                    audio_max_length=FLAGS.audio_max_length)]),
            batch_size=FLAGS.batch_size, shuffle=True, num_workers=0,
            collate_fn=seq_collate, drop_last=True)

        self.dataloader_val = DataLoader(
            dataset=MergedDataset([
                Librispeech(
                    '../LibriSpeech/test-clean/',
                    tokenizer=self.tokenizer,
                    transforms=transform,
                    reverse_sorted_by_length=True)]),
            batch_size=FLAGS.eval_batch_size, shuffle=False, num_workers=0,
            collate_fn=seq_collate)

        self.tokenizer.build(self.dataloader_train.dataset.texts())
        self.vocab_size = self.dataloader_train.dataset.tokenizer.vocab_size

        # Model
        self.model = CTCEncoder(
            vocab_size=self.vocab_size,
            input_size=input_size,
            enc_hidden_size=FLAGS.enc_hidden_size,
            enc_layers=FLAGS.enc_layers,
            enc_dropout=FLAGS.enc_dropout,
            proj_size=FLAGS.proj_size).to(device)

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
                self.optim, patience=1, factor=0.5, min_lr=1e-6, verbose=1)
        else:
            self.sched = None
        # Loss
        # self.loss_fn = torch.nn.CTCLoss(blank=NUL, reduction='sum')
        self.loss_fn = CTCLoss(blank=NUL, size_average=True)

        # Apex
        if FLAGS.apex:
            self.model, self.optim = amp.initialize(
                self.model, self.optim, opt_level=FLAGS.opt_level)
        if FLAGS.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)

    def train(self):
        looper = infloop(self.dataloader_train)
        losses = []
        steps = len(self.dataloader_train) * FLAGS.epochs
        with trange(steps, dynamic_ncols=True) as pbar:
            for step in pbar:
                batch, epoch = next(looper)
                loss = self.train_step(batch)
                losses.append(loss)
                lr = self.optim.param_groups[0]['lr']
                pbar.set_description(
                    'Epoch %d, loss: %.4f, lr: %.2E' % (epoch, loss, lr))

                if step > 0 and step % 50 == 0:
                    train_loss = torch.stack(losses).mean()
                    self.writer.add_scalar('train_loss', train_loss, step)
                    losses = []

                if step > 0 and step % FLAGS.save_step == 0:
                    self.save(step)

                if step >= 0 and step % FLAGS.eval_step == 0:
                    pbar.set_description('Evaluating ...')
                    val_loss, wer, pred_seqs, true_seqs = self.evaluate()
                    self.writer.add_scalar('WER', wer, step)
                    self.writer.add_scalar('val_loss', val_loss, step)
                    if FLAGS.sched:
                        self.sched.step(val_loss)
                    for i in range(FLAGS.sample_size):
                        log = "`%s`\n\n`%s`" % (true_seqs[i], pred_seqs[i])
                        self.writer.add_text('val/%d' % i, log, step)
                    pbar.write(
                        'Epoch %d, step %d, loss: %.4f, WER: %.4f' % (
                            epoch, step, val_loss, wer))

    def train_step(self, batch):
        sub_losses = []
        start_idxs = range(0, FLAGS.batch_size, FLAGS.sub_batch_size)
        self.optim.zero_grad()
        for sub_batch_idx, start_idx in enumerate(start_idxs):
            sub_slice = slice(start_idx, start_idx + FLAGS.sub_batch_size)
            xs, ys, xlen, ylen = [x[sub_slice] for x in batch]
            xs = xs.to(device)
            xs = xs[:, :xlen.max()]
            ys = ys[:, :ylen.max()].contiguous()
            logprobs = self.model(xs)
            logprobs = logprobs.transpose(0, 1)
            ys_flatten = torch.cat([y[:leng] for y, leng in zip(ys, ylen)])
            loss = self.loss_fn(logprobs, ys_flatten, xlen, ylen)
            loss = loss / len(xs) / len(start_idxs)
            if FLAGS.apex:
                delay_unscale = sub_batch_idx < len(start_idxs) - 1
                with amp.scale_loss(
                        loss,
                        self.optim,
                        delay_unscale=delay_unscale) as scaled_loss:
                    scaled_loss.backward()
            else:
                pass
                loss.backward()
            sub_losses.append(loss.detach())

        if FLAGS.gradclip is not None:
            if FLAGS.apex:
                parameters = amp.master_params(self.optim)
            else:
                parameters = self.model.parameters()
            torch.nn.utils.clip_grad_norm_(parameters, FLAGS.gradclip)
        self.optim.step()
        # self.sched.step()

        loss = torch.stack(sub_losses).sum()
        return loss

    def evaluate(self):
        self.model.eval()
        wers = []
        losses = []
        pred_seqs = []
        true_seqs = []
        with tqdm(self.dataloader_val, dynamic_ncols=True) as pbar:
            for batch in pbar:
                loss, wer, pred_seq, true_seq = self.evaluate_step(batch)
                losses.append(loss)
                wers.append(wer)
                sample_nums = FLAGS.sample_size - len(pred_seqs)
                pred_seqs.extend(pred_seq[:sample_nums])
                true_seqs.extend(true_seq[:sample_nums])
                pbar.set_description('wer: %.4f' % wer)
        loss = np.mean(losses)
        wer = np.mean(wers)
        self.model.train()
        return loss, wer, pred_seqs, true_seqs

    def evaluate_step(self, batch):
        with torch.no_grad():
            xs, ys, xlen, ylen = batch
            xs = xs.to(device)
            xs = xs[:, :xlen.max()]
            ys = ys[:, :ylen.max()].contiguous()
            logprobs = self.model(xs.to(device))
            ys_flatten = torch.cat([y[:leng] for y, leng in zip(ys, ylen)])
            loss = self.loss_fn(
                logprobs.transpose(0, 1), ys_flatten, xlen, ylen)
            loss = loss / len(xs)

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
        self.evaluate_step(batch=next(iter(self.dataloader_val)))
        self.model.train()


def main(argv):
    rnn_trainer = CTCTrainer()

    if FLAGS.eval_model:
        rnn_trainer.load(FLAGS.eval_model)
        val_loss, wer, pred_seqs, true_seqs = rnn_trainer.evaluate()
        for pred_seq, true_seq in zip(pred_seqs, true_seqs):
            print('True: %s\n\nPred:%s' % (pred_seq, true_seq))
            print('=' * 20)
        print('Evaluate, loss: %.4f, WER: %.4f' % (val_loss, wer))
    else:
        if FLAGS.resume_from:
            rnn_trainer.load(FLAGS.resume_from)
        rnn_trainer.sanity_check()
        rnn_trainer.train()


if __name__ == "__main__":
    app.run(main)
