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

from dataset import seq_collate, MergedDataset, Librispeech
from models import Transducer
from transforms import CatDeltas, CMVN, Downsample, Transpose, KaldiMFCC
from tokenizer import NUL, HuggingFaceTokenizer, CharTokenizer


FLAGS = flags.FLAGS
flags.DEFINE_string('name', 'rnn-t-v2', help='session name')
flags.DEFINE_string('eval_model', None, help='evaluate and exit')
flags.DEFINE_string('resume_from', None, help='evaluate and exit')
# learning
flags.DEFINE_multi_enum('optim', "sgd", ['adam', 'sgd'], help='optimizer')
flags.DEFINE_float('lr', 1e-4, help='initial learning rate')
flags.DEFINE_bool('sched', True, help='reduce lr on plateau')
flags.DEFINE_integer('epochs', 30, help='epoch')
flags.DEFINE_integer('batch_size', 8, help='batch size')
flags.DEFINE_integer('sub_batch_size', 8, help='accumulate batch size')
flags.DEFINE_integer('eval_batch_size', 4, help='evaluation batch size')
flags.DEFINE_float('gradclip', None, help='clip norm value')
# encoder
flags.DEFINE_integer('enc_hidden_size', 320, help='encoder hidden dimension')
flags.DEFINE_integer('enc_layers', 2, help='encoder layers')
flags.DEFINE_float('enc_dropout', 0.3, help='encoder dropout')
# decoder
flags.DEFINE_integer('dec_hidden_size', 512, help='decoder hidden dimension')
flags.DEFINE_integer('dec_layers', 1, help='decoder layers')
flags.DEFINE_float('dec_dropout', 0., help='decoder dropout')
# joint
flags.DEFINE_integer('proj_size', 320, help='RNN hidden dimension')
flags.DEFINE_integer('joint_size', 512, help='RNN hidden dimension')
# data preprocess
flags.DEFINE_integer('audio_max_length', 14, help='max length in seconds')
flags.DEFINE_enum('feature', 'mfcc', ['mfcc', 'melspec', 'kaldi'], help='audio feature')
flags.DEFINE_integer('feature_size', 40, help='mel_bins')
flags.DEFINE_integer('n_fft', 1024, help='spectrogram')
flags.DEFINE_integer('win_length', 1024, help='spectrogram')
flags.DEFINE_integer('hop_length', 512, help='spectrogram')
flags.DEFINE_bool('delta', False, help='concat delta and detal of dealt')
flags.DEFINE_bool('cmvn', False, help='normalize spectrogram')
flags.DEFINE_integer('downsample', 1, help='downsample audio feature')
flags.DEFINE_multi_enum('tokenizer', 'char', ['char', 'bpe'], help='tokenizer')
flags.DEFINE_integer('bpe_size', 256, help='BPE vocabulary size')
# apex
flags.DEFINE_bool('apex', default=True, help='use mix precision')
flags.DEFINE_string('opt_level', 'O1', help='operation level')
# parallel
flags.DEFINE_bool('multi_gpu', False, help='DataParallel')
# log
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


class TransducerTrainer:
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
        if FLAGS.feature == 'mfcc':
            transform = [
                transforms.MFCC(
                    n_mfcc=FLAGS.feature_size,
                    log_mels=True,
                    melkwargs={
                        'n_fft': FLAGS.n_fft,
                        'win_length': FLAGS.win_length,
                        'hop_length': FLAGS.hop_length}),
                Transpose()]
        elif FLAGS.feature == 'melspec':
            transform = [
                transforms.MelSpectrogram(
                    n_mels=FLAGS.feature_size,
                    n_fft=FLAGS.n_fft,
                    win_length=FLAGS.win_length,
                    hop_length=FLAGS.hop_length),
                Transpose()]
        elif FLAGS.feature == 'kaldi':
            transform = [
                KaldiMFCC()]
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
        self.model = Transducer(
            vocab_size=self.vocab_size,
            input_size=input_size,
            enc_hidden_size=FLAGS.enc_hidden_size,
            enc_layers=FLAGS.enc_layers,
            enc_dropout=FLAGS.enc_dropout,
            dec_hidden_size=FLAGS.dec_hidden_size,
            dec_layers=FLAGS.dec_layers,
            dec_dropout=FLAGS.dec_dropout,
            joint_size=FLAGS.joint_size,
            proj_size=FLAGS.proj_size,
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
                self.optim, patience=1, factor=0.5, min_lr=1e-6, verbose=1)
        else:
            self.sched = None
        # Loss
        self.loss_fn = RNNTLoss(blank=NUL)

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

    def scale_length(self, xlen):
        # xlen = (xlen.float() / 3).ceil()
        return xlen.int()

    def train_step(self, batch):
        batch = [x.to(device) for x in batch]
        sub_losses = []
        start_idxs = range(0, FLAGS.batch_size, FLAGS.sub_batch_size)
        self.optim.zero_grad()
        for sub_batch_idx, start_idx in enumerate(start_idxs):
            sub_slice = slice(start_idx, start_idx + FLAGS.sub_batch_size)
            xs, ys, xlen, ylen = [x[sub_slice] for x in batch]
            xs = xs[:, :xlen.max()]
            ys = ys[:, :ylen.max()].contiguous()
            prob = self.model(xs, ys)
            xlen = self.scale_length(xlen)
            loss = self.loss_fn(prob, ys, xlen, ylen) / len(start_idxs)
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
        wer = []
        losses = []
        pred_seqs = []
        true_seqs = []
        with torch.no_grad():
            with tqdm(self.dataloader_val, dynamic_ncols=True) as pbar:
                for batch in pbar:
                    xs, ys, xlen, ylen = [x.to(device) for x in batch]
                    xs = xs[:, :xlen.max()]
                    ys = ys[:, :ylen.max()].contiguous()
                    prob = self.model(xs, ys)
                    xlen = self.scale_length(xlen)
                    loss = self.loss_fn(prob, ys, xlen, ylen)
                    losses.append(loss.item())

                    if FLAGS.multi_gpu:
                        ys_hat, nll = self.model.module.greedy_decode(xs, xlen)
                    else:
                        ys_hat, nll = self.model.greedy_decode(xs, xlen)
                    pred_seq = self.tokenizer.decode_plus(ys_hat)
                    true_seq = self.tokenizer.decode_plus(ys.cpu().numpy())
                    wer.append(jiwer.wer(true_seq, pred_seq))
                    pbar.set_description('wer: %.4f' % wer[-1])
                    sample_nums = FLAGS.sample_size - len(pred_seqs)
                    pred_seqs.extend(pred_seq[:sample_nums])
                    true_seqs.extend(true_seq[:sample_nums])
        loss = np.mean(losses)
        wer = np.mean(wer)
        self.model.train()
        return loss, wer, pred_seqs, true_seqs

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
        with torch.no_grad():
            batch = next(iter(self.dataloader_val))
            xs, ys, xlen, ylen = [x.to(device) for x in batch]
            xlen = self.scale_length(xlen)
            prob = self.model(xs, ys)
            self.loss_fn(prob, ys, xlen, ylen)
            print('Max xs, ys in validation:', xs.shape, ys.shape)
        self.model.train()


class DecoderTrainer:
    def __init__(self, rnn_trainer):
        self.rnn_trainer = rnn_trainer
        self.proj_vocab = torch.nn.Sequential(
            torch.nn.Linear(FLAGS.proj_size,
                            self.rnn_trainer.tokenizer.vocab_size),
            torch.nn.LogSoftmax(dim=-1)).to(device)
        self.optim = optim.Adam(
            (list(self.proj_vocab.parameters()) +
             list(self.rnn_trainer.model.predictor.parameters()) +
             list(self.rnn_trainer.model.embed.parameters())), lr=1e-3)
        self.loss_fn = torch.nn.NLLLoss(reduction='none')

    def train(self, steps):
        with trange(steps, dynamic_ncols=True, desc="DecoderTrainer") as pbar:
            looper = infloop(self.rnn_trainer.dataloader_train)
            for step in pbar:
                batch, _ = next(looper)
                _, ys, _, ylen = [x.to(device) for x in batch]
                ys = ys
                ylen = (ylen).long()
                mask = torch.arange(ylen.max()).unsqueeze(0).expand(
                    ys.size(0), -1).to(ys.device)
                mask = mask < ylen.unsqueeze(1)
                logits, _ = self.rnn_trainer.model.predict(ys[:, :-1].long())
                logits = self.proj_vocab(logits).transpose(1, 2)
                loss = self.loss_fn(logits, ys[:, 1:])
                loss = loss.sum() / mask.float().sum()
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                pbar.set_description('loss: %.4f' % loss)


def main(argv):
    rnn_trainer = TransducerTrainer()

    if FLAGS.eval_model:
        rnn_trainer.load(FLAGS.eval_model)
        val_loss, wer, pred_seqs, true_seqs = rnn_trainer.evaluate()
        for pred_seq, true_seq in zip(pred_seqs, true_seqs):
            print('True: %s\n\nPred:%s' % (pred_seq, true_seq))
            print('=' * 20)
        print('Evaluate, loss: %.4f, WER: %.4f' % (val_loss, wer))
    else:
        # dec_trainer = DecoderTrainer(rnn_trainer)
        # dec_trainer.train(10000)

        if FLAGS.resume_from:
            rnn_trainer.load(FLAGS.resume_from)
        rnn_trainer.sanity_check()
        rnn_trainer.train()


if __name__ == "__main__":
    app.run(main)
