import os

import jiwer
import torch
import numpy as np
import torch.optim as optim
from absl import app
from apex import amp
from tqdm import trange, tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from rnnt.args import FLAGS
from rnnt.dataset import seq_collate, MergedDataset, Librispeech, CommonVoice, TEDLIUM, YoutubeCaption
from rnnt.models import Transducer
from rnnt.tokenizer import HuggingFaceTokenizer, CharTokenizer
from rnnt.transforms import build_transform


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

from torch.nn.utils.rnn import pack_padded_sequence

def infloop(dataloader):
    epoch = 1
    while True:
        for batch in dataloader:
            yield batch, epoch
        epoch += 1


class Trainer:
    def __init__(self):
        self.name = FLAGS.name
        self.logdir = os.path.join('logs', FLAGS.name)
        self.model_dir = os.path.join(self.logdir, 'models')

        # Transform
        transform_train, transform_test, input_size = build_transform(
            feature_type=FLAGS.feature, feature_size=FLAGS.feature_size,
            n_fft=FLAGS.n_fft, win_length=FLAGS.win_length,
            hop_length=FLAGS.hop_length, delta=FLAGS.delta, cmvn=FLAGS.cmvn,
            downsample=FLAGS.downsample,
            T_mask=FLAGS.T_mask, T_num_mask=FLAGS.T_num_mask,
            F_mask=FLAGS.F_mask, F_num_mask=FLAGS.F_num_mask
        )

        # Tokenizer
        if FLAGS.tokenizer == 'char':
            self.tokenizer = CharTokenizer(cache_dir=self.logdir)
        else:
            self.tokenizer = HuggingFaceTokenizer(
                cache_dir='BPE-2048', vocab_size=FLAGS.bpe_size)

        # Dataloader
        self.dataloader_train = DataLoader(
            dataset=MergedDataset([
                Librispeech(
                    root=FLAGS.LibriSpeech_train_500,
                    tokenizer=self.tokenizer,
                    transform=transform_train,
                    audio_max_length=FLAGS.audio_max_length),
                Librispeech(
                    root=FLAGS.LibriSpeech_train_360,
                    tokenizer=self.tokenizer,
                    transform=transform_train,
                    audio_max_length=FLAGS.audio_max_length),
                # Librispeech(
                #     root=FLAGS.LibriSpeech_train_100,
                #     tokenizer=self.tokenizer,
                #     transform=transform_train,
                #     audio_max_length=FLAGS.audio_max_length),
                TEDLIUM(
                    root=FLAGS.TEDLIUM_train,
                    tokenizer=self.tokenizer,
                    transform=transform_train,
                    audio_max_length=FLAGS.audio_max_length),
                CommonVoice(
                    root=FLAGS.CommonVoice, labels='train.tsv',
                    tokenizer=self.tokenizer,
                    transform=transform_train,
                    audio_max_length=FLAGS.audio_max_length),
                YoutubeCaption(
                    root='../speech_data/youtube-speech-text/', labels='bloomberg2_meta.csv',
                    tokenizer=self.tokenizer,
                    transform=transform_train,
                    audio_max_length=FLAGS.audio_max_length),
                YoutubeCaption(
                    root='../speech_data/youtube-speech-text/', labels='life_meta.csv',
                    tokenizer=self.tokenizer,
                    transform=transform_train,
                    audio_max_length=FLAGS.audio_max_length),                    
                YoutubeCaption(
                    root='../speech_data/youtube-speech-text/', labels='news_meta.csv',
                    tokenizer=self.tokenizer,
                    transform=transform_train,
                    audio_max_length=FLAGS.audio_max_length),
                YoutubeCaption(
                    root='../speech_data/youtube-speech-text/', labels='english2_meta.csv',
                    tokenizer=self.tokenizer,
                    transform=transform_train,
                    audio_max_length=FLAGS.audio_max_length),
            ]),
            batch_size=FLAGS.batch_size, shuffle=True,
            num_workers=FLAGS.num_workers, collate_fn=seq_collate,
            drop_last=True)

        self.dataloader_val = DataLoader(
            dataset=MergedDataset([
                Librispeech(
                    root=FLAGS.LibriSpeech_test,
                    tokenizer=self.tokenizer,
                    transform=transform_test,
                    reverse_sorted_by_length=True)]),
            batch_size=FLAGS.eval_batch_size, shuffle=False,
            num_workers=FLAGS.num_workers, collate_fn=seq_collate)

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

    def train(self, start_step=1):
        if FLAGS.mode == "resume":
            exist_ok = True
        else:
            exist_ok = False
        os.makedirs(self.model_dir, exist_ok=exist_ok)
        writer = SummaryWriter(self.logdir)
        writer.add_text(
            'flagfile', FLAGS.flags_into_string().replace('\n', '\n\n'))
        FLAGS.append_flags_into_file(os.path.join(self.logdir, 'flagfile.txt'))

        looper = infloop(self.dataloader_train)
        losses = []
        steps = len(self.dataloader_train) * FLAGS.epochs
        with trange(start_step, steps + 1, dynamic_ncols=True) as pbar:
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
                    writer.add_scalar('train_loss', train_loss, step)

                if step % FLAGS.save_step == 0:
                    self.save(step)

                if step % FLAGS.eval_step == 0:
                    pbar.set_description('Evaluating ...')
                    val_loss, wer, pred_seqs, true_seqs = self.evaluate()
                    if FLAGS.sched:
                        self.sched.step(val_loss)
                    writer.add_scalar('WER', wer, step)
                    writer.add_scalar('val_loss', val_loss, step)
                    for i in range(FLAGS.sample_size):
                        log = "`%s`\n\n`%s`" % (true_seqs[i], pred_seqs[i])
                        writer.add_text('val/%d' % i, log, step)
                    pbar.write(
                        'Epoch %d, step %d, loss: %.4f, WER: %.4f' % (
                            epoch, step, val_loss, wer))

    def train_step(self, batch):
        sub_losses = []
        start_idxs = range(0, FLAGS.batch_size, FLAGS.sub_batch_size)
        self.optim.zero_grad()
        for sub_batch_idx, start_idx in enumerate(start_idxs):
            sub_slice = slice(start_idx, start_idx + FLAGS.sub_batch_size)
            xs, ys, xlen, ylen = [x[sub_slice].to(device) for x in batch]
            xs = xs[:, :xlen.max()].contiguous()
            ys = ys[:, :ylen.max()].contiguous()
            loss = self.model(xs, ys, xlen, ylen)
            if FLAGS.multi_gpu:
                loss = loss.mean() / len(start_idxs)
            else:
                loss = loss / len(start_idxs)
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
                    pbar.set_description('wer: %.4f, loss: %.4f' % (wer, loss))
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
            checkpoint.update({'model': self.model.module.state_dict()})
        else:
            checkpoint.update({'model': self.model.state_dict()})

        if self.sched is not None:
            checkpoint.update({'sched': self.sched.state_dict()})

        if FLAGS.apex:
            checkpoint.update({'amp': amp.state_dict()})

        path = os.path.join(self.model_dir, '%d.pt' % step)
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path)
        # self.optim.load_state_dict(checkpoint['optim'])

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

    if FLAGS.mode == 'eval':
        path = os.path.join(trainer.model_dir, '%d.pt' % FLAGS.resume_step)
        trainer.load(path)
        val_loss, wer, pred_seqs, true_seqs = trainer.evaluate()
        for pred_seq, true_seq in zip(pred_seqs, true_seqs):
            print('True: %s\n\nPred:%s' % (pred_seq, true_seq))
            print('=' * 20)
        print('Evaluate, loss: %.4f, WER: %.4f' % (val_loss, wer))

    step = 1

    if FLAGS.mode == 'resume':
        step = FLAGS.resume_step
        path = os.path.join(trainer.model_dir, '%d.pt' % step)
        trainer.load(path)

    if FLAGS.mode == 'train' or FLAGS.mode == 'resume':
        # trainer.sanity_check()
        trainer.train(start_step=step)


if __name__ == "__main__":
    app.run(main)
