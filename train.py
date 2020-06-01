import argparse
import os
import textwrap
import json
from datetime import datetime

import jiwer
import torch
import torchaudio.transforms as transforms
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import trange, tqdm
from torch.utils.data import DataLoader
from warprnnt_pytorch import RNNTLoss

import transforms as mtransforms
from models import Transducer
from tokenizer import NUL, HuggingFaceTokenizer, CharTokenizer
from dataset import (
    seq_collate,
    MergedDataset,
    Librispeech,
    # CommonVoice,
    # TEDLIUM,
    # YoutubeCaption,
)


parser = argparse.ArgumentParser(description='RNN-T')
parser.add_argument('--name', type=str, default='rnn-t')
parser.add_argument('--eval_model', type=str, default=None,
                    help='only evaluate model')
# learning
parser.add_argument('--optim', default="adam", choices=['adam', 'sgd'],
                    help='initial learning rate')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='initial learning rate')
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
parser.add_argument('--enc_num_layers', type=int, default=4,
                    help='number rnn layers')
parser.add_argument('--enc_dropout', type=float, default=0.,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dec_num_layers', type=int, default=2,
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
parser.add_argument('--example_size', type=int, default=20,
                    help='size of visualized examples')
device = torch.device('cuda:0')

args = parser.parse_args()
if args.apex:
    from apex import amp


log_pattern = textwrap.dedent(
    '''
    `True: %s`

    `Pred: %s`
    ''')


def infloop(dataloader):
    epoch = 1
    while True:
        for batch in dataloader:
            yield batch, epoch
        epoch += 1


def save(model, optim, sched, epoch):
    if not os.path.exists(os.path.join(args.logdir, 'models')):
        os.mkdir(os.path.join(args.logdir, 'models'))
    if args.multi_gpu:
        ckpt = {'model': model.module.state_dict()}
    else:
        ckpt = {'model': model.state_dict()}
    ckpt.update({'optim': optim})
    ckpt.update({'sched': sched})
    torch.save(
        ckpt, os.path.join(args.logdir, 'models', 'epoch-%d' % epoch))


def evaluate(model, dataloader, loss_fn, example_size):
    model.eval()
    tokenizer = dataloader.dataset.tokenizer
    losses = []
    pred_seqs = []
    true_seqs = []
    wer = []
    with torch.no_grad(), tqdm(dataloader, dynamic_ncols=True) as pbar:
        for batch in pbar:
            xs, ys, xlen, ylen = [x.to(device) for x in batch]

            prob = model(xs, ys)
            loss = loss_fn(prob, ys, xlen, ylen)
            losses.append(loss.item())

            xs = xs.to(device)
            if args.multi_gpu:
                ys_hat, nll = model.module.greedy_decode(xs, xlen)
            else:
                ys_hat, nll = model.greedy_decode(xs, xlen)
            pred_seq = tokenizer.decode_plus(ys_hat)
            true_seq = tokenizer.decode_plus(ys.cpu().numpy())
            wer.append(jiwer.wer(true_seq, pred_seq))
            pbar.set_description('wer: %.4f' % wer[-1])
            if len(pred_seqs) < example_size:
                pred_seqs.extend(pred_seq[:example_size - len(pred_seqs)])
                true_seqs.extend(true_seq[:example_size - len(true_seqs)])
    loss = np.mean(losses)
    wer = np.mean(wer)
    model.train()
    return loss, wer, pred_seqs, true_seqs


def main():
    current = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    args.logdir = os.path.join('logs', '%s-%s' % (args.name, current))
    writer = SummaryWriter(args.logdir)
    writer.add_text('args', json.dumps(vars(args), indent=4))
    print(json.dumps(vars(args)))

    transform = torch.nn.Sequential(
        transforms.MFCC(
            n_mfcc=args.audio_feat_size,
            melkwargs={
                'n_fft': args.n_fft,
                'win_length': args.win_length,
                'hop_length': args.hop_length}),
        # mtransforms.Transpose(),
        # mtransforms.CatDeltas(),
        # mtransforms.CMVN(),
        # mtransforms.Downsample(args.sample_frame),
    )

    # transform = torch.nn.Sequential(
    #     mtransforms.KaldiMFCC(num_ceps=args.audio_feat_size),
    #     mtransforms.CatDeltas(),
    #     mtransforms.CMVN(),
    #     mtransforms.Downsample(args.sample_frame)
    # )

    if args.tokenizer == 'bpe':
        tokenizer = HuggingFaceTokenizer(
            cache_dir=args.logdir, vocab_size=args.bpe_size)
    else:
        tokenizer = CharTokenizer(cache_dir=args.logdir)

    train_dataloader = DataLoader(
        dataset=MergedDataset([
            Librispeech(
                '../LibriSpeech/train-clean-360/',
                tokenizer=tokenizer,
                transforms=transform,
                audio_max_length=args.audio_max_length)]),
        batch_size=args.batch_size, shuffle=True, num_workers=4,
        collate_fn=seq_collate, drop_last=True)

    val_dataloader = DataLoader(
        dataset=MergedDataset([
            Librispeech(
                '../LibriSpeech/test-clean/',
                tokenizer=tokenizer,
                transforms=transform)]),
        batch_size=args.eval_batch_size, shuffle=False, num_workers=4,
        collate_fn=seq_collate)

    tokenizer.build(train_dataloader.dataset.texts())

    model = Transducer(
        vocab_size=train_dataloader.dataset.tokenizer.vocab_size,
        vocab_embed_size=args.vocab_embed_size,
        audio_feat_size=args.audio_feat_size * args.sample_frame,
        hidden_size=args.hidden_size,
        enc_num_layers=args.enc_num_layers,
        enc_dropout=args.enc_dropout,
        dec_num_layers=args.dec_num_layers,
        dec_dropout=args.dec_dropout,
        proj_size=args.hidden_size,
    ).to(device)
    if args.optim == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    sched = None
    loss_fn = RNNTLoss(blank=NUL)
    if args.apex:
        model, optim = amp.initialize(model, optim, opt_level=args.opt_level)
    if args.multi_gpu:
        model = torch.nn.DataParallel(model)
    if args.eval_model:
        model.load_state_dict(torch.load(args.eval_model)['model'])
        val_loss, wer, _, _ = evaluate(
            model, val_dataloader, loss_fn, args.example_size)
        print('Evaluate, loss: %.4f, WER: %.4f' % (val_loss, wer))
        exit(0)
    losses = []
    looper = infloop(train_dataloader)
    total_steps = len(train_dataloader) * args.epochs
    with trange(total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            batch, epoch = next(looper)
            batch = [x.to(device) for x in batch]

            start_idxs = range(0, args.batch_size, args.sub_batch_size)
            sub_losses = []
            optim.zero_grad()
            for sub_batch_idx, start_idx in enumerate(start_idxs):
                sub_slice = slice(start_idx, start_idx + args.sub_batch_size)
                xs, ys, xlen, ylen = [x[sub_slice] for x in batch]
                xs = xs[:, :xlen.max()]
                ys = ys[:, :ylen.max()].contiguous()
                prob = model(xs, ys)
                loss = loss_fn(prob, ys, xlen, ylen) / len(start_idxs)
                if args.apex:
                    delay_unscale = sub_batch_idx < len(start_idxs) - 1
                    with amp.scale_loss(
                            loss,
                            optim,
                            delay_unscale=delay_unscale) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                sub_losses.append(loss.detach())

            if args.gradclip:
                if args.apex:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optim), args.gradclip)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.gradclip)
            optim.step()

            loss = torch.stack(sub_losses).sum()
            losses.append(loss)
            pbar.set_description('Epoch %d, loss: %.4f' % (epoch, loss))

            if step > 0 and step % 5 == 0:
                train_loss = torch.stack(losses).mean()
                writer.add_scalar('train_loss', train_loss, step)
                losses = []

            if step > 0 and step % args.save_step == 0:
                save(model, optim, sched, step)

            if step > 0 and step % args.eval_step == 0:
                pbar.set_description('Evaluating ...')
                val_loss, wer, pred_seqs, true_seqs = evaluate(
                    model, val_dataloader, loss_fn, args.example_size)
                writer.add_scalar('WER', wer, step)
                writer.add_scalar('val_loss', val_loss, step)
                for i in range(args.example_size):
                    log = log_pattern % (true_seqs[i], pred_seqs[i])
                    writer.add_text('val/%d' % i, log, step)
                pbar.write(
                    'Epoch %d, step %d, loss: %.4f, WER: %.4f' % (
                        epoch, step, val_loss, wer))


if __name__ == "__main__":
    main()
