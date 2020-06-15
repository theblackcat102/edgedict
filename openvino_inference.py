import os
import time

import torch.onnx
import numpy as np
import jiwer
from absl import app, flags
from openvino.inference_engine import IECore
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from rnnt.args import FLAGS                             # define training FLAGS
from rnnt.transforms import build_transform
from rnnt.tokenizer import HuggingFaceTokenizer, BOS, NUL
from rnnt.models import Transducer
from rnnt.dataset import MergedDataset, Librispeech


flags.DEFINE_integer('step', 105000, help='steps of checkpoint')
flags.DEFINE_integer('step_n_frame', 10, help='input frame(stacked)')


def pytorch_fullseq_decode(encoder, decoder, joint, waveform, transform,
                           tokenizer, verbose=0):
    # Encode waveform at a time
    xs = transform(waveform).transpose(1, 2)
    length = xs.shape[1]
    length = length - length % 2
    xs = xs[:, :length]
    enc_h = torch.zeros(FLAGS.enc_layers, 1, FLAGS.enc_hidden_size)
    enc_c = torch.zeros(FLAGS.enc_layers, 1, FLAGS.enc_hidden_size)
    enc_xs, _ = encoder(xs, (enc_h, enc_c))

    dec_x = torch.ones(1, 1).long() * BOS
    dec_h = torch.zeros(FLAGS.dec_layers, 1, FLAGS.dec_hidden_size)
    dec_c = torch.zeros(FLAGS.dec_layers, 1, FLAGS.dec_hidden_size)
    dec_x, (dec_h, dec_c) = decoder(dec_x, (dec_h, dec_c))

    pred_seq = ""
    total_frames = waveform.shape[1]
    for i in range(enc_xs.shape[1]):
        prob = joint(enc_xs[:, i], dec_x[:, 0])
        pred = prob.argmax(dim=-1).item()

        if pred != NUL:
            dec_x = torch.ones(1, 1).long() * pred
            dec_x, (dec_h, dec_c) = decoder(dec_x, (dec_h, dec_c))
            seq = tokenizer.tokenizer.id_to_token(pred)
            seq = seq.replace('</w>', ' ')
            if verbose > 0:
                print(seq, end='', flush=True)
            pred_seq += seq
    return pred_seq, total_frames


def pytorch_framewise_decode(encoder, decoder, joint, waveform, transform,
                             tokenizer, verbose=0):
    win_size = (
        FLAGS.win_length +
        FLAGS.hop_length * (FLAGS.downsample * FLAGS.step_n_frame - 1))
    hop_size = (
        FLAGS.hop_length * (FLAGS.downsample * FLAGS.step_n_frame))

    # Encode waveform frame by frame
    enc_h = torch.zeros(FLAGS.enc_layers, 1, FLAGS.enc_hidden_size)
    enc_c = torch.zeros(FLAGS.enc_layers, 1, FLAGS.enc_hidden_size)

    dec_x = torch.ones(1, 1).long() * BOS
    dec_h = torch.zeros(FLAGS.dec_layers, 1, FLAGS.dec_hidden_size)
    dec_c = torch.zeros(FLAGS.dec_layers, 1, FLAGS.dec_hidden_size)
    dec_x, (dec_h, dec_c) = decoder(dec_x, (dec_h, dec_c))

    pred_seq = ""
    total_frames = FLAGS.win_length
    for start in range(0, waveform.shape[1] - win_size, hop_size):
        total_frames += hop_size
        xs = transform(
            waveform[:, start: start + win_size]).transpose(1, 2)
        # print(xs.shape)
        enc_xs, (enc_h, enc_c) = encoder(xs, (enc_h, enc_c))
        # print(enc_xs.shape)
        for k in range(enc_xs.shape[1]):
            prob = joint(enc_xs[:, k], dec_x[:, 0])
            pred = prob.argmax(dim=-1).item()

            if pred != NUL:
                dec_x = torch.ones(1, 1).long() * pred
                dec_x, (dec_h, dec_c) = decoder(dec_x, (dec_h, dec_c))
                seq = tokenizer.tokenizer.id_to_token(pred)
                seq = seq.replace('</w>', ' ')
                if verbose > 0:
                    print(seq, end='', flush=True)
                pred_seq += seq
    return pred_seq, total_frames


def openvino_framewise_decode(encoder, decoder, joint, waveform, tokenizer,
                              transform, verbose=0):
    win_size = (
        FLAGS.win_length +
        FLAGS.hop_length * (FLAGS.downsample * FLAGS.step_n_frame - 1))
    hop_size = (
        FLAGS.hop_length * (FLAGS.downsample * FLAGS.step_n_frame))

    # Encode waveform frame by frame
    enc_h = np.zeros(
        (FLAGS.enc_layers, 1, FLAGS.enc_hidden_size), dtype=np.float)
    enc_c = np.zeros(
        (FLAGS.enc_layers, 1, FLAGS.enc_hidden_size), dtype=np.float)

    dec_x = np.ones((1, 1), dtype=np.long) * BOS
    dec_h = np.zeros(
        (FLAGS.dec_layers, 1, FLAGS.dec_hidden_size), dtype=np.float)
    dec_c = np.zeros(
        (FLAGS.dec_layers, 1, FLAGS.dec_hidden_size), dtype=np.float)
    outputs = decoder.infer({
        'input': dec_x,
        'input_hidden': dec_h,
        'input_cell': dec_c,
    })
    # print(outputs.keys())
    dec_x = outputs['Add_26']
    dec_h = outputs['Concat_23']
    dec_c = outputs['Concat_24']

    pred_seq = ""
    total_frames = FLAGS.win_length
    for start in range(0, waveform.shape[1] - win_size, hop_size):
        total_frames += hop_size
        xs = transform(
            waveform[:, start: start + win_size]).transpose(1, 2).numpy()
        # enc_xs, (enc_h, enc_c) = encoder(xs, (enc_h, enc_c))
        outputs = encoder.infer(inputs={
            'input': xs,
            'input_hidden': enc_h,
            'input_cell': enc_c,
        })
        # print(outputs.keys())
        enc_xs = outputs['Add_156']
        enc_h = outputs['Concat_153']
        enc_c = outputs['Concat_154']
        for k in range(enc_xs.shape[1]):
            outputs = joint.infer({
                'input_h_enc': enc_xs[:, k],
                'input_h_dec': dec_x[:, 0]
            })
            # print(outputs.keys())
            prob = outputs['Gemm_3']
            pred = prob.argmax(axis=-1).item()

            if pred != NUL:
                dec_x = np.ones((1, 1), dtype=np.long) * pred
                outputs = decoder.infer({
                    'input': dec_x,
                    'input_hidden': dec_h,
                    'input_cell': dec_c,
                })
                # print(outputs.keys())
                dec_x = outputs['Add_26']
                dec_h = outputs['Concat_23']
                dec_c = outputs['Concat_24']
                seq = tokenizer.tokenizer.id_to_token(pred)
                seq = seq.replace('</w>', ' ')
                if verbose > 0:
                    print(seq, end='', flush=True)
                pred_seq += seq
    return pred_seq, total_frames


def load_pytorch_model():
    logdir = os.path.join('logs', FLAGS.name)

    tokenizer = HuggingFaceTokenizer(
        cache_dir=logdir, vocab_size=FLAGS.bpe_size)

    _, transform, input_size = build_transform(
        feature_type=FLAGS.feature, feature_size=FLAGS.feature_size,
        n_fft=FLAGS.n_fft, win_length=FLAGS.win_length,
        hop_length=FLAGS.hop_length, delta=FLAGS.delta, cmvn=FLAGS.cmvn,
        downsample=FLAGS.downsample, pad_to_divisible=False,
        T_mask=FLAGS.T_mask, T_num_mask=FLAGS.T_num_mask,
        F_mask=FLAGS.F_mask, F_num_mask=FLAGS.F_num_mask)

    model_path = os.path.join(logdir, 'models', '%d.pt' % FLAGS.step)
    checkpoint = torch.load(model_path, lambda storage, loc: storage)
    transducer = Transducer(
        vocab_embed_size=FLAGS.vocab_embed_size,
        vocab_size=tokenizer.vocab_size,
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
    )
    transducer.load_state_dict(checkpoint['model'])
    transducer.eval()
    encoder = transducer.encoder
    decoder = transducer.decoder
    joint = transducer.joint
    return encoder, decoder, joint, tokenizer, transform


def load_openvino_model():
    logdir = os.path.join('logs', FLAGS.name)

    tokenizer = HuggingFaceTokenizer(
        cache_dir=logdir, vocab_size=FLAGS.bpe_size)

    _, transform, input_size = build_transform(
        feature_type=FLAGS.feature, feature_size=FLAGS.feature_size,
        n_fft=FLAGS.n_fft, win_length=FLAGS.win_length,
        hop_length=FLAGS.hop_length, delta=FLAGS.delta, cmvn=FLAGS.cmvn,
        downsample=FLAGS.downsample, pad_to_divisible=False,
        T_mask=FLAGS.T_mask, T_num_mask=FLAGS.T_num_mask,
        F_mask=FLAGS.F_mask, F_num_mask=FLAGS.F_num_mask)

    ie = IECore()
    encoder_net = ie.read_network(
        model=os.path.join(logdir, 'encoder.xml'),
        weights=os.path.join(logdir, 'encoder.bin'))
    encoder = ie.load_network(network=encoder_net, device_name='CPU')

    decoder_net = ie.read_network(
        model=os.path.join(logdir, 'decoder.xml'),
        weights=os.path.join(logdir, 'decoder.bin'))
    decoder = ie.load_network(network=decoder_net, device_name='CPU')

    joint_net = ie.read_network(
        model=os.path.join(logdir, 'joint.xml'),
        weights=os.path.join(logdir, 'joint.bin'))
    joint = ie.load_network(network=joint_net, device_name='CPU')

    return encoder, decoder, joint, tokenizer, transform


def main(argv):
    assert FLAGS.step_n_frame % 2 == 0, ("step_n_frame must be divisible by "
                                         "reduction_factor of TimeReduction")

    encoder, decoder, joint, tokenizer, transform = load_pytorch_model()

    dataloader = DataLoader(
        dataset=Subset(
            MergedDataset([
                Librispeech(
                    root=FLAGS.LibriSpeech_test,
                    tokenizer=tokenizer,
                    transform=None,
                    reverse_sorted_by_length=True)]),
            indices=np.arange(10)),
        batch_size=1, shuffle=False, num_workers=0)

    wers = []
    total_time = 0
    total_frame = 0
    with tqdm(dataloader, dynamic_ncols=True) as pbar:
        pbar.set_description("Pytorch full sequence decode")
        for waveform, tokens in pbar:
            true_seq = tokenizer.decode(tokens[0].numpy())
            # pytorch: Encode waveform at a time
            start = time.time()
            pred_seq, frames = pytorch_fullseq_decode(
                encoder, decoder, joint, waveform, transform, tokenizer)
            # pbar.write(true_seq)
            # pbar.write(pred_seq)
            elapsed = time.time() - start
            total_time += elapsed
            total_frame += frames
            wer = jiwer.wer(true_seq, pred_seq)
            wers.append(wer)
            pbar.set_postfix(wer='%.3f' % wer, elapsed='%.3f' % elapsed)
    wer = np.mean(wers)
    print('Mean wer: %.3f, Frame: %d, Time: %.3f, FPS: %.3f, speed: %.3f' % (
        wer, total_frame, total_time, total_frame / total_time,
        total_frame / total_time / 16000))

    wers = []
    total_time = 0
    total_frame = 0
    with tqdm(dataloader, dynamic_ncols=True) as pbar:
        pbar.set_description("Pytorch frame wise decode")
        for waveform, tokens in pbar:
            true_seq = tokenizer.decode(tokens[0].numpy())
            # pytorch: Encode waveform at a time
            start = time.time()
            pred_seq, frames = pytorch_framewise_decode(
                encoder, decoder, joint, waveform, transform, tokenizer)
            elapsed = time.time() - start
            total_time += elapsed
            total_frame += frames
            wer = jiwer.wer(true_seq, pred_seq)
            wers.append(wer)
            pbar.set_postfix(wer='%.3f' % wer, elapsed='%.3f' % elapsed)
    wer = np.mean(wers)
    print('Mean wer: %.3f, Frame: %d, Time: %.3f, FPS: %.3f, speed: %.3f' % (
        wer, total_frame, total_time, total_frame / total_time,
        total_frame / total_time / 16000))

    encoder, decoder, joint, tokenizer, transform = load_openvino_model()
    wers = []
    total_time = 0
    total_frame = 0
    with tqdm(dataloader, dynamic_ncols=True) as pbar:
        pbar.set_description("OpenVINO frame wise decode")
        for waveform, tokens in pbar:
            true_seq = tokenizer.decode(tokens[0].numpy())
            # pytorch: Encode waveform at a time
            start = time.time()
            pred_seq, frames = openvino_framewise_decode(
                encoder, decoder, joint, waveform, tokenizer, transform)
            # pbar.write(true_seq)
            # pbar.write(pred_seq)
            elapsed = time.time() - start
            total_time += elapsed
            total_frame += frames
            wer = jiwer.wer(true_seq, pred_seq)
            wers.append(wer)
            pbar.set_postfix(wer='%.3f' % wer, elapsed='%.3f' % elapsed)
    wer = np.mean(wers)
    print('Mean wer: %.3f, Frame: %d, Time: %.3f, FPS: %.3f, speed: %.3f' % (
        wer, total_frame, total_time, total_frame / total_time,
        total_frame / total_time / 16000))


if __name__ == '__main__':
    app.run(main)
