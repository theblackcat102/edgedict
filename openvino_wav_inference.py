import os
import time

import numpy as np
import jiwer
from absl import app, flags
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from rnnt.args import FLAGS                             # define training FLAGS
from rnnt.tokenizer import HuggingFaceTokenizer
from rnnt.dataset import MergedDataset, Librispeech
from rnnt.stream import OpenVINOStreamDecoder, PytorchStreamDecoder


flags.DEFINE_string('model_name', "last.pt", help='steps of checkpoint')
flags.DEFINE_integer('step_n_frame', 10, help='input frame(stacked)')


def fullseq_decode(fullseq_decoder, waveform, verbose=0):
    # Encode waveform at a time
    total_frames = waveform.shape[1]
    pred_seq = fullseq_decoder.decode(waveform)
    return pred_seq, total_frames


def stream_decode(stream_decoder, waveform, verbose=0):
    win_size = (
        FLAGS.win_length +
        FLAGS.hop_length * (FLAGS.downsample * FLAGS.step_n_frame - 1))
    hop_size = (
        FLAGS.hop_length * (FLAGS.downsample * FLAGS.step_n_frame))

    pred_seq = ""
    total_frames = FLAGS.win_length
    stream_decoder.reset()
    for start in range(0, waveform.shape[1] - win_size, hop_size):
        total_frames += hop_size
        seq = stream_decoder.decode(waveform[:, start: start + win_size])
        if verbose > 0:
            print(seq, end='', flush=True)
        pred_seq += seq

    return pred_seq, total_frames


def main(argv):
    assert FLAGS.step_n_frame % 2 == 0, ("step_n_frame must be divisible by "
                                         "reduction_factor of TimeReduction")

    tokenizer = HuggingFaceTokenizer(
        cache_dir=os.path.join('logs', FLAGS.name), vocab_size=FLAGS.bpe_size)

    dataloader = DataLoader(
        dataset=Subset(
            MergedDataset([
                Librispeech(
                    root=FLAGS.LibriSpeech_test,
                    tokenizer=tokenizer,
                    transform=None,
                    reverse_sorted_by_length=False)]),
            indices=np.arange(10)),
        batch_size=1, shuffle=False, num_workers=0)

    pytorch_decoder = PytorchStreamDecoder(FLAGS)
    wers = []
    total_time = 0
    total_frame = 0
    with tqdm(dataloader, dynamic_ncols=True) as pbar:
        pbar.set_description("Pytorch full sequence decode")
        for waveform, tokens in pbar:
            true_seq = tokenizer.decode(tokens[0].numpy())
            # pytorch: Encode waveform at a time
            start = time.time()
            pred_seq, frames = fullseq_decode(pytorch_decoder, waveform)
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
            pred_seq, frames = stream_decode(pytorch_decoder, waveform)
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

    openvino_decoder = OpenVINOStreamDecoder(FLAGS)
    wers = []
    total_time = 0
    total_frame = 0
    with tqdm(dataloader, dynamic_ncols=True) as pbar:
        pbar.set_description("OpenVINO frame wise decode")
        for waveform, tokens in pbar:
            true_seq = tokenizer.decode(tokens[0].numpy())
            # pytorch: Encode waveform at a time
            start = time.time()
            pred_seq, frames = stream_decode(openvino_decoder, waveform)
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
