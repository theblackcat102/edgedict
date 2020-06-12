import os

import numpy as np
import onnxruntime
import torch.onnx
from torch.utils.data import DataLoader
from absl import app, flags

from rnnt.args import FLAGS                             # define training FLAGS
from rnnt.transforms import build_transform
from rnnt.tokenizer import HuggingFaceTokenizer
from rnnt.models import Transducer
from rnnt.dataset import MergedDataset, Librispeech, seq_collate


flags.DEFINE_string('model_dir', './logs/E6D2-Adam-NV-20200609-141601',
                    help='path to root dir of log')
flags.DEFINE_integer('step', 105000, help='steps of checkpoint')
flags.DEFINE_integer('step_n_frame', 4, help='input frame(stacked)')
flags.mark_flags_as_required(['model_dir', 'step'])


def main(argv):
    assert FLAGS.step_n_frame % 2 == 0, ("step_n_frame must be divisible by "
                                         "reduction_factor of TimeReduction")

    tokenizer = HuggingFaceTokenizer(
        cache_dir=FLAGS.model_dir, vocab_size=FLAGS.bpe_size)

    _, transform, input_size = build_transform(
        feature_type=FLAGS.feature, feature_size=FLAGS.feature_size,
        n_fft=FLAGS.n_fft, win_length=FLAGS.win_length,
        hop_length=FLAGS.hop_length, delta=FLAGS.delta, cmvn=FLAGS.cmvn,
        downsample=FLAGS.downsample,
        T_mask=FLAGS.T_mask, T_num_mask=FLAGS.T_num_mask,
        F_mask=FLAGS.F_mask, F_num_mask=FLAGS.F_num_mask
    )

    dataloader = DataLoader(
        dataset=MergedDataset([
            Librispeech(
                root=FLAGS.LibriSpeech_test,
                tokenizer=tokenizer,
                transform=None,
                reverse_sorted_by_length=True)]),
        batch_size=1, shuffle=False, num_workers=0)

    model_path = os.path.join(
        FLAGS.model_dir, 'models', 'epoch-%d' % FLAGS.step)
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

    n_frame_iter = (
        FLAGS.win_length +
        FLAGS.hop_length * (FLAGS.downsample * FLAGS.step_n_frame - 1))

    for waveform, tokens in dataloader:
        for start in range(0, len(waveform), n_frame_iter):
            xs = transform(waveform[:, start: start + n_frame_iter])
            print(xs.shape)


if __name__ == '__main__':
    app.run(main)
