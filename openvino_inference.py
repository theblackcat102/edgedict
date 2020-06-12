import os

# import numpy as np
# import onnxruntime
import torch.onnx
from torch.utils.data import DataLoader
from absl import app, flags

from rnnt.args import FLAGS                             # define training FLAGS
from rnnt.transforms import build_transform
from rnnt.tokenizer import HuggingFaceTokenizer, BOS, NUL
from rnnt.models import Transducer
from rnnt.dataset import MergedDataset, Librispeech


flags.DEFINE_integer('step', 105000, help='steps of checkpoint')
flags.DEFINE_integer('step_n_frame', 2, help='input frame(stacked)')


def main(argv):
    assert FLAGS.step_n_frame % 2 == 0, ("step_n_frame must be divisible by "
                                         "reduction_factor of TimeReduction")

    logdir = os.path.join('logs', FLAGS.name)

    tokenizer = HuggingFaceTokenizer(
        cache_dir=logdir, vocab_size=FLAGS.bpe_size)

    _, transform, input_size = build_transform(
        feature_type=FLAGS.feature, feature_size=FLAGS.feature_size,
        n_fft=FLAGS.n_fft, win_length=FLAGS.win_length,
        hop_length=FLAGS.hop_length, delta=FLAGS.delta, cmvn=FLAGS.cmvn,
        downsample=3, pad_to_divisible=False,
        T_mask=FLAGS.T_mask, T_num_mask=FLAGS.T_num_mask,
        F_mask=FLAGS.F_mask, F_num_mask=FLAGS.F_num_mask,
    )
    input_size = 240
    dataloader = DataLoader(
        dataset=MergedDataset([
            Librispeech(
                root=FLAGS.LibriSpeech_test,
                tokenizer=tokenizer,
                transform=None,
                reverse_sorted_by_length=True)]),
        batch_size=1, shuffle=False, num_workers=0)

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

    n_frame_iter = (
        FLAGS.win_length +
        FLAGS.hop_length * (FLAGS.downsample * FLAGS.step_n_frame - 1))

    for waveform, tokens in dataloader:
        enc_h = torch.zeros(FLAGS.enc_layers, 1, FLAGS.enc_hidden_size)
        enc_c = torch.zeros(FLAGS.enc_layers, 1, FLAGS.enc_hidden_size)

        dec_x = torch.ones(1, 1).long() * BOS
        dec_h = torch.zeros(FLAGS.dec_layers, 1, FLAGS.dec_hidden_size)
        dec_c = torch.zeros(FLAGS.dec_layers, 1, FLAGS.dec_hidden_size)
        dec_x, (dec_h, dec_c) = decoder(dec_x, (dec_h, dec_c))

        preds = []

        for start in range(0, len(waveform), n_frame_iter):
            xs = transform(waveform).transpose(1, 2)
            xs = transform(
                waveform[:, start: start + n_frame_iter]).transpose(1, 2)
            enc_x, (enc_h, enc_c) = encoder(xs, (enc_h, enc_c))
            prob = joint(enc_x[:, 0], dec_x[:, 0])
            print(prob.shape)
            pred = prob.argmax(dim=-1).item()

            if pred != NUL:
                dec_x = torch.ones(1, 1).long() * pred
                dec_x, (dec_h, dec_c) = decoder(dec_x, (dec_h, dec_c))
                preds.append(pred)

        true_seq = tokenizer.decode(tokens[0].numpy())
        pred_seq = tokenizer.decode(preds)

        print(true_seq)
        print("=" * 40)
        print(pred_seq)

        exit(0)


if __name__ == '__main__':
    app.run(main)
