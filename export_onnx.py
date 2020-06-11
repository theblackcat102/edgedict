import os

import numpy as np
import onnxruntime
import torch.onnx
import torch.nn as nn
from absl import app, flags

import train                                    # define training FLAGS
from transforms import build_transform
from tokenizer import HuggingFaceTokenizer
from models import Transducer, Encoder, ResLayerNormLSTM


FLAGS = flags.FLAGS
flags.DEFINE_string('model_dir', None, help='path to root dir of log')
flags.DEFINE_integer('step', None, help='steps of checkpoint')
flags.DEFINE_integer('n_frame', 4, help='input frame(stacked)')
flags.mark_flags_as_required(['model_dir', 'step'])


def export_encoder(transducer, input_size, vocab_size):
    assert FLAGS.n_frame % 2 == 0, ("n_frame must be divisible by "
                                    "reduction_factor of TimeReduction")
    encoder = transducer.encoder
    encoder.eval()
    x = torch.rand(1, FLAGS.n_frame, input_size, requires_grad=True)
    x_h = torch.rand(
        FLAGS.enc_layers, 1, FLAGS.enc_hidden_size, requires_grad=True)
    x_c = torch.rand(
        FLAGS.enc_layers, 1, FLAGS.enc_hidden_size, requires_grad=True)
    y, (y_h, y_c) = encoder(x, (x_h, x_c))

    input_names = ['input', 'input_hidden', 'input_cell']
    output_names = ['output', 'output_hidden', 'output_cell']
    path = os.path.join(FLAGS.model_dir, 'encoder.onnx')
    torch.onnx.export(
        encoder,
        (x, (x_h, x_c)),
        path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            'input': {0: 'batch_size'},
            'input_hidden': {1: 'batch_size'},
            'input_cell': {1: 'batch_size'},
            'output': {0: 'batch_size'},
            'output_hidden': {1: 'batch_size'},
            'output_cell': {1: 'batch_size'},
        }
    )

    session = onnxruntime.InferenceSession(path)
    inputs = {
        'input': x.detach().numpy(),
        'input_hidden': x_h.detach().numpy(),
        'input_cell': x_c.detach().numpy(),
    }
    onnx_y, onnx_y_h, onnx_y_c = session.run(output_names, inputs)

    np.testing.assert_allclose(
        y.detach().numpy(), onnx_y, rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(
        y_h.detach().numpy(), onnx_y_h, rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(
        y_c.detach().numpy(), onnx_y_c, rtol=1e-03, atol=1e-05)

    print("Encoder has been exported")


def export_decoder(transducer, input_size, vocab_size):
    decoder = transducer.decoder
    decoder.eval()
    x = torch.randint(0, vocab_size, size=(1, 1))
    x_h = torch.rand(
        FLAGS.dec_layers, 1, FLAGS.dec_hidden_size, requires_grad=True)
    x_c = torch.rand(
        FLAGS.dec_layers, 1, FLAGS.dec_hidden_size, requires_grad=True)
    y, (y_h, y_c) = decoder(x, (x_h, x_c))

    input_names = ['input', 'input_hidden', 'input_cell']
    output_names = ['output', 'output_hidden', 'output_cell']
    path = os.path.join(FLAGS.model_dir, 'decoder.onnx')
    torch.onnx.export(
        decoder,
        (x, (x_h, x_c)),
        path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            'input': {0: 'batch_size'},
            'input_hidden': {1: 'batch_size'},
            'input_cell': {1: 'batch_size'},
            'output': {0: 'batch_size'},
            'output_hidden': {1: 'batch_size'},
            'output_cell': {1: 'batch_size'},
        }
    )

    session = onnxruntime.InferenceSession(path)
    inputs = {
        'input': x.detach().numpy(),
        'input_hidden': x_h.detach().numpy(),
        'input_cell': x_c.detach().numpy(),
    }
    onnx_y, onnx_y_h, onnx_y_c = session.run(output_names, inputs)

    np.testing.assert_allclose(
        y.detach().numpy(), onnx_y, rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(
        y_h.detach().numpy(), onnx_y_h, rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(
        y_c.detach().numpy(), onnx_y_c, rtol=1e-03, atol=1e-05)

    print("Decoder has been exported")


def export_join(transducer, input_size, vocab_size):
    joint = transducer.joint
    joint.eval()
    h_enc = torch.rand(1, FLAGS.enc_proj_size, requires_grad=True)
    h_dec = torch.rand(1, FLAGS.dec_proj_size, requires_grad=True)
    y = joint(h_enc, h_dec)

    input_names = ['input_h_enc', 'input_h_dec']
    output_names = ['output']
    path = os.path.join(FLAGS.model_dir, 'joint.onnx')
    torch.onnx.export(
        joint,
        (h_enc, h_dec),
        path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            'input_h_enc': {0: 'batch_size'},
            'input_h_dec': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }
    )

    session = onnxruntime.InferenceSession(path)
    inputs = {
        'input_h_enc': h_enc.detach().numpy(),
        'input_h_dec': h_dec.detach().numpy(),
    }
    onnx_y = session.run(output_names, inputs)[0]

    np.testing.assert_allclose(
        y.detach().numpy(), onnx_y, rtol=1e-03, atol=1e-05)

    print("Joint has been exported")


def main(argv):
    assert FLAGS.n_frame % 2 == 0, ("n_frame must be divisible by "
                                    "reduction_factor of TimeReduction")

    tokenizer = HuggingFaceTokenizer(
        cache_dir=FLAGS.model_dir, vocab_size=FLAGS.bpe_size)

    transform_train, transform_test, input_size = build_transform(
        feature_type=FLAGS.feature, feature_size=FLAGS.feature_size,
        n_fft=FLAGS.n_fft, win_length=FLAGS.win_length,
        hop_length=FLAGS.hop_length, delta=FLAGS.delta, cmvn=FLAGS.cmvn,
        downsample=FLAGS.downsample,
        T_mask=FLAGS.T_mask, T_num_mask=FLAGS.T_num_mask,
        F_mask=FLAGS.F_mask, F_num_mask=FLAGS.F_num_mask
    )

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

    export_encoder(transducer, input_size, tokenizer.vocab_size)
    export_decoder(transducer, input_size, tokenizer.vocab_size)
    export_join(transducer, input_size, tokenizer.vocab_size)


if __name__ == '__main__':
    app.run(main)
