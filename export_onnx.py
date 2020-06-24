import os

import numpy as np
import onnxruntime
import torch.onnx
from absl import app, flags

from rnnt.args import FLAGS                             # define training FLAGS
from rnnt.transforms import build_transform
from rnnt.tokenizer import HuggingFaceTokenizer
from rnnt.models import Transducer


flags.DEFINE_string('model_name', "last.pt", help='checkpoint name')
flags.DEFINE_integer('step_n_frame', 2, help='input frame(stacked)')


def export_encoder(transducer, input_size, vocab_size, logdir):
    print("=" * 40)
    assert FLAGS.step_n_frame % 2 == 0, ("step_n_frame must be divisible by "
                                         "reduction_factor of TimeReduction")
    encoder = transducer.encoder
    encoder.eval()
    x = torch.rand(1, FLAGS.step_n_frame, input_size, requires_grad=True)
    x_h = torch.rand(
        FLAGS.enc_layers, 1, FLAGS.enc_hidden_size, requires_grad=True)
    x_c = torch.rand(
        FLAGS.enc_layers, 1, FLAGS.enc_hidden_size, requires_grad=True)
    y, (y_h, y_c) = encoder(x, (x_h, x_c))

    input_names = ['input', 'input_hidden', 'input_cell']
    output_names = ['output', 'output_hidden', 'output_cell']
    path = os.path.join(logdir, 'encoder.onnx')
    torch.onnx.export(
        encoder,
        (x, (x_h, x_c)),
        path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        example_outputs=(y, (y_h, y_c)),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            'input': {0: 'batch_size'},
            'input_hidden': {1: 'batch_size'},
            'input_cell': {1: 'batch_size'},
            'output': {0: 'batch_size'},
            'output_hidden': {1: 'batch_size'},
            'output_cell': {1: 'batch_size'},
        },
        verbose=True
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
    for name in input_names:
        print("%-12s : %s" % (name, str(inputs[name].shape)))
    for name, value in zip(output_names, [onnx_y, onnx_y_h, onnx_y_c]):
        print("%-12s : %s" % (name, str(value.shape)))


def export_decoder(transducer, input_size, vocab_size, logdir):
    print("=" * 40)
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
    path = os.path.join(logdir, 'decoder.onnx')
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
        },
        verbose=True
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
    for name in input_names:
        print("%-12s : %s" % (name, str(inputs[name].shape)))
    for name, value in zip(output_names, [onnx_y, onnx_y_h, onnx_y_c]):
        print("%-12s : %s" % (name, str(value.shape)))


def export_join(transducer, input_size, vocab_size, logdir):
    print("=" * 40)
    joint = transducer.joint
    joint.eval()
    h_enc = torch.rand(1, FLAGS.enc_proj_size, requires_grad=True)
    h_dec = torch.rand(1, FLAGS.dec_proj_size, requires_grad=True)
    y = joint(h_enc, h_dec)

    input_names = ['input_h_enc', 'input_h_dec']
    output_names = ['output']
    path = os.path.join(logdir, 'joint.onnx')
    torch.onnx.export(
        joint,
        (h_enc, h_dec),
        path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        example_outputs=y,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            'input_h_enc': {0: 'batch_size'},
            'input_h_dec': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
        verbose=True
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
    for name in input_names:
        print("%-12s : %s" % (name, str(inputs[name].shape)))
    for name, value in zip(output_names, [onnx_y]):
        print("%-12s : %s" % (name, str(value.shape)))


def main(argv):
    assert FLAGS.step_n_frame % 2 == 0, ("step_n_frame must be divisible by "
                                         "reduction_factor of TimeReduction")

    logdir = os.path.join('logs', FLAGS.name)

    tokenizer = HuggingFaceTokenizer(
        cache_dir=logdir, vocab_size=FLAGS.bpe_size)

    transform_train, transform_test, input_size = build_transform(
        feature_type=FLAGS.feature, feature_size=FLAGS.feature_size,
        n_fft=FLAGS.n_fft, win_length=FLAGS.win_length,
        hop_length=FLAGS.hop_length, delta=FLAGS.delta, cmvn=FLAGS.cmvn,
        downsample=FLAGS.downsample,
        T_mask=FLAGS.T_mask, T_num_mask=FLAGS.T_num_mask,
        F_mask=FLAGS.F_mask, F_num_mask=FLAGS.F_num_mask
    )

    model_path = os.path.join(logdir, 'models', FLAGS.model_name)
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

    export_encoder(transducer, input_size, tokenizer.vocab_size, logdir)
    export_decoder(transducer, input_size, tokenizer.vocab_size, logdir)
    export_join(transducer, input_size, tokenizer.vocab_size, logdir)


if __name__ == '__main__':
    app.run(main)
