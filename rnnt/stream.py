import time

import os
import numpy as np
import torch
try:
    from openvino.inference_engine import IECore
except:
    pass
from rnnt.models import Transducer, convert_lightning2normal
from rnnt.transforms import build_transform
from rnnt.tokenizer import HuggingFaceTokenizer, BOS, NUL


class StreamTransducerDecoder:
    def reset_profile(self):
        self.encoder_elapsed = []
        self.decoder_elapsed = []
        self.joint_elapsed = []

    def reset(self):
        raise NotImplementedError()

    def decode(self, frame):
        raise NotImplementedError()


class PytorchStreamDecoder(StreamTransducerDecoder):
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        logdir = os.path.join('logs', FLAGS.name)

        self.tokenizer = HuggingFaceTokenizer(
            cache_dir='BPE-'+str(FLAGS.bpe_size), vocab_size=FLAGS.bpe_size)
        
        assert self.tokenizer.tokenizer != None

        _, self.transform, input_size = build_transform(
            feature_type=FLAGS.feature, feature_size=FLAGS.feature_size,
            n_fft=FLAGS.n_fft, win_length=FLAGS.win_length,
            hop_length=FLAGS.hop_length, delta=FLAGS.delta, cmvn=FLAGS.cmvn,
            downsample=FLAGS.downsample, pad_to_divisible=False,
            T_mask=FLAGS.T_mask, T_num_mask=FLAGS.T_num_mask,
            F_mask=FLAGS.F_mask, F_num_mask=FLAGS.F_num_mask)

        model_path = os.path.join(logdir, 'models', FLAGS.model_name)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, lambda storage, loc: storage)
        else:
            model_path = os.path.join(logdir, FLAGS.model_name)
            checkpoint = torch.load(model_path, lambda storage, loc: storage)

        transducer = Transducer(
            vocab_embed_size=FLAGS.vocab_embed_size,
            vocab_size=self.tokenizer.vocab_size,
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
            output_loss=False,
        )

        transducer.load_state_dict(convert_lightning2normal(checkpoint)['model'])
        transducer.eval()
        self.encoder = transducer.encoder
        self.decoder = transducer.decoder
        self.joint = transducer.joint

        self.reset_profile()
        self.reset()

    @torch.no_grad()
    def reset(self):
        self.enc_h = torch.zeros(
            self.FLAGS.enc_layers, 1, self.FLAGS.enc_hidden_size)
        self.enc_c = torch.zeros(
            self.FLAGS.enc_layers, 1, self.FLAGS.enc_hidden_size)

        dec_x = torch.ones(1, 1).long() * BOS
        dec_h = torch.zeros(
            self.FLAGS.dec_layers, 1, self.FLAGS.dec_hidden_size)
        dec_c = torch.zeros(
            self.FLAGS.dec_layers, 1, self.FLAGS.dec_hidden_size)
        self.dec_x, (self.dec_h, self.dec_c) = self.decoder(
            dec_x, (dec_h, dec_c))

    @torch.no_grad()
    def decode(self, frame):
        start = time.time()
        xs = self.transform(frame).transpose(1, 2)
        enc_xs, (self.enc_h, self.enc_c) = self.encoder(
            xs, (self.enc_h, self.enc_c))
        self.encoder_elapsed.append(time.time() - start)

        tokens = []
        for k in range(enc_xs.shape[1]):
            start = time.time()
            prob = self.joint(enc_xs[:, k], self.dec_x[:, 0])
            pred = prob.argmax(dim=-1).item()
            if self.tokenizer.tokenizer.id_to_token(pred) == '<unk>':
                prob[:, pred] = 0
                pred = prob.argmax(dim=-1).item()
            self.joint_elapsed.append(time.time() - start)

            if pred != NUL:
                dec_x = torch.ones(1, 1).long() * pred
                start = time.time()
                self.dec_x, (self.dec_h, self.dec_c) = self.decoder(
                    dec_x, (self.dec_h, self.dec_c))
                self.decoder_elapsed.append(time.time() - start)
                seq = self.tokenizer.tokenizer.id_to_token(pred)
                seq = seq.replace('</w>', ' ')
                tokens.append(seq)
        return "".join(tokens)


class OpenVINOStreamDecoder(StreamTransducerDecoder):
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        logdir = os.path.join('logs', FLAGS.name)

        self.tokenizer = HuggingFaceTokenizer(
            cache_dir=logdir, vocab_size=FLAGS.bpe_size)

        _, self.transform, input_size = build_transform(
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
        self.encoder = ie.load_network(network=encoder_net, device_name='CPU')

        decoder_net = ie.read_network(
            model=os.path.join(logdir, 'decoder.xml'),
            weights=os.path.join(logdir, 'decoder.bin'))
        self.decoder = ie.load_network(network=decoder_net, device_name='CPU')

        joint_net = ie.read_network(
            model=os.path.join(logdir, 'joint.xml'),
            weights=os.path.join(logdir, 'joint.bin'))
        self.joint = ie.load_network(network=joint_net, device_name='CPU')

        self.reset_profile()
        self.reset()

    def reset(self):
        self.enc_h = np.zeros(
            (self.FLAGS.enc_layers, 1, self.FLAGS.enc_hidden_size),
            dtype=np.float)
        self.enc_c = np.zeros(
            (self.FLAGS.enc_layers, 1, self.FLAGS.enc_hidden_size),
            dtype=np.float)

        dec_x = np.ones((1, 1), dtype=np.long) * BOS
        dec_h = np.zeros(
            (self.FLAGS.dec_layers, 1, self.FLAGS.dec_hidden_size),
            dtype=np.float)
        dec_c = np.zeros(
            (self.FLAGS.dec_layers, 1, self.FLAGS.dec_hidden_size),
            dtype=np.float)
        outputs = self.decoder.infer({
            'input': dec_x,
            'input_hidden': dec_h,
            'input_cell': dec_c,
        })
        # print(outputs.keys())
        self.dec_x = outputs['Add_26']
        self.dec_h = outputs['Concat_23']
        self.dec_c = outputs['Concat_24']

    def decode(self, frame):
        start = time.time()
        xs = self.transform(frame).transpose(1, 2).numpy()
        outputs = self.encoder.infer(inputs={
            'input': xs,
            'input_hidden': self.enc_h,
            'input_cell': self.enc_c,
        })
        enc_xs = outputs['Add_156']
        self.enc_h = outputs['Concat_153']
        self.enc_c = outputs['Concat_154']
        self.encoder_elapsed.append(time.time() - start)

        tokens = []
        for k in range(enc_xs.shape[1]):
            start = time.time()
            outputs = self.joint.infer({
                'input_h_enc': enc_xs[:, k],
                'input_h_dec': self.dec_x[:, 0]
            })
            prob = outputs['Gemm_3']
            pred = prob.argmax(axis=-1).item()
            self.joint_elapsed.append(time.time() - start)

            if pred != NUL:
                dec_x = np.ones((1, 1), dtype=np.long) * pred
                start = time.time()
                outputs = self.decoder.infer({
                    'input': dec_x,
                    'input_hidden': self.dec_h,
                    'input_cell': self.dec_c,
                })
                # print(outputs.keys())
                self.dec_x = outputs['Add_26']
                self.dec_h = outputs['Concat_23']
                self.dec_c = outputs['Concat_24']
                self.decoder_elapsed.append(time.time() - start)
                seq = self.tokenizer.tokenizer.id_to_token(pred)
                seq = seq.replace('</w>', ' ')
                tokens.append(seq)
        return "".join(tokens)
