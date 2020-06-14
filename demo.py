import os
import subprocess

import av
import numpy as np
import torch
from absl import app, flags
from openvino.inference_engine import IECore

from rnnt.args import FLAGS
from rnnt.tokenizer import HuggingFaceTokenizer, BOS, NUL
from rnnt.transforms import build_transform
av.logging.set_level(av.logging.ERROR)


flags.DEFINE_integer('step_n_frame', 10, help='input frame(stacked)')
flags.DEFINE_string('url', 'https://www.youtube.com/watch?v=2EppLNonncc',
                    help='youtube live link')


class OpenVINOStreamDecoder():
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
        xs = self.transform(frame).transpose(1, 2).numpy()
        outputs = self.encoder.infer(inputs={
            'input': xs,
            'input_hidden': self.enc_h,
            'input_cell': self.enc_c,
        })
        # print(outputs.keys())
        enc_xs = outputs['Add_156']
        self.enc_h = outputs['Concat_153']
        self.enc_c = outputs['Concat_154']

        tokens = []
        for k in range(enc_xs.shape[1]):
            outputs = self.joint.infer({
                'input_h_enc': enc_xs[:, k],
                'input_h_dec': self.dec_x[:, 0]
            })
            # print(outputs.keys())
            prob = outputs['Gemm_3']
            pred = prob.argmax(axis=-1).item()

            if pred != NUL:
                dec_x = np.ones((1, 1), dtype=np.long) * pred
                outputs = self.decoder.infer({
                    'input': dec_x,
                    'input_hidden': self.dec_h,
                    'input_cell': self.dec_c,
                })
                # print(outputs.keys())
                self.dec_x = outputs['Add_26']
                self.dec_h = outputs['Concat_23']
                self.dec_c = outputs['Concat_24']
                seq = self.tokenizer.tokenizer.id_to_token(pred)
                seq = seq.replace('</w>', ' ')
                tokens.append(seq)
        return "".join(tokens)


def main(argv):
    '''
        youtube-dl
        pip install av
    '''
    print(FLAGS.url)

    filepath = 'bloom.mp3'
    save_strean = False

    command = ['youtube-dl', '-f', '91', '-g', FLAGS.url]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**8)
    out, err = proc.communicate()
    videolink = out.decode("utf-8").strip()

    resampler = av.AudioResampler("s16p", layout=1, rate=16 * 1000)

    if save_strean:
        output_container = av.open(filepath, 'w')
        output_stream = output_container.add_stream('mp3')

    input_container = av.open(videolink)
    input_stream = input_container.streams.get(audio=0)[0]

    win_size = (
        FLAGS.win_length +
        FLAGS.hop_length * (FLAGS.downsample * FLAGS.step_n_frame - 1))
    hop_size = (
        FLAGS.hop_length * (FLAGS.downsample * FLAGS.step_n_frame))

    stream_decoder = OpenVINOStreamDecoder(FLAGS)

    track_counter = 0
    buffers = torch.empty(0)
    for frame in input_container.decode(input_stream):
        frame.pts = None
        resample_frame = resampler.resample(frame)

        waveform = np.frombuffer(
            resample_frame.planes[0].to_bytes(), dtype='int16')
        waveform = torch.tensor(waveform.copy())
        waveform = waveform.float() / 32768

        # waveform = waveform.clamp(-1, 1)
        # waveform[waveform != waveform] = 0
        if torch.isnan(waveform).any():
            print("[NAN]", flush=True, end=" ")

        if len(buffers) < win_size:
            buffers = torch.cat([buffers, waveform], dim=0)
        else:
            print("[BUFFER OVERFLOW]", flush=True, end=" ")

        if len(buffers) >= win_size:
            waveform = buffers[:win_size]
            buffers = buffers[hop_size:]
            if torch.isnan(waveform).any():
                print("[NAN] waveform", flush=True, end=" ")
                continue

            seq = stream_decoder.decode(waveform[None])
            print(seq, end='', flush=True)

            track_counter += 1
            if track_counter % 200 == 0:
                print('[reset state]')
                stream_decoder.reset()

        if save_strean:
            for packet in output_stream.encode(resample_frame):
                output_container.mux(packet)

    if save_strean:
        for packet in output_stream.encode(None):
            output_container.mux(packet)
        output_container.close()


if __name__ == "__main__":
    app.run(main)
