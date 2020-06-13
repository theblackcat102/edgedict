import subprocess

import av
import numpy as np
import torch
from absl import app, flags

from rnnt.args import FLAGS
from rnnt.stream import PytorchStreamDecoder, OpenVINOStreamDecoder
av.logging.set_level(av.logging.ERROR)


flags.DEFINE_integer('step', 105000, help='steps of checkpoint')
flags.DEFINE_integer('step_n_frame', 10, help='input frame(stacked)')
flags.DEFINE_enum('stream_decoder', 'openvino', ['torch', 'openvino'],
                  help='stream decoder implementation')
flags.DEFINE_string('url', 'https://www.youtube.com/watch?v=2EppLNonncc',
                    help='youtube live link')


def main(argv):
    '''
        youtube-dl
        pip install av
    '''
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

    if FLAGS.stream_decoder == 'torch':
        stream_decoder = PytorchStreamDecoder(FLAGS)
    else:
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
