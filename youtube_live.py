import subprocess
from datetime import datetime

import av
import torch
import torchaudio
from absl import app, flags

from rnnt.args import FLAGS
from rnnt.stream import PytorchStreamDecoder, OpenVINOStreamDecoder
av.logging.set_level(av.logging.ERROR)

# PytorchStreamDecoder
flags.DEFINE_string('model_name', "last.pt", help='steps of checkpoint')
flags.DEFINE_integer('step_n_frame', 2, help='input frame(stacked)')

flags.DEFINE_enum('stream_decoder', 'torch', ['torch', 'openvino'],
                  help='stream decoder implementation')
flags.DEFINE_string('url', 'https://www.youtube.com/watch?v=2EppLNonncc',
                    help='youtube live link')
flags.DEFINE_integer('reset_step', 500, help='reset hidden state')
flags.DEFINE_string('path', None, help='path to .wav')


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


def wav():
    waveform, sr = torchaudio.load(FLAGS.path, normalization=True)
    if sr != 16000:
        resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resample(waveform)
        sr = 16000

    stream_decoder = PytorchStreamDecoder(FLAGS)
    print("Pytorch: ")
    seq = stream_decoder.decode(waveform[:1])
    print(seq)

    stream_decoder = OpenVINOStreamDecoder(FLAGS)
    print("OpenVINO: ")
    seq, _ = stream_decode(stream_decoder, waveform[:1])
    print(seq)

    print(seq)


def live():
    '''
        youtube-dl
        pip install av
    '''
    filepath = './youtube_live.mp3'
    save_strean = True
    infinite = True
    duration = 50                    # seconds

    command = ['youtube-dl', '-f', '91', '-g', FLAGS.url]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**8)
    out, err = proc.communicate()
    videolink = out.decode("utf-8").strip()
    resampler = av.AudioResampler("s16p", layout=1, rate=16 * 1000)

    if not infinite and save_strean:
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

    # track_counter = 0
    begin_time = datetime.now()
    buffer = torch.empty(1, 0)
    blank_counter = 0
    for frame in input_container.decode(input_stream):
        frame.pts = None
        resample_frame = resampler.resample(frame)

        waveform = resample_frame.to_ndarray()
        waveform = torch.tensor(waveform.copy())
        waveform = waveform.float() / 32768

        if torch.isnan(waveform).any():
            print("[NAN]", flush=True, end=" ")

        if buffer.shape[1] < win_size:
            buffer = torch.cat([buffer, waveform], dim=-1)

        while buffer.shape[1] >= win_size:
            waveform = buffer[:, :win_size]
            buffer = buffer[:, hop_size:]
            if torch.isnan(waveform).any():
                print("[NAN] waveform", flush=True, end=" ")
                continue
            seq = stream_decoder.decode(waveform)
            if seq == "":
                blank_counter += 1
                if blank_counter == 35:
                    print(' [Background]')
                    stream_decoder.reset()
            else:
                blank_counter = 0
                print(seq, end='', flush=True)

        if not infinite and save_strean:
            for packet in output_stream.encode(resample_frame):
                output_container.mux(packet)

        if not infinite:
            if (datetime.now() - begin_time).total_seconds() > duration:
                break

    if not infinite and save_strean:
        for packet in output_stream.encode(None):
            output_container.mux(packet)
        output_container.close()


def main(argv):
    if FLAGS.path is not None:
        wav()
    else:
        live()


if __name__ == "__main__":
    app.run(main)
