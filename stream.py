import torch
import os
import time
import random
import argparse
import numpy as np
import torch.nn.functional as F
import torchaudio
import json
import sounddevice as sd
import soundfile as sf
from parts.text.cleaners import english_cleaners
from datetime import datetime
from absl import app, flags

import av
import torch
import torchaudio
from absl import app, flags

from rnnt.args import FLAGS
from rnnt.stream import PytorchStreamDecoder, OpenVINOStreamDecoder

import tempfile
import queue
import sys

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



'''
server:  AudioPreprocessing(
        normalize='none', sample_rate=16000, window_size=0.02, 
        window_stride=0.015, features=args.audio_feat, n_fft=512, log=True,
        feat_type='logfbank', trim_silence=True, window='hann',dither=0.00001, frame_splicing=1, transpose_out=False
    ),
rust:   AudioPreprocessing(
        normalize='none', sample_rate=16000, window_size=0.02, 
        window_stride=0.01, features=args.audio_feat, n_fft=512, log=True,
        feat_type='logfbank', trim_silence=True, window='hann',dither=0.00001, frame_splicing=1, transpose_out=False
    ),

'''
global blank_counter
blank_counter = 0
buffer = []



sd.default.samplerate = 16000


'''
SHALL I NEVER MISS HOME TALK AND BLESSING AND THE COMMON KISS THAT 
COMES TO EACH IN TURN NOR COUNT IT STRANGE WHEN I LOOK UP TO DROP ON 
A NEW RANGE OF WALLS AND FLOORS ANOTHER HOME THAN THIS
'''

def callback(raw_indata, outdata,frames, time, status):
    global buffer
    global encoder_h
    global blank_counter

    if status: # usually something bad
        print("X", flush=True, end=" ")
    else:
        indata = raw_indata.copy()

        buffer.append(indata)
        buffer = buffer[-2:]

        indata = np.concatenate(buffer[-2:], axis=0)

        # print(indata.shape)
        indata = indata / (1<<16)
        waveform = torch.from_numpy(indata.flatten()).float()
        waveform = waveform.unsqueeze(0)

        seq = stream_decoder.decode(waveform)
        if seq == "":
            blank_counter += 1
            if blank_counter == 35:
                print(' [Background]')
                stream_decoder.reset()
        else:
            blank_counter = 0
            print(seq, end='', flush=True)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def test_wav(wav_file):
    import torchaudio

    data, sr = torchaudio.load(wav_file, normalization=True)
    if sr != 16000:
        resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        data = resample(data)
        sr = 16000
    data_ = data[0]
    data_ = data_.unsqueeze(0)
    seq = stream_decoder.decode(data_)
    print(seq)


def main(argv):
    global stream_decoder
    stream_decoder = PytorchStreamDecoder(FLAGS)
    duration = 80
    if FLAGS.path is not None:
        test_wav(FLAGS.path)
    else:
        with sd.Stream(channels=1,dtype='float32', samplerate=16000, 
            blocksize=FLAGS.win_length*FLAGS.step_n_frame+ (FLAGS.step_n_frame-1), callback=callback, 
            latency='high'):

            sd.sleep(duration * 1000)

if __name__ == "__main__":
    app.run(main)

