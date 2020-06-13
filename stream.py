import torch
import os
import time
import random
import argparse
import numpy as np
import torch.nn.functional as F
import torchaudio
from models import Transducer
from tokenizer import HuggingFaceTokenizer, CharTokenizer
import json
import sounddevice as sd
import soundfile as sf
from parts.features import AudioPreprocessing
from parts.text.cleaners import english_cleaners
from recurrent import MFCC_
from augmentation import ConcatFeature
from pydub import AudioSegment, effects  
import tempfile
import queue
import sys
from speechpy.processing import cmvn, cmvnw


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

parser = argparse.ArgumentParser(description='RNN-T')
parser.add_argument('--name', type=str)
parser.add_argument('-w', '--window-size', type=float, default=0.02)
parser.add_argument('-m', '--mode', type=str, default='greedy', choices=['greedy', 'beam'])
parser.add_argument('-u', '--url', type=str, default='https://www.youtube.com/watch?v=dp8PhLsUcFE')
eval_args = parser.parse_args()

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)



best_checkpoint = os.path.join(eval_args.name, 'amp_checkpoint.pt')
if not os.path.exists(best_checkpoint):
    raise ValueError('Not found')

checkpoint = torch.load(best_checkpoint, map_location= 'cpu')
with open(os.path.join(eval_args.name, 'vars.json'), 'r') as f:
    params = json.load(f)
print('Checkpoint at epoch %d ' % checkpoint['epoch'])
args = Struct(**params)
window_size = eval_args.window_size
window_stride = 0.01
sd.default.samplerate = 16000
duration = 60  # seconds

if args.tokenizer == 'char':
    _tokenizer = CharTokenizer()
else:
    _tokenizer = HuggingFaceTokenizer() # use BPE-400
    print('use bpe')

model = Transducer(args.audio_feat, _tokenizer.vocab_size,
        args.vocab_dim, # vocab embedding dim
        args.h_dim, # hidden dim
        args.layers, pred_num_layers=args.pred_layers, dropout=args.dropout).cpu()

if args.audio_feat > 80:
    args.audio_feat = args.audio_feat// 3

transforms = torch.nn.Sequential(AudioPreprocessing(
                normalize='none', sample_rate=16000, window_size=window_size, 
                window_stride=window_stride, features=args.audio_feat, n_fft=512, log=True,
                feat_type='logfbank', trim_silence=True, window='hann',dither=0.00001, frame_splicing=1, transpose_out=False
            ), ConcatFeature(merge_size=3))


pytorch_total_params = sum(p.numel() for p in model.parameters())
print("Total parameters {:.3f}M".format(pytorch_total_params/1e6))

model.load_state_dict(checkpoint['model'])

model.eval()
for param in model.parameters():
    param.requires_grad = False

bos = torch.ones((1, 1)).long() * 1
h_pre, (h, c) = model.decoder(model.embed(bos))     # decode first zero
y_seq = []

encoder_h=None
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
    if status: # usually something bad
        print("X", flush=True, end=" ")
    else:
        indata = raw_indata.copy()

        # buffer.append(indata)
        # buffer = buffer[-2:]

        # indata = np.concatenate(buffer[-2:], axis=0)

        # print(indata.shape)
        # indata = indata / (1<<16)
        output = transforms(torch.from_numpy(indata.flatten()).float()).T#[ -1:, :]
        # output = output[-1:,:]
        # print(output.shape)

        # if encoder_h != None:
        #     print('start ',encoder_h[0][0][0][0][:10])
        h_enc, encoder_h = model.encoder(output.unsqueeze(0), hid=encoder_h)

        # print(len(encoder_h))
        # h_enc = h_enc[:, 1:, :]
        # print(h_enc.shape)
        for i in range(h_enc.shape[1]):
            # joint
            # print(h_pre[0, 0])
            logits = model.joint(h_enc[:, i], h_pre[:, 0])
            probs = F.log_softmax(logits, dim=1)
            prob, pred = torch.max(probs, dim=1)

            if pred.item() != model.blank:
                y_seq.append(pred)
                print(_tokenizer.decode([pred]), flush=True, end=" ")
            # else:
            #     print("_", flush=True, end=" ")


            # replace non blank entities with new state
            not_blank = pred.item() != model.blank
            if not_blank:
                embed_pred = model.embed(pred.unsqueeze(1))
                new_h_pre, (new_h, new_c) = model.decoder(embed_pred, (h, c))

                h_pre[not_blank, ...] = new_h_pre[not_blank, ...]
                h[:, not_blank, :] = new_h[:, not_blank, :]
                c[:, not_blank, :] = new_c[:, not_blank, :]

        # outdata[:] = raw_indata 

        # print(h_enc.shape)
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
    print(sr)
    transforms = torch.nn.Sequential(AudioPreprocessing(
                normalize='none', sample_rate=16000, window_size=window_size, 
                window_stride=window_stride, features=args.audio_feat, n_fft=512, log=True,
                feat_type='logfbank', trim_silence=True, window='hann',dither=0.00001, frame_splicing=1, transpose_out=False
            ), ConcatFeature(merge_size=3))
    output = transforms(data_).T#[ -1:, :]
    print(output.shape)
    y, nll = model.greedy_decode(output.unsqueeze(0),torch.from_numpy(np.array([len(output)])).int())
    hypothesis = _tokenizer.decode_plus(y)
    print(hypothesis)

def stream_wav(wav_file):
    import torchaudio
    frames = 8
    data, sr = torchaudio.load(wav_file, normalization=True)
    if sr != 16000:
        resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        data = resample(data)
        sr = 16000
    transforms = torch.nn.Sequential(AudioPreprocessing(
                normalize='none', sample_rate=16000, window_size=window_size, 
                window_stride=window_stride, features=args.audio_feat, n_fft=512, log=True,
                feat_type='logfbank', trim_silence=True, window='hann',dither=0.00001, frame_splicing=1, transpose_out=False
            ), ConcatFeature(merge_size=3))
    full_output = transforms(data.flatten()).T.squeeze(-1)

    # mfcc_cmvn = cmvnw(full_output.numpy(), win_size=frames*3, variance_normalization=False)
    # full_output = torch.from_numpy(mfcc_cmvn)
    # print(torch.mean(full_output), torch.var(full_output))
    # full_output = (full_output-torch.mean(full_output))/torch.var(full_output) 
    print('full_output mean and vars: ',torch.mean(full_output), torch.var(full_output))
    
    print('full_output ', full_output.shape)
    true_h_enc, _ = model.encoder(full_output.unsqueeze(0))
    print('true_h_enc ', true_h_enc.shape)

    data = data.T.numpy()
    # for audio_data in chunks(data, 599*frames + (frames-1)):
    #     callback(audio_data, np.zeros((audio_data.shape)), 0, 0, False)
    h_enc = []
    encoder_h = None
    mfcc = []
    buffer_size = int(16*1000 * window_size * 3 - 1)
    for audio_data in chunks(data, buffer_size*frames + (frames-1)):
        # print(len(audio_data.flatten()))
        output = transforms(torch.from_numpy(audio_data.flatten()).float()).T#[ -1:, :]
        # output = (output-torch.mean(output))/torch.var(output)
        # output = torch.from_numpy(cmvn(output.numpy(), variance_normalization=False))
        mfcc.append( output )
        # print(output.shape)
        h_enc_, encoder_h = model.encoder(output.unsqueeze(0), hid=encoder_h)
        # print(h_enc_.shape)
        h_enc.append(h_enc_[:, :, :])

    # print(h_enc[0].shape)
    print('segmented logfbank ',torch.cat(mfcc).shape)
    mfcc= torch.cat(mfcc)
    h_enc_2, encoder_h = model.encoder(mfcc.unsqueeze(0))
    print('chunk mfcc mean, var')
    print(torch.mean(mfcc), torch.var(mfcc))
    print('chunk mfcc vs full mfcc')
    print((mfcc-full_output)[-100:, ])
    print('mfcc vs full mfcc difference')
    print(torch.abs((mfcc-full_output)).max())

    h_enc = torch.cat(h_enc, dim=1)
    print('full-full vs segment-segment encoder diff')
    print(true_h_enc[:, :111, :] - h_enc[:, :111, :])
    print('segment-full vs segment-segment encoder fidd')
    print(h_enc_2[:, :111, :] - h_enc[:, :111, :])

    # h_enc = true_h_enc
    print('segment-segment')
    decode(h_enc)
    print('segment-full')
    decode(h_enc_2)
    print('full-full')
    decode(true_h_enc)

def decode(h_enc):
    y_seq = []
    log_p = []
    bos = torch.ones(1, 1).long() * 1
    h_pre, (h, c) = model.decoder(model.embed(bos))
    for i in range(h_enc.shape[1]):
        # joint
        logits = model.joint(h_enc[:, i], h_pre[:, 0])
        probs = F.log_softmax(logits, dim=1)
        prob, pred = torch.max(probs, dim=1)
        y_seq.append(pred)
        log_p.append(prob)
        embed_pred = model.embed(pred.unsqueeze(1))
        new_h_pre, (new_h, new_c) = model.decoder(embed_pred, (h, c))
        # replace non blank entities with new state
        h_pre[pred != model.blank, ...] = new_h_pre[pred != model.blank, ...]
        h[:, pred != model.blank, :] = new_h[:, pred != model.blank, :]
        c[:, pred != model.blank, :] = new_c[:, pred != model.blank, :]
    y_seq = torch.stack(y_seq, dim=1)
    y_seq = [list(filter(lambda tok: tok != model.blank, y_seq[0])) ]
    hypothesis = _tokenizer.decode_plus(y_seq)
    print(hypothesis)

q = queue.Queue()

def record_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

def record_and_decode():
    os.remove('demo.wav')
    try:
        with sf.SoundFile('demo.wav', mode='x', samplerate=16000,
                        channels=1, subtype='PCM_16') as file:
            with sd.InputStream(samplerate=16000, channels=1, callback=record_callback):
                print('#' * 80)
                print('press Ctrl+C to stop the recording')
                print('#' * 80)
                while True:
                    file.write(q.get())

    except KeyboardInterrupt:
        test_wav('demo.wav')

if __name__ == "__main__":

    print('start streaming')
    ''' 
        The floating point representations 'float32' 
        and 'float64' use +1.0 and -1.0 as 
        the maximum and minimum values
    '''
    frames = 8
    #with sd.Stream(channels=1,dtype='float32', samplerate=16000, 
    #    blocksize=599*frames+ (frames-1), callback=callback, 
    #    latency='high'):
    #    sd.sleep(duration * 1000)
    # record_and_decode()
    test_wav('3729-6852-0035.flac')
    stream_wav('3729-6852-0035.flac')
    # test_wav('test.mp3')
