from urllib.request import urlopen
from datetime import datetime, timedelta
import subprocess
import av
import numpy as np
from stream import transforms, model, _tokenizer, test_wav, window_size, eval_args, lm_model
import torchaudio
import torch
import torch.nn.functional as F
import logging


hidden_ = lm_model.init_hidden(1)
lm_logist, lm_hidden = lm_model(torch.ones(1).long().unsqueeze(0), hidden_ )

av.logging.set_level(0)

frames = 4
single_input_chunk = int(16*1000 * window_size * 3 - 1)
buffer_size = single_input_chunk*frames+ (frames-1)
resampler = av.AudioResampler("s16p",rate=16*1000, layout=1)
buffers = []

bos = torch.ones((1, 1)).long() * 1
h_pre, (h, c) = model.decoder(model.embed(bos))     # decode first zero
y_seq = []

encoder_h=None
buffer = []

def reset_hidden_state():
    global buffer
    global encoder_h
    global h_enc
    encoder_h = None
    if len(buffer) > 0:
        output = transforms(buffer).T#[ -1:, :]
        h_enc, encoder_h = model.encoder(output.unsqueeze(0), hid=None)



def extract_stream_url():
    command = ['youtube-dl', '-f', '91', '-g', BLOOMBERG_LIVE]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**8)
    out, err = proc.communicate()
    stream_url = out.decode("utf-8").strip()
    return stream_url

def pyav_example(filepath, videolink, duration, output_stream=False, infinite=True):
    global buffers, y_seq
    global encoder_h
    global h, c
    global h_pre
    global lm_hidden, lm_logist

    if output_stream:
        output_container = av.open(filepath, 'w')
        output_stream = output_container.add_stream('mp3')
    begin = datetime.now()
    duration = timedelta(milliseconds=duration)
    
    timeout = timedelta(milliseconds=30*1000)
    track_cnt = 0


    if infinite:
        print('Loop forever')
    reconnect = False
    

    input_container = av.open(videolink)
    input_stream = input_container.streams.get(audio=0)[0]

    '''
    Start of decoding loop
    '''

    for frame in input_container.decode(input_stream):
        frame.pts = None
        resample_frame = resampler.resample(frame)

        waveform = torch.from_numpy(np.frombuffer(resample_frame.planes[0].to_bytes(), dtype='int16'))
        waveform = waveform.float() / 32768

        # waveform = waveform.clamp(-1, 1)
        # waveform[waveform != waveform] = 0
        if torch.isnan(waveform).any():
            print("X",flush=True, end=" ")
        # print(" ",flush=True, end=" ")
        if len(buffers) == 0:
            buffers = waveform
        elif len(buffers) < buffer_size:
            buffers = torch.cat([ buffers, waveform ], dim=0)

        # print(len(buffers), len(waveform))

        if len(buffers) >= buffer_size:
            # print('pop buffers')

            output_wave = buffers[:buffer_size]
            buffers = buffers[buffer_size:]
            # print(output.shape)
            # update_state(output)
            if torch.isnan(output_wave).any():
                print("X",flush=True, end=" ")
                continue
            # print('trans')
            output = transforms(output_wave).T#[ -1:, :]
            # print(output[:2, :5], output.shape)

            if torch.isnan(output).any():
                print("X",flush=True, end=" ")
                continue

            h_enc, encoder_h = model.encoder(output.unsqueeze(0), hid=encoder_h)
            # print('decode')
            for i in range(h_enc.shape[1]):
                logits = model.joint(h_enc[:, i], h_pre[:, 0])
                probs = F.log_softmax(logits, dim=1)
                prob, pred = torch.max(probs, dim=1)

                not_blank = pred.item() != model.blank
                if not_blank:
                    probs = lm_logist*0.1 + probs
                    prob, pred = torch.max(probs, dim=1)

                    lm_logist, lm_hidden = lm_model(pred.unsqueeze(1), lm_hidden )
                    track_cnt -= 1
                    y_seq.append(_tokenizer.token.id_to_token(pred.item()))
                    if len(y_seq) > 0 and '</w>' in y_seq[-1]:
                        print(''.join(y_seq).replace('</w>',''), flush=True, end=" ")
                        y_seq = []
                    # print(_tokenizer.token.id_to_token(pred.item()), flush=True, end=" ")

                if not_blank:
                    embed_pred = model.embed(pred.unsqueeze(1))
                    new_h_pre, (new_h, new_c) = model.decoder(embed_pred, (h, c))

                    h_pre[not_blank, ...] = new_h_pre[not_blank, ...]
                    h[:, not_blank, :] = new_h[:, not_blank, :]
                    c[:, not_blank, :] = new_c[:, not_blank, :]    
            # print('finish decode')
            # track_cnt += 1
            # if (track_cnt+1) % 200 == 0:
            #     print('[reset state]')
            #     reset_hidden_state()
            #     track_cnt = 0

        if output_stream:
            for packet in output_stream.encode(resample_frame):
                output_container.mux(packet)

        if (datetime.now() - begin) > duration and not infinite:
            print('exit')
            break
 
    '''
    End of decoding loop
    '''

    if output_stream:
        for packet in output_stream.encode(None):
            output_container.mux(packet)
        output_container.close()

if __name__ == "__main__":
    '''
        youtube-dl
        pip install av
    '''
    command = ['youtube-dl', '-f', '91', '-g', eval_args.url]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**8)
    out, err = proc.communicate()
    stream_url = out.decode("utf-8").strip()
    # print(stream_url)
    pyav_example('bloom.mp3', stream_url, 600*1000, 
        output_stream=False, infinite=True)
    # test_wav('bloom.mp3')
