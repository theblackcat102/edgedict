# RNN-Transducer example

### Install:

Make sure warprnnt pytorch is installed

Execute models.py to check if everything is properly installed

```
python models.py
```

### Datasets:

0. VoxCeleb

    i. Apply for username and password from VoxCeleb

    ii. Download corpus using this repo: https://github.com/clovaai/voxceleb_trainer

1. Common Voice

'''
mkdir common_voice
cd common_voice
wget https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/en.tar.gz
tar -zxvf en.tar.gz
'''

    i. Download english dataset from https://voice.mozilla.org/en/datasets

    ii. execute preprocess_common_voice.py to convert audio to 16k, PCM 16bits wav files ( this takes around 20 hours )

2. Youtube Caption 


3. Librispeech release 1

```
pip install git+https://github.com/mcfletch/sphfile.git

```


## Data path
```

RNN-T/ -> github repo
common_voice/
    clips
        - all the audio
    train.tsv
youtube-speech-text/
    english/
        - all the audio
    english_meta.csv
```


## Install Apex

Note wrap rnn DO NOT support Half precision operation, so no Apex support

https://nvidia.github.io/apex/amp.html

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```



