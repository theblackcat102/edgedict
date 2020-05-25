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

    i. Download english dataset from https://voice.mozilla.org/en/datasets

    ii. execute preprocess_common_voice.py to convert audio to 16k, PCM 16bits wav files

2. Youtube Caption 



## Install Apex

Note wrap rnn DO NOT support Half precision operation, so no Apex support

https://nvidia.github.io/apex/amp.html

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
