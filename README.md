# RNN-Transducer example

## TODO

- [x] Parallelize model training

- [x] Use BPE instead of character based tokenizer, should reduce more memory

- [x] Write checkpointing and tensorboardX logger

- [x] Modify wraprnnt-pytorch to compatible with apex mixed precision

## Install:

- Install apex
    https://nvidia.github.io/apex/amp.html

    ```
    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    ```

- Install warprnnt-pytorch
    https://github.com/HawkAaron/warp-transducer
    ```
    git clone https://github.com/HawkAaron/warp-transducer
    cd warp-transducer
    mkdir build
    cd build
    cmake -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME ..
    make
    cd ../pytorch_binding
    export CUDA_HOME="/usr/local/cuda"
    python setup.py install
    ```

## Datasets:

- Common Voice : 178.621 hrs

    ```
    mkdir common_voice
    cd common_voice
    wget https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/en.tar.gz
    tar -zxvf en.tar.gz
    ```

    i. Download english dataset from https://voice.mozilla.org/en/datasets

    ii. execute preprocess_common_voice.py to convert audio to 16k, PCM 16bits wav files ( this takes around 20 hours )

- Youtube Caption : 118 hrs


- Librispeech release 1 : 348 hrs

    ```
    python dataprep_librispeech.py [path/to/data] --download --extract
    ```

- TEDLIUM: 118.05 hrs

    Either download release 1 or 3 ( version 1 is smaller )

    SPHFile needed to extract audio segments based on timestamp from STM files

    ```
    wget http://www.openslr.org/resources/51/TEDLIUM_release-3.tgz
    wget http://www.openslr.org/resources/51/TEDLIUM_release-1.tgz
    pip install git+https://github.com/mcfletch/sphfile.git
    ```


## Data path
```
.
├──RNN-T/                   # this repo
├──common_voice/
│   ├──clips/               # all the audio
│   └──train.tsv
├──youtube-speech-text/
│   ├──english/             # all the audio
│   └──english_meta.csv
├──TEDLIUM_release1/
│   ├──train/
│   │   └──wav              # all the audio
│   └──test/
│       └──wav              # all the audio
└──LibriSpeech/
    ├──train-clean-360/
    └──test-clean/
```
