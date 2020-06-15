# Online Speech recognition using RNN-Transducer

Online speech recognition on Youtube Live video with ( 4 ~ 10 seconds faster than Youtube's caption )

## Install:

- Install `torch` and `torchaudio` with compatible version

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

- Install other dependencies
    ```
    pip install -r requirements.txt
    ```

## Datasets:

### Common Voice : 178.621 hrs

```
mkdir common_voice
cd common_voice
wget https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-4-2019-12-10/en.tar.gz
tar -zxvf en.tar.gz
```

- Download english dataset from https://voice.mozilla.org/en/datasets

- execute preprocess_common_voice.py to convert audio to 16k, PCM 16bits wav files ( this takes around 20 hours )

### Youtube Caption : 118 hrs


### Librispeech release 1 : 348 hrs

```
python dataprep_librispeech.py [path/to/data] --download --extract
```

### TEDLIUM: 118.05 hrs

- Either download release 1 or 3 ( version 1 is smaller )

```
wget http://www.openslr.org/resources/51/TEDLIUM_release-3.tgz
wget http://www.openslr.org/resources/51/TEDLIUM_release-1.tgz
pip install git+https://github.com/mcfletch/sphfile.git
```

### Data path
    ```
    └──RNN-T/                   # this repo
        ├──train.py
        ├──...
        └──datasets
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

## OpenVINO cheat sheet

- Export pytorch model to ONNX format
    ```
    python export_onnx.py \
        --flagfile ./logs/E6D2-smallbatch/flagfile.txt \
        --step 15000 \
        --step_n_frame 10
    ```

- Install OpenVINO inference engine Python API
    ```
    sudo -E apt update
    sudo -E apt -y install python3-pip python3-venv libgfortran3
    pip install -r /opt/intel/openvino/deployment_tools/model_optimizer/requirements.txt
    ```

- Model Optimizer
    - Setup envs
        ```
        source /opt/intel/openvino/bin/setupvars.sh
        ```
    - Encoder
        ```
        python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
            --framework onnx \
            --input_model ./logs/E6D2-smallbatch/encoder.onnx \
            --model_name encoder \
            --input "input[1 10 240],input_hidden[6 1 1024],input_cell[6 1 1024]" \
            --output_dir ./logs/E6D2-smallbatch/
        ```
    - Decoder
        ```
        python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
            --framework onnx \
            --input_model ./logs/E6D2-smallbatch/decoder.onnx \
            --model_name decoder \
            --input "input[1 1]{i32},input_hidden[2 1 256],input_cell[2 1 256]" \
            --output_dir ./logs/E6D2-smallbatch/
        ```
    - Joint
        ```
        python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
            --framework onnx \
            --input_model ./logs/E6D2-smallbatch/joint.onnx \
            --model_name joint \
            --input "input_h_enc[1 640],input_h_dec[1 256]" \
            --output_dir ./logs/E6D2-smallbatch/
        ```


## TODO

- [x] Parallelize model training

- [x] Use BPE instead of character based tokenizer, should reduce more memory

- [x] Write checkpointing and tensorboardX logger

- [x] Modify wraprnnt-pytorch to compatible with apex mixed precision
