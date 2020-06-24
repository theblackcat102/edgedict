#! /bin/bash

if [ -z "$1" ]
then
    echo "Usage: ./export_openvino.sh LOGDIR MODEL_NAME STEP_N_FRAME"
    echo "  e.g. ./export_openvino.sh ./logs/E6D2-smallbatch/ 45000.pt 2"
    exit 0
else
    LOGDIR=$1
fi

if [ -z "$2" ]
then
    MODEL_NAME="last.pt"
else
    MODEL_NAME=$2
fi

if [ -z "$3" ]
then
    STEP_N_FRAME="2"
else
    STEP_N_FRAME=$3
fi


python export_onnx.py \
    --flagfile ${LOGDIR}/flagfile.txt \
    --model_name ${MODEL_NAME} \
    --step_n_frame ${STEP_N_FRAME}

python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
    --framework onnx \
    --input_model ${LOGDIR}/encoder.onnx \
    --model_name encoder \
    --input "input[1 ${STEP_N_FRAME} 240],input_hidden[6 1 1024],input_cell[6 1 1024]" \
    --output_dir ${LOGDIR}

python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
    --framework onnx \
    --input_model ${LOGDIR}/decoder.onnx \
    --model_name decoder \
    --input "input[1 1]{i32},input_hidden[2 1 256],input_cell[2 1 256]" \
    --output_dir ${LOGDIR}

python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
    --framework onnx \
    --input_model ${LOGDIR}joint.onnx \
    --model_name joint \
    --input "input_h_enc[1 640],input_h_dec[1 256]" \
    --output_dir ${LOGDIR}