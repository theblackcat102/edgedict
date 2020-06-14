#! /bin/bash

if [ -z "$1" ]
then
    echo "Usage: ./export_openvino.sh LOGDIR STEP"
    echo "  e.g. ./export_openvino.sh ./logs/E6D2-smallbatch/ 45000"
    exit 0
else
    LOGDIR=$1
fi

if [ -z "$2" ]
then
    STEP="last.py"
else
    STEP=$2
fi


python export_onnx.py \
    --flagfile ${LOGDIR}/flagfile.txt \
    --step ${STEP} \
    --step_n_frame 10

python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
    --framework onnx \
    --input_model ${LOGDIR}/encoder.onnx \
    --model_name encoder \
    --input "input[1 10 240],input_hidden[6 1 1024],input_cell[6 1 1024]" \
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