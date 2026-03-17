#!/bin/bash

RAROG_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})/.." && pwd)"

PYTHON_VENV_PATH="/home/elisa/general_venv/bin/activate"
NASBENCH_PATH="/home/elisa/Codes/nasbench/onnx"

source $PYTHON_VENV_PATH

if [ -z $MODEL_IDX ]
then
    MODEL_IDX=1
fi

echo "Running model_$MODEL_IDX"

ONNX_MODEL="model_${MODEL_IDX}.onnx"
MLIR_MODEL="model_${MODEL_IDX}.mlir"
LINALG_MODEL="model_${MODEL_IDX}_linalg.mlir"

# Convert ONNX model to MLIR (torch dialect)
torch-mlir-import-onnx $ONNX_MODEL -o $MLIR_MODEL

# Lower from torch to linalg dialect
torch-mlir-opt \
    --torch-onnx-to-torch-backend-pipeline \
    --torch-backend-to-linalg-on-tensors-backend-pipeline \
    $MLIR_MODEL -o $LINALG_MODEL