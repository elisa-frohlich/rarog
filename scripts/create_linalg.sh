#!/bin/bash

RAROG_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})/.." && pwd)"

PYTHON_VENV_PATH="${PYTHON_VENV_PATH:-$RAROG_ROOT/venv/bin/activate}"

source $PYTHON_VENV_PATH

NASBENCH_PATH="${RAROG_ROOT}/onnx_models"

if [ -z $MODEL_IDX ]
then
    MODEL_IDX=1
fi

echo "Running model_$MODEL_IDX"

mkdir -p tmp

ONNX_MODEL="${RAROG_ROOT}/onnx_models/model_${MODEL_IDX}.onnx"
MLIR_MODEL="${RAROG_ROOT}/tmp/model_${MODEL_IDX}.mlir"
LINALG_MODEL="${RAROG_ROOT}/tmp/model_${MODEL_IDX}_linalg.mlir"

# Convert ONNX model to MLIR (torch dialect)
torch-mlir-import-onnx $ONNX_MODEL -o $MLIR_MODEL

# Lower from torch to linalg dialect
torch-mlir-opt \
    --torch-onnx-to-torch-backend-pipeline \
    --torch-backend-to-linalg-on-tensors-backend-pipeline \
    $MLIR_MODEL -o $LINALG_MODEL