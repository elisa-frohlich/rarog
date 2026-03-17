#!/bin/bash

RAROG_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})/.." && pwd)"

PYTHON_VENV_PATH="${PYTHON_VENV_PATH:-$RAROG_ROOT/venv/bin/activate}"
source $PYTHON_VENV_PATH

IREE_IMPORT_ONNX="${IREE_IMPORT_ONNX:-iree-import-onnx}"
IREE_COMPILE="${IREE_COMPILE:-iree-compile}"
IREE_RUN_MODULE="${IREE_RUN_MODULE:-iree-run-module}"


MODEL_IDX="${MODEL_IDX:-1}"

echo "Running model_$MODEL_IDX"

ONNX_MODEL="$RAROG_ROOT/onnx_models/model_${MODEL_IDX}.onnx"
MLIR_MODEL="$RAROG_ROOT/tmp/model_${MODEL_IDX}.mlir"
IREE_MODEL="$RAROG_ROOT/tmp/model_${MODEL_IDX}.vmfb"

mkdir -p "$RAROG_ROOT/tmp"

# Convert ONNX model to MLIR (torch dialect)
$IREE_IMPORT_ONNX $ONNX_MODEL -o $MLIR_MODEL

# Compile MLIR using IREE
$IREE_COMPILE \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    --iree-llvmcpu-target-cpu=host \
    --iree-opt-level=O2 \
    --iree-opt-data-tiling \
    $MLIR_MODEL -o $IREE_MODEL

# Run model through IREE
$IREE_RUN_MODULE \
    --device=local-task \
    --module=$IREE_MODEL \
    --function=tf2onnx \
    --input="1x32x32x3xf32=1"

rm -rf "$RAROG_ROOT/tmp"