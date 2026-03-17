#!/bin/bash

MLIR_OPT="/home/elisa/Apps/MLIR/bin/mlir-opt"

if [ -z $MODEL_IDX ]
then
    MODEL_IDX=1
fi

LINALG_MODEL="model_${MODEL_IDX}_linalg.mlir"
BUFFERED_MODEL="model_${MODEL_IDX}_buffer.mlir"
LOWERED_MODEL="model_${MODEL_IDX}_lowered.mlir"

# Apply bufferization passes
$MLIR_OPT \
    --one-shot-bufferize="bufferize-function-boundaries" \
    --buffer-deallocation-pipeline \
    $LINALG_MODEL -o $BUFFERED_MODEL

# Apply passes to lower to almost LLVM
$MLIR_OPT \
    --convert-linalg-to-loops \
    --expand-strided-metadata \
    --lower-affine \
    --convert-vector-to-llvm \
    --convert-math-to-llvm \
    --convert-math-to-libm \
    --convert-scf-to-cf \
    --convert-arith-to-llvm \
    --convert-func-to-llvm \
    --convert-cf-to-llvm \
    --finalize-memref-to-llvm \
    --reconcile-unrealized-casts \
    $BUFFERED_MODEL -o $LOWERED_MODEL