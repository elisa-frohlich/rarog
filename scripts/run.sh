#!/bin/bash

RAROG_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})/.." && pwd)"

MLIR_RUNNER=${MLIR_RUNNER:-mlir-runner}
MLIR_UTILS=${MLIR_UTILS:-/usr/lib}
MLIR_C_UTILS=${MLIR_C_UTILS:-/usr/lib}

MODEL_IDX="${MODEL_IDX:-1}"

LOWERED_MODEL="${RAROG_ROOT}/tmp/model_${MODEL_IDX}_lowered.mlir"

time $MLIR_RUNNER \
    $LOWERED_MODEL \
    --entry-point-result=void \
    --shared-libs=$MLIR_UTILS \
    --shared-libs=$MLIR_C_UTILS