#!/bin/bash

RAROG_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})/.." && pwd)"

if [ -f ${RAROG_ROOT}/.env ]
then
    source ${RAROG_ROOT}/.env
fi

MLIR_RUNNER=${MLIR_RUNNER:-mlir-runner}
MLIR_UTILS=${MLIR_UTILS:-/usr/lib/llvm/lib/libmlir_runner_utils.so}
MLIR_C_UTILS=${MLIR_C_UTILS:-/usr/lib/llvm/lib/libmlir_c_runner_utils.so}

MODEL_NAME="${MODEL_NAME:-model_1}"

LOWERED_MODEL="${RAROG_ROOT}/tmp/${MODEL_NAME}_lowered.mlir"

if ! [ -f $LOWERED_MODEL ]
then
    bash "${RAROG_ROOT}/scripts/lower.sh" &> /dev/null
fi

/usr/bin/time --format="\ntime elapsed: %es\nmax memory used: %Mkb\nCPU used: %P" $MLIR_RUNNER \
    $LOWERED_MODEL \
    --entry-point-result=void \
    --shared-libs=$MLIR_UTILS \
    --shared-libs=$MLIR_C_UTILS