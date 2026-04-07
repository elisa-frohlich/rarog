#!/bin/bash

RAROG_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})/.." && pwd)"

if [ -f ${RAROG_ROOT}/.env ]
then
    source ${RAROG_ROOT}/.env
fi

RAROG_OPT_PATH="${RAROG_ROOT}/build/bin/rarog-opt"

MLIR_RUNNER=${MLIR_RUNNER:-mlir-runner}
MLIR_UTILS=${MLIR_UTILS:-/usr/lib/llvm/lib/libmlir_runner_utils.so}
MLIR_C_UTILS=${MLIR_C_UTILS:-/usr/lib/llvm/lib/libmlir_c_runner_utils.so}


# MODEL_PATH="${MODEL_PATH:-$RAROG_ROOT/onnx_models}"
MODEL_NAME="${MODEL_NAME:-model_1}"

if ! [ -f $RAROG_OPT_PATH ]
then
    # echo "rarog-opt is not compiled. Starting compilation process..."
    cd $RAROG_ROOT
    cmake -B build . --fresh
    cmake --build build
    if [[ $? != 0 ]]
    then
        echo "Compilation failed! Terminating..."
        exit 1
    fi
    cd -
fi

LINALG_MODEL="${RAROG_ROOT}/tmp/${MODEL_NAME}_linalg.mlir"
LOWERED_MODEL="${RAROG_ROOT}/tmp/${MODEL_NAME}_lowered.mlir"
HOISTED_MODEL="${RAROG_ROOT}/tmp/${MODEL_NAME}_hoisted.mlir"

if ! [ -f $LINALG_MODEL ]
then
    echo $LINALG_MODEL not found, running create_linalg.sh
    bash ${RAROG_ROOT}/scripts/create_linalg.sh
fi

# Apply lowering pipeline
$RAROG_OPT_PATH \
    --nasbench-lowering-pipeline \
    $LINALG_MODEL -o $LOWERED_MODEL

# Apply lowering pipeline with dealloc hoisting
$RAROG_OPT_PATH \
    --nasbench-lowering-pipeline="enable-reorder-frees" \
    $LINALG_MODEL -o $HOISTED_MODEL

$MLIR_RUNNER \
    $LOWERED_MODEL \
    --entry-point-result=void \
    --shared-libs=$MLIR_UTILS \
    --shared-libs=$MLIR_C_UTILS > ${RAROG_ROOT}/tmp/${MODEL_NAME}_output_lowered.txt

$MLIR_RUNNER \
    $HOISTED_MODEL \
    --entry-point-result=void \
    --shared-libs=$MLIR_UTILS \
    --shared-libs=$MLIR_C_UTILS > ${RAROG_ROOT}/tmp/${MODEL_NAME}_output_hoisted.txt

tail -n1 ${RAROG_ROOT}/tmp/${MODEL_NAME}_output_lowered.txt > ${RAROG_ROOT}/tmp/${MODEL_NAME}_output_lowered_tailed.txt
tail -n1 ${RAROG_ROOT}/tmp/${MODEL_NAME}_output_hoisted.txt > ${RAROG_ROOT}/tmp/${MODEL_NAME}_output_hoisted_tailed.txt

diff ${RAROG_ROOT}/tmp/${MODEL_NAME}_output_lowered_tailed.txt ${RAROG_ROOT}/tmp/${MODEL_NAME}_output_hoisted_tailed.txt