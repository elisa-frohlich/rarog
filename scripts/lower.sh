#!/bin/bash

RAROG_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})/.." && pwd)"

RAROG_OPT_PATH="${RAROG_ROOT}/build/bin/rarog-opt"

if [ -z $MODEL_IDX ]
then
    MODEL_IDX=1
fi

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

LINALG_MODEL="${RAROG_ROOT}/tmp/model_${MODEL_IDX}_linalg.mlir"
LOWERED_MODEL="${RAROG_ROOT}/tmp/model_${MODEL_IDX}_lowered.mlir"

# Apply lowering pipeline
$RAROG_OPT_PATH \
    --nasbench-lowering-pipeline \
    $LINALG_MODEL -o $LOWERED_MODEL