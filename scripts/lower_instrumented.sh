#!/bin/bash

RAROG_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})/.." && pwd)"

MEMORY_VISUALIZER="${RAROG_ROOT}/memory_visualizer"
RAROG_OPT_PATH="${MEMORY_VISUALIZER}/build/bin/rarog-opt"

if [ -z $MODEL_IDX ]
then
    MODEL_IDX=1
fi

if ! [ -f $RAROG_OPT_PATH ]
then
    # echo "rarog-opt is not compiled. Starting compilation process..."
    cd $MEMORY_VISUALIZER
    cmake -B build .
    cmake --build build
    if [[ $? != 0 ]]
    then
        echo "Compilation failed! Terminating..."
        exit 1
    fi
    cd -
fi

LINALG_MODEL="${RAROG_ROOT}/tmp/model_${MODEL_IDX}_linalg.mlir"
INSTRUMENTED_MODEL="${RAROG_ROOT}/tmp/model_${MODEL_IDX}_instrumented.mlir"

$RAROG_OPT_PATH \
    --nasbench-lowering-pipeline \
    --instrument-malloc \
    $LINALG_MODEL -o $INSTRUMENTED_MODEL