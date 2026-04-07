#!/bin/bash

RAROG_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})/.." && pwd)"
RAROG_OPT_PATH="${RAROG_ROOT}/build/bin/rarog-opt"
 
print_blue() {
    echo -e "\e[0;34m * ${1}\e[0m"
}
print_red() {
    echo -e "\e[0;31m ! ${1}\e[0m"
}

if [ -z $MODEL_IDX ]
then
    MODEL_IDX=1
fi

if ! [ -f $RAROG_OPT_PATH ]
then
    print_blue "rarog-opt is not compiled. Starting compilation process..."
    cd $RAROG_ROOT
    cmake -B build . --fresh
    cmake --build build
    if [[ $? != 0 ]]
    then
        print_red "Compilation failed! Terminating..."
        exit 1
    fi
    cd -
fi

print_blue "Calculating Shuffing Number for model_$MODEL_IDX"

mkdir -p tmp

BUFFER_MODEL="${RAROG_ROOT}/tmp/model_${MODEL_IDX}_buffer.mlir"
# ? This is just an analysis, so output is unchanged
OUTPUT_MODEL="${RAROG_ROOT}/tmp/model_${MODEL_IDX}_shuffling.mlir"

# Analyize with our tool
$RAROG_OPT_PATH \
    --shuffling-number-pass \
    $BUFFER_MODEL -o $OUTPUT_MODEL