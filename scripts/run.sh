#!/bin/bash

RAROG_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})/.." && pwd)"

# MLIR_RUNNER="/home/elisa/Apps/MLIR/bin/mlir-runner"
# MLIR_UTILS="/home/elisa/Apps/MLIR/lib/libmlir_runner_utils.so"
# MLIR_C_UTILS="/home/elisa/Apps/MLIR/lib/libmlir_c_runner_utils.so"

# if [ -z $MODEL_IDX ]
# then
#     MODEL_IDX=1
# fi

# LOWERED_MODEL="model_${MODEL_IDX}_lowered.mlir"

# time $MLIR_RUNNER \
#     $LOWERED_MODEL \
#     --entry-point-result=void \
#     --shared-libs=$MLIR_UTILS \
#     --shared-libs=$MLIR_C_UTILS