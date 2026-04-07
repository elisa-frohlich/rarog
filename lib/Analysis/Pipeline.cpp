#include "Pipeline.h"
#include "MemoryAllocationInstantiation.h"
#include "ShufflingNumber.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

void addMemoryAllocationInstantiationPipeline(OpPassManager &pm) {

    // --one-shot-bufferize="bufferize-function-boundaries"
    bufferization::OneShotBufferizePassOptions bufferizationOptions;
    bufferizationOptions.bufferizeFunctionBoundaries = true;
    pm.addPass(bufferization::createOneShotBufferizePass(bufferizationOptions));

    // --memory-AllocationInstantiation
    pm.addPass(rarog::createMemoryAllocationInstantiationPass());
}

void addShufflingNumberPass(OpPassManager &pm) {
    pm.addPass(rarog::createShufflingNumberPass());
}

} // namespace

namespace rarog {

void registerMemoryAllocationInstantiationPipeline() {
    PassPipelineRegistration<>("memory-allocation-instantiation",
                               "Instantiate memory allocation problem",
                               addMemoryAllocationInstantiationPipeline);
}

void registerShufflingNumberPass() {
    PassPipelineRegistration<>("shuffling-number-pass",
                               "Calculate the shuffling number of a function",
                               addShufflingNumberPass);
}

} // namespace rarog