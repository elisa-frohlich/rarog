#include "Pipeline.h"
#include "MemoryVisualizer.h"

#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

void addMemoryVisualizerPipeline(OpPassManager &pm) {
    mlir::bufferization::OneShotBufferizePassOptions bufferizationOptions;
    bufferizationOptions.bufferizeFunctionBoundaries = true;
    pm.addPass(
        mlir::bufferization::createOneShotBufferizePass(bufferizationOptions)
    );
    pm.addPass(rarog::createMemoryVisualizerPass());
}

} // namespace

namespace rarog {

void registerMemoryVisualizerPipeline() {
  PassPipelineRegistration<>("memory-visualizer", "Visualize memory allocation",
                             addMemoryVisualizerPipeline);
}

} // namespace rarog