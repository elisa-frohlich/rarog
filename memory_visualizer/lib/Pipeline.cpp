#include "Pipeline.h"
#include "MemoryVisualizer.h"

#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

void addMemoryVisualizerPipeline(OpPassManager &pm) {

    // --one-shot-bufferize="bufferize-function-boundaries"
    bufferization::OneShotBufferizePassOptions bufferizationOptions;
    bufferizationOptions.bufferizeFunctionBoundaries = true;
    pm.addPass(
        bufferization::createOneShotBufferizePass(bufferizationOptions)
    );

    // // --buffer-deallocation-pipeline
    // memref::ExpandReallocPassOptions expandAllocPassOptions{
    //   /*emitDeallocs=*/false};
    // pm.addPass(memref::createExpandReallocPass(expandAllocPassOptions));
    // pm.addPass(createCanonicalizerPass());

    // bufferization::OwnershipBasedBufferDeallocationPassOptions deallocationOptions;
    // deallocationOptions.privateFuncDynamicOwnership = true;

    // pm.addPass(
    //     bufferization::createOwnershipBasedBufferDeallocationPass(deallocationOptions)
    // );
    // pm.addPass(createCanonicalizerPass());
    // pm.addPass(bufferization::createBufferDeallocationSimplificationPass());
    // pm.addPass(bufferization::createLowerDeallocationsPass());
    // pm.addPass(createCSEPass());
    // pm.addPass(createCanonicalizerPass());

    // --memory-visualizer
    pm.addPass(rarog::createMemoryVisualizerPass());
}

} // namespace

namespace rarog {

void registerMemoryVisualizerPipeline() {
  PassPipelineRegistration<>("memory-visualizer", "Visualize memory allocation",
                             addMemoryVisualizerPipeline);
}

} // namespace rarog