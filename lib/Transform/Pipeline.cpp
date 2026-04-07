#include "Pipeline.h"
#include "AddNasbenchMainFunction.h"
#include "InstrumentMalloc.h"
#include "ReorderFrees.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

struct NasbenchLoweringPipelineOptions : public PassPipelineOptions<NasbenchLoweringPipelineOptions> {
  Option<bool> enableReorderFrees{*this, "enable-reorder-frees",
                                  llvm::cl::desc("Enable reorder-frees pass"),
                                  llvm::cl::init(false)};
};

namespace {

void addNasbenchLoweringPipeline(OpPassManager &pm, const NasbenchLoweringPipelineOptions &options) {
  pm.addPass(rarog::createAddNasbenchMainFunctionPass());

  // --one-shot-bufferize="bufferize-function-boundaries"
  bufferization::OneShotBufferizePassOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  pm.addPass(
      bufferization::createOneShotBufferizePass(bufferizationOptions)
  );

  // --buffer-deallocation-pipeline
  // funcPM.addPass(bufferization::createBufferLoopHoistingPass());
  bufferization::BufferDeallocationPipelineOptions bufferDeallocOptions;
  mlir::bufferization::buildBufferDeallocationPipeline(pm, bufferDeallocOptions);
  // funcPM.addPass(mlir::bufferization::createOptimizeAllocationLivenessPass());
  // funcPM.addPass(mlir::createConvertBufferizationToMemRefPass());

  if (options.enableReorderFrees) {
    // Hoist memref.realloc instructions close to last use of deallocated buffer
    pm.addPass(rarog::createReorderFreesPass());
  
    pm.addPass(createCanonicalizerPass());
  
    pm.addPass(createCSEPass());
  }

  // --convert-linalg-to-loops
  pm.addPass(createConvertLinalgToLoopsPass());
  // --expand-strided-metadata
  pm.addPass(memref::createExpandStridedMetadataPass());
  // --lower-affine
  pm.addPass(createLowerAffinePass());
  // --convert-vector-to-llvm
  pm.addPass(createConvertVectorToLLVMPass());
  // --convert-math-to-llvm
  pm.addPass(createConvertMathToLLVMPass());
  // --convert-math-to-libm
  pm.addPass(createConvertMathToLibmPass());
  // --convert-scf-to-cf
  pm.addPass(createSCFToControlFlowPass());
  // --convert-arith-to-llvm
  pm.addPass(createArithToLLVMConversionPass());
  // --convert-func-to-llvm
  pm.addPass(createConvertFuncToLLVMPass());
  // --convert-cf-to-llvm
  pm.addPass(createConvertControlFlowToLLVMPass());
  // --finalize-memref-to-llvm
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  // --reconcile-unrealized-casts
  pm.addPass(createReconcileUnrealizedCastsPass());
}

void addInstrumentMallocPipeline(OpPassManager &pm) {
  pm.addPass(rarog::createInstrumentMallocPass());
}

void addReorderFreesPipeline(OpPassManager &pm) {
  pm.addPass(rarog::createReorderFreesPass());
}

} // namespace

namespace rarog {

void registerNasbenchLoweringPipeline() {
  PassPipelineRegistration<NasbenchLoweringPipelineOptions>(
    "nasbench-lowering-pipeline",
    "Insert main function and lower nasbench model to llvm",
    addNasbenchLoweringPipeline
  );
}

void registerInstrumentMallocPipeline() {
  PassPipelineRegistration<>(
    "instrument-malloc",
    "Change calls to malloc into calls to instrument_malloc",
    addInstrumentMallocPipeline
  );
}

void registerReorderFreesPipeline() {
  PassPipelineRegistration<>(
    "reorder-frees",
    "Hoist memref.realloc instructions close to last use of deallocated buffer",
    addReorderFreesPipeline
  );
}

} // namespace rarog

// We want to create a pass for deallocating buffers on ML models. This
// deallocation will insert memref.dealloc instructions after the last use of a
// buffer and remove the memref.dealloc instructions inserted by the
// buffer-deallocation-pipeline pass

// Notice: a buffer can be referenced by instructions like memref.view,
// memref.subview, memref.expand_shape, memref.collapse_shape and so on.

// Deepseek suggested using the BufferViewFlowAnalysis MLIR pass, but I need to
// verify if it inspects every instruction. Otherwise, it should be easy to
// implement a BFS that starts with the uses of the allocated buffer, and each
// time it finds one of the instructions above, it will add its uses to the
// queue, keeping track of the last used instruction

// After getting the last used instruction of the buffer and its aliases, we
// insert a memref.dealloc instruction after the last use. Then the pipeline
// runs canonicalization and CSE