#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Block.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace rarog {

namespace {

  struct MemoryVisualizerPass : public PassWrapper<MemoryVisualizerPass, OperationPass<ModuleOp>> {
  
    void runOnOperation() override {
      ModuleOp module = getOperation();
  
      // Iterate over all functions in the module
      for (auto funcOp : module.getOps<func::FuncOp>()) {
        // Function name
        llvm::outs() << "Function: " << funcOp.getName() << "\n";
  
        for (auto allocOp : funcOp.getOps<memref::AllocOp>()) {
          llvm::outs() << "Found alloc instruction: " << allocOp << "\n";

          auto type = allocOp.getResult().getType();

          int64_t numElements = 1;
          for (auto dim : type.getShape()) {
            if (dim == ShapedType::kDynamic) dim = 1;
            numElements *= dim;
          }

          auto elemType = type.getElementType();
          auto typeSize = elemType.getIntOrFloatBitWidth() / 8;

          llvm::outs() << "\tSize: " << numElements*typeSize << "\n";

          llvm::outs() << "\t\tType: " << type << "\n";
          llvm::outs() << "\t\tElement Type: " << elemType << "\n";
          llvm::outs() << "\t\tNumber of Elements: " << numElements << "\n";
          llvm::outs() << "\t\tType size: " << typeSize << "\n";
        }
      }
    }
  };
} // namespace

std::unique_ptr<mlir::Pass> createMemoryVisualizerPass() {
  return std::make_unique<MemoryVisualizerPass>();
}

} // namespace rarog