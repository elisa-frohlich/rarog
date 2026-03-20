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
        
        llvm::DenseMap<Operation*,int64_t> instructionMapping;
        int64_t idx = 0;
        Block &bodyBlock = funcOp.getBody().front();
        funcOp.walk([&](Operation *op) {
          llvm::outs() << idx << ": " << *op << "\n";
          instructionMapping[op] = idx++;
        });
  
        llvm::outs() << "\n\n";

        for (auto allocOp : funcOp.getOps<memref::AllocOp>()) {
          auto type = allocOp.getResult().getType();
          auto result = allocOp.getResult();
          
          llvm::outs() << "Found alloc instruction at idx: " << instructionMapping.at(result.getDefiningOp()) << "\n\n";

          // Size computation
          int64_t numElements = 1;
          for (auto dim : type.getShape()) {
            if (dim == ShapedType::kDynamic) dim = 1;
            numElements *= dim;
          }

          auto elemType = type.getElementType();
          auto typeSize = elemType.getIntOrFloatBitWidth() / 8;

          // Size information
          llvm::outs() << "\tSize: " << numElements*typeSize << "\n";
          
          llvm::outs() << "\t\tType: " << type << "\n";
          llvm::outs() << "\t\tElement Type: " << elemType << "\n";
          llvm::outs() << "\t\tNumber of Elements: " << numElements << "\n";
          llvm::outs() << "\t\tType size: " << typeSize << "\n\n";

          // Usage information
          llvm::outs() << "\tUsed at:\n";
          for (auto user : result.getUsers()) {
            llvm::outs() << instructionMapping.at(user) << ": ";
            llvm::outs() << *user << "\n";
          }
          llvm::outs() << "\n\n"; 
        }
      }
    }
  };
} // namespace

std::unique_ptr<mlir::Pass> createMemoryVisualizerPass() {
  return std::make_unique<MemoryVisualizerPass>();
}

} // namespace rarog