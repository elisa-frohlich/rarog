#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenACC/OpenACCUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define debug(x) llvm::outs() << #x << " = " << x << " "

using namespace mlir;
using namespace func;

namespace rarog {

namespace {

struct ShufflingNumberPass
    : public PassWrapper<ShufflingNumberPass, OperationPass<FuncOp>> {

  void runOnOperation() override {
    FuncOp fn = getOperation();
    auto fnName = fn.getName();
    // https://github.com/llvm/llvm-project/blob/main/mlir/test/lib/IR/TestPrintNesting.cpp
    // https://github.com/llvm/llvm-project/issues/56214
    // printAsOperand

    // TODO: Will work on any function, after this base case tf2onnx works
    if (fnName != "tf2onnx") {
      return;
    }

    llvm::outs() << "### Shuffling Number of " << fnName << "\n";

    int op_count = 0;

    fn.walk([&](Operation *op) -> WalkResult {
      op_count++;
      std::string opName = op->getName().getStringRef().str();
      auto operands = op->getOperands();
      llvm::outs() << opName << " {" << operands.size() << "}[ ";

      // https://github.com/llvm/llvm-project/blob/main/mlir/docs/LangRef.md#identifiers-and-keywords

      // Looks useful...
      // mlir::acc::getVariableName(nullptr);

      // https://discourse.llvm.org/t/get-the-ssa-name-of-value/60025/10
      // https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/#traversing-the-def-use-chains

      return WalkResult::advance();
    });

    llvm::outs() << "There were " << op_count << " operands\n";
  }
};

}; // namespace

std::unique_ptr<mlir::Pass> createShufflingNumberPass() {
  return std::make_unique<ShufflingNumberPass>();
}
}; // namespace rarog
