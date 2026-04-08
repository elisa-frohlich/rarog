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

      // op->getOpResult();
      for (Value rs : op->getOpResults()) {
        auto name = mlir::acc::getVariableName(rs);
        debug(name);
      }

      // ? Goal: For each operation, get the identifier that it defined
      // i.e. :
      // %c0 = arith.constant 0 : index
      // defines de identifier %c0

      /*
      for (auto operand : operands) {
          auto uses = operand.getNumUses();
          llvm::outs() << uses << " ";

          // TODO: Map out which VARIABLES are used, WHEN they are
          // defined.
          //
      https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/#traversing-the-def-use-chains
      }
      llvm::outs() << "]\n";

      for (auto result : op->getResults()) {
          // result.get
          for (auto user : result.getUsers()) {
              user->getName();
          }
          auto owner = result.getOwner();
      }
      */

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
