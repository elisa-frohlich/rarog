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

    // https://github.com/llvm/llvm-project/blob/main/mlir/test/lib/IR/TestPrintNesting.cpp
    // https://github.com/llvm/llvm-project/issues/56214
    // printAsOperand
    // https://github.com/llvm/llvm-project/blob/main/mlir/docs/LangRef.md#identifiers-and-keywords
    //
    // Looks useful...
    // mlir::acc::getVariableName(nullptr);
    // https://discourse.llvm.org/t/get-the-ssa-name-of-value/60025/10
    // https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/#traversing-the-def-use-chains

    llvm::outs() << "### Shuffling Number of " << fnName << "\n";

    fn.walk([&](Operation *op) -> WalkResult {
      // <results...> = <opName> <operands...>
      auto opName = op->getName();
      for (Value result : op->getResults()) {
        auto valuePortName = getValuePortName(result);
        llvm::outs() << valuePortName << " ";
      }
      llvm::outs() << "= " << opName << " ";
      for (Value operand : op->getOperands()) {
        auto valuePortName = getValuePortName(operand);
        llvm::outs() << valuePortName << " ";
      }
      llvm::outs() << "\n";

      // TODO: For operations
      // %c0 = arith.constant 0 : index
      // ==> (%c0,[])

      // %cst = arith.constant 0.000000e+00 : f32
      // ==> (%cst,[])

      // %dim = tensor.dim %arg0, %c0 : tensor<?x32x32x3xf32>
      // ==> (%dim,[%arg0,%c0])

      // %0 = tensor.empty(%dim) : tensor<?x3x32x32xf32>
      // ==> (%0, [%dim])

      // %1 = tensor.empty(%dim) : tensor<?x10xf32>
      // ==> (%1, [%dim])

      // %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<?x10xf32>) ->
      // tensor<?x10xf32>
      // ==> (%2, [%cst, %1])

      return WalkResult::advance();
    });

    llvm::outs() << "### END\n";
  }

private:
  // https://github.com/llvm/llvm-project/blob/42804379944cd0b221f9557ce219d4dc77a6055a/mlir/lib/Transforms/ViewOpGraph.cpp#L45-L50
  /// Return all values printed onto a stream as a string.
  static std::string strFromOs(function_ref<void(raw_ostream &)> func) {
    std::string buf;
    llvm::raw_string_ostream os(buf);
    func(os);
    return buf;
  }

  // https://github.com/llvm/llvm-project/blob/fbd0bcf55447c94dfa27bafb096f32e9c083f7ee/mlir/lib/Transforms/ViewOpGraph.cpp#L293-L302
  std::string getValuePortName(Value operand) {
    // Print value as an operand and omit the leading '%' character.
    auto str = strFromOs([&](raw_ostream &os) {
      operand.printAsOperand(os, OpPrintingFlags());
    });
    // ? Replace % and # with _
    // llvm::replace(str, '%', '_');
    // llvm::replace(str, '#', '_');
    return str;
  }
};

}; // namespace

std::unique_ptr<mlir::Pass> createShufflingNumberPass() {
  return std::make_unique<ShufflingNumberPass>();
}
}; // namespace rarog
