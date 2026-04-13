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

    fn.walk([&](Operation *op) -> WalkResult {
      // <results...> = <opName> <operands...>
      std::vector<std::string> resultNames, operandNames;

      auto opName = op->getName();
      for (Value result : op->getResults()) {
        auto valuePortName = getValuePortName(result);
        resultNames.push_back(valuePortName);
      }

      for (Value operand : op->getOperands()) {
        auto valuePortName = getValuePortName(operand);
        operandNames.push_back(valuePortName);
      }

      for (auto resultName : resultNames) {
        for (auto operandName : operandNames) {
          llvm::outs() << operandName << " -> " << resultName << "\n";

          // TODO: Create edges here
        }
      }

      llvm::outs() << "\n";

      return WalkResult::advance();
    });
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
