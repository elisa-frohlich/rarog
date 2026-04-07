#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#define debug(x) llvm::outs() << #x << " = " << x << "\n"

using namespace mlir;
using namespace func;

namespace rarog {

namespace {

struct ShufflingNumberPass
    : public PassWrapper<ShufflingNumberPass, OperationPass<FuncOp>> {

    void runOnOperation() override {
        FuncOp fn = getOperation();

        // TODO: Will work on any function, after this base case tf2onnx works
        if (fn.getName() != "tf2onnx") {
            return;
        }

        llvm::outs() << "Hello World!" << "\n";
    }
};

}; // namespace

std::unique_ptr<mlir::Pass> createShufflingNumberPass() {
    return std::make_unique<ShufflingNumberPass>();
}

}; // namespace rarog
