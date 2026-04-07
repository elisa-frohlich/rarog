#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace rarog {

namespace {

struct ShufflingNumberPass
    : public PassWrapper<ShufflingNumberPass, OperationPass<ModuleOp>> {

    void runOnOperation() override {
        ModuleOp module = getOperation();

        llvm::outs() << "Hello World!" << "\n";
    }
};

}; // namespace

std::unique_ptr<mlir::Pass> createShufflingNumberPass() {
    return std::make_unique<ShufflingNumberPass>();
}

}; // namespace rarog