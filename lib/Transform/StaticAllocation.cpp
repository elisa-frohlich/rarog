#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Block.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/IR/Type.h"
#include <fstream>
#include <unordered_set>

using namespace mlir;

namespace rarog {

namespace {

struct StaticAllocationPass : public PassWrapper<StaticAllocationPass, OperationPass<ModuleOp>> {

public:

  StaticAllocationPass(std::string resultFilename) : ResultFilename(resultFilename) {}

  void runOnOperation() override {
    if (!llvm::sys::fs::exists(ResultFilename)) {
      getOperation().emitError() << "Missing result file: " << ResultFilename;
      return signalPassFailure();
    }

    ModuleOp module = getOperation();

    LLVM::LLVMFuncOp targetFunc = nullptr;
    for (auto func : module.getOps<LLVM::LLVMFuncOp>()) {
      if (func.getName() == "torch_jit" || func.getName() == "tf2onnx") {
        targetFunc = func;
        break;
      }
    }

    if (!targetFunc) return;

    // Get the LLVM pointer type
    auto ptrType = LLVM::LLVMPointerType::get(module.getContext());

    // Get the LLVM void type
    auto voidType = LLVM::LLVMVoidType::get(module.getContext());

    // Declare rarog_malloc(ptr, i64)
    // ptr is the pointer to the big allocated buffer
    // i64 is the offset from the ptr to allocate the memory
    declareFunction(
      module,
      "rarog_malloc",
      {ptrType, IntegerType::get(module.getContext(), 64)},
      ptrType
    );

    // Declare rarog_free(ptr, ptr)
    // The first ptr is the pointer to the big allocated buffer
    // The second ptr is the pointer to the buffer being deallocated
    declareFunction(
      module,
      "rarog_free",
      {ptrType, ptrType},
      voidType
    );

    // Declare instrumented_malloc
  declareFunction(
    module,
    "instrumented_malloc",
    {IntegerType::get(module.getContext(), 64)},
    LLVM::LLVMPointerType::get(module.getContext())
  );

  // Declare instrumented_free
  declareFunction(
    module,
    "instrumented_free",
    {LLVM::LLVMPointerType::get(module.getContext())},
    LLVM::LLVMVoidType::get(module.getContext())
  );
    
    OpBuilder functionBuilder(module.getContext());
    Block &entryBlock = targetFunc.getBlocks().front();
    OpBuilder::InsertionGuard guard(functionBuilder);
    functionBuilder.setInsertionPointToStart(&entryBlock);

    Location loc = module.getLoc();

    // Compute the size of the allocated buffers for the model
    size_t neededSize = parseResultFile();
    Value mallocSize = functionBuilder.create<LLVM::ConstantOp>(loc, functionBuilder.getI64Type(), neededSize);

    // Create a call to malloc with size mallocSize
    Value mallocPtr = functionBuilder.create<LLVM::CallOp>(
      loc,
      ptrType,
      functionBuilder.getStringAttr("instrumented_malloc"),
      mallocSize
    ).getResult();

    size_t offset = 0;
    int curBuffer = 0;

    // modify calls in tf2onnx function
    targetFunc.walk([&](LLVM::CallOp callOp) {
      // We don't want to modify the first malloc call
      if (callOp.getResult() == mallocPtr) return;

      auto callee = callOp.getCallee();
      if (callee && *callee == "malloc") {
        // Find if the current malloc will be deallocated
        // If yes, change it to rarog_malloc
        // If not, it should not be deallocated and should be kept

        std::unordered_set<Operation *> visited;
        if (!isFreed(callOp.getResult(), visited)) return;

        OpBuilder builder(callOp);
        size_t curBufferSize = bufferSizes[curBuffer];

        auto cstSize = builder.create<LLVM::ConstantOp>(
          callOp.getLoc(),
          builder.getI64Type(),
          curBufferSize
        );

        // Add mallocPtr as first operand
        llvm::SmallVector<Value> operands = {mallocPtr, cstSize};

        // Create a call to the rarog_malloc function
        auto newCall = builder.create<LLVM::CallOp>(
          callOp.getLoc(),
          callOp.getResultTypes(),
          builder.getStringAttr("rarog_malloc"),
          operands
        );

        // Replace the previous uses of malloc for the rarog_malloc result
        callOp.replaceAllUsesWith(newCall.getResults());

        // Erase the previous malloc call
        callOp.erase();

        // Update offset and curBuffer
        offset += curBufferSize;
        ++curBuffer;
      } else if (callee && *callee == "free") {
        OpBuilder builder(callOp);

        // Add mallocPtr as first operand
        llvm::SmallVector<Value> operands = {mallocPtr};
        for (auto op : callOp.getOperands()) {
          operands.push_back(op);
        }

        // Create a call to the rarog_free function
        auto newCall = builder.create<LLVM::CallOp>(
          callOp.getLoc(),
          callOp.getResultTypes(),
          builder.getStringAttr("rarog_free"),
          operands
        );

        // Replace the previous uses of malloc for the rarog_free result
        // (probably useless)
        callOp.replaceAllUsesWith(newCall.getResults());

        // Erase the previous free call
        callOp.erase();
      }
    });

    // Create free call to dealloc created malloc before each return instruction
    // of the function
for (auto &block : targetFunc.getBlocks()) {
  auto *terminator = block.getTerminator();

  // Check if block terminator is a return instruction
  if (terminator && isa<LLVM::ReturnOp>(terminator)) {
    OpBuilder builder(terminator);

    builder.create<LLVM::CallOp>(
      loc,
      TypeRange{},
      functionBuilder.getStringAttr("instrumented_free"),
      mallocPtr
    );
  }
}
  }

private:
  std::string ResultFilename;
  llvm::SmallVector<size_t> bufferSizes;

  bool isFreed(Value pointer, std::unordered_set<Operation *> &visited) {
    for (Operation *user : pointer.getUsers()) {
      if (visited.count(user)) continue;
      visited.insert(user);

      if (auto call = dyn_cast<LLVM::CallOp>(user)) {
        auto callee = call.getCallee();
        // Check if user callee is a free and, if so, stop modifying the
        // current malloc call
        if (callee && *callee == "free") {
          return true;
        }
      } else if (auto bitcast = dyn_cast<LLVM::BitcastOp>(user)) {
        return isFreed(bitcast.getResult(), visited);
      } else if (auto gep = dyn_cast<LLVM::GEPOp>(user)) {
        return isFreed(gep.getResult(), visited);
      } else if (auto insertValue = dyn_cast<LLVM::InsertValueOp>(user)) {
        return isFreed(insertValue.getResult(), visited);
      } else if (auto extractValue = dyn_cast<LLVM::ExtractValueOp>(user)) {
        return isFreed(extractValue.getResult(), visited);
      }
    }

    return false;
  }

  void declareFunction(
    ModuleOp module,
    llvm::StringRef name,
    llvm::ArrayRef<Type> argTypes,
    Type retType
  ) {
    for (auto func : module.getOps<LLVM::LLVMFuncOp>()) {
      if (func.getName() == name) return;
    }

    OpBuilder builder(module.getBodyRegion());
    auto funcType = LLVM::LLVMFunctionType::get(retType, argTypes);
    builder.create<LLVM::LLVMFuncOp>(module.getLoc(), name, funcType);
  }

  int parseResultFile() {
    std::ifstream resultFile(ResultFilename);

    size_t totalSize = 0;
    std::string op;
    while (resultFile >> op) {
      if (op == "malloc") {
        std::string ptr;
        size_t size;
        resultFile >> ptr >> size;
        totalSize += size;
        bufferSizes.emplace_back(size);
      } else {
        std::string ptr;
        resultFile >> ptr;
      }
    }
    return totalSize;
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createStaticAllocationPass(std::string resultFilename) {
  return std::make_unique<StaticAllocationPass>(resultFilename);
}

} // namespace rarog