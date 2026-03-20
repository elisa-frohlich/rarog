#ifndef RAROG_INCLUDE_MEMORYVISUALIZER_H
#define RAROG_INCLUDE_MEMORYVISUALIZER_H

#include "mlir/Pass/Pass.h"

namespace rarog {

std::unique_ptr<mlir::Pass> createMemoryVisualizerPass();

} // namespace rarog

#endif //RAROG_INCLUDE_MEMORYVISUALIZER_H