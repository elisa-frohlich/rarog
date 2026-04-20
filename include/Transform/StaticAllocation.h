#ifndef RAROG_INCLUDE_TRANSFORM_STATICALLOCATION_H
#define RAROG_INCLUDE_TRANSFORM_STATICALLOCATION_H

#include "mlir/Pass/Pass.h"

namespace rarog {

std::unique_ptr<mlir::Pass> createStaticAllocationPass(std::string resultFilename);

} // namespace rarog

#endif //RAROG_INCLUDE_TRANSFORM_STATICALLOCATION_H