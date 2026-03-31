#ifndef RAROG_INCLUDE_TRANSFORM_ADDNASBENCHMAINFUNCTION_H
#define RAROG_INCLUDE_TRANSFORM_ADDNASBENCHMAINFUNCTION_H

#include "mlir/Pass/Pass.h"

namespace rarog {

std::unique_ptr<mlir::Pass> createAddNasbenchMainFunctionPass();

} // namespace rarog

#endif //RAROG_INCLUDE_TRANSFORM_ADDNASBENCHMAINFUNCTION_H