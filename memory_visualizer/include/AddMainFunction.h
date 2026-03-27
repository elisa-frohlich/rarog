#ifndef RAROG_INCLUDE_ADDMAINFUNCTION_H
#define RAROG_INCLUDE_ADDMAINFUNCTION_H

#include "mlir/Pass/Pass.h"

namespace rarog {

std::unique_ptr<mlir::Pass> createAddMainFunctionPass();

} // namespace rarog

#endif //RAROG_INCLUDE_ADDMAINFUNCTION_H