#ifndef RAROG_INCLUDE_ANALYSIS_SHUFFLINGNUMBER_H
#define RAROG_INCLUDE_ANALYSIS_SHUFFLINGNUMBER_H

#include "mlir/Pass/Pass.h"

namespace rarog {

std::unique_ptr<mlir::Pass> createShufflingNumberPass(bool verbose);

}

#endif
