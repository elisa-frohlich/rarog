#ifndef RAROG_INCLUDE_UTILS_FIRSTFITALLOCATION_H
#define RAROG_INCLUDE_UTILS_FIRSTFITALLOCATION_H

#include "llvm/ADT/SmallVector.h"

namespace rarog {

std::pair<llvm::SmallVector<size_t>, size_t> first_fit_allocation(llvm::SmallVector<std::tuple<size_t, size_t, size_t>> buffers);

} // namespace rarog

#endif //RAROG_INCLUDE_UTILS_FIRSTFITALLOCATION_H