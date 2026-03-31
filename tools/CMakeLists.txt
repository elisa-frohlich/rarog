set(LLVM_LINK_COMPONENTS Support)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(rarog-opt
  rarog-opt.cpp
)

llvm_update_compile_flags(rarog-opt)

target_link_libraries(rarog-opt
  PRIVATE

  RarogAnalysis
  RarogTransform

  # MLIR dependencies
  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  ${dialect_libs}
  ${conversion_libs}
)

target_include_directories(rarog-opt
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

mlir_check_link_libraries(rarog-opt)