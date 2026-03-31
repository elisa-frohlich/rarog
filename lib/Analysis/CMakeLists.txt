add_mlir_library(RarogAnalysis
    Pipeline.cpp
    MemoryAllocationInstantiation.cpp

    LINK_LIBS PUBLIC
      MLIRIR
      MLIRPass
      MLIRAffineDialect
      MLIRArithDialect
      MLIRLinalgDialect
      MLIRBufferizationDialect
      MLIRFuncDialect
)

target_include_directories(RarogAnalysis
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/Analysis
)