add_mlir_library(Rarog
    Pipeline.cpp
    MemoryVisualizer.cpp

    LINK_LIBS PUBLIC
      MLIRIR
      MLIRPass
      MLIRAffineDialect
      MLIRArithDialect
      MLIRLinalgDialect
      MLIRBufferizationDialect
      MLIRFuncDialect
)

target_include_directories(Rarog
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)