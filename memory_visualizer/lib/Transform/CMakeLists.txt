add_mlir_library(RarogTransform
    Pipeline.cpp
    AddNasbenchMainFunction.cpp
    InstrumentMalloc.cpp

    LINK_LIBS PUBLIC
      MLIRIR
      MLIRPass
      MLIRAffineDialect
      MLIRArithDialect
      MLIRLinalgDialect
      MLIRBufferizationDialect
      MLIRFuncDialect
)

target_include_directories(RarogTransform
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/Transform
)