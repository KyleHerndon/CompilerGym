if(MLIR_ENVIRONMENT)

cg_add_all_subdirs()

cg_py_library(
  NAME
    mlir
  SRCS
    "__init__.py"
  DATA
    compiler_gym::envs::mlir::service::service
  DEPS
    ::mlir_env
  PUBLIC
)

cg_py_library(
  NAME
    mlir_env
  SRCS
    "mlir_env.py"
  DEPS
    compiler_gym::datasets::datasets
    compiler_gym::envs::compiler_env
    compiler_gym::spaces::spaces
  PUBLIC
)

endif()
