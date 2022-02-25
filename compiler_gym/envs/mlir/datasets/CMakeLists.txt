cg_add_all_subdirs()

cg_py_library(
  NAME
    datasets
  SRCS
    "__init__.py"
    "matmul.py"
  DATA
    compiler_gym::third_party::csmith::all
  DEPS
    compiler_gym::datasets::datasets
    compiler_gym::service::proto::proto
    compiler_gym::third_party::llvm::llvm
    compiler_gym::util::util
  PUBLIC
)
