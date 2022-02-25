# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

set(WRAPPERS_SRCS
    "__init__.py"
    "commandline.py"
    "core.py"
    "datasets.py"
    "llvm.py"
    "time_limit.py"
    "validation.py")
if(MLIR_ENVIRONMENT)
  list(APPEND WRAPPERS_SRCS "mlir.py")
endif()

cg_py_library(
  NAME
    wrappers
  SRCS
    ${WRAPPERS_SRCS}
  DEPS
    compiler_gym::datasets::datasets
    compiler_gym::envs::envs
    compiler_gym::util::util
    compiler_gym::views::views
  PUBLIC
)
