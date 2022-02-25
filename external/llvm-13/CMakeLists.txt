# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

project(gflags)
cmake_minimum_required(VERSION 3.15)

include(ExternalProject)

include(ProcessorCount)
if (DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL})
  set(_JOBS $ENV{CMAKE_BUILD_PARALLEL_LEVEL})
else()
  ProcessorCount(_JOBS)
endif()

# LLVM is memory hungry during linking
cmake_host_system_information(RESULT _RAM QUERY AVAILABLE_PHYSICAL_MEMORY)
set(_RAM_PER_LINK_JOB 10240) # 10 GiB
math(EXPR _LINK_JOBS "${_RAM} / ${_RAM_PER_LINK_JOB}" OUTPUT_FORMAT DECIMAL)
if (NOT _LINK_JOBS)
  set(_LINK_JOBS 1)
endif()

# The source is still required even if version is external.
if(NOT COMPILER_GYM_LLVM_PROVIDER STREQUAL "internal")
  set(_BUILD_COMMAND "")
  set(_INSTALL_COMMAND "")
else()
  set(_BUILD_COMMAND "${CMAKE_COMMAND}" --build "<BINARY_DIR>")
  set(_INSTALL_COMMAND "${CMAKE_COMMAND}" --install "<BINARY_DIR>")
endif()

file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/llvm_initial_cache.cmake"
  "set(LLVM_ENABLE_PROJECTS \"clang;polly;mlir\" CACHE STRING \"\" FORCE)\n")

ExternalProject_Add(
  llvm
  PREFIX "${CMAKE_CURRENT_BINARY_DIR}/llvm-13"
  GIT_REPOSITORY "https://github.com/llvm/llvm-project.git"
  GIT_TAG 7d176844d05bf11b32c4379f1422379d3d979a1e
  SOURCE_SUBDIR llvm
  CMAKE_ARGS
    -G Ninja
    -C "${CMAKE_CURRENT_BINARY_DIR}/llvm_initial_cache.cmake"
    "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}"
    -DLLVM_ABI_BREAKING_CHECKS=FORCE_OFF
    -DLLVM_PARALLEL_LINK_JOBS=${_LINK_JOBS}
    -DLLVM_PARALLEL_COMPILE_JOBS=${_JOBS}
    -DLLVM_TARGETS_TO_BUILD=host
  BUILD_COMMAND ${_BUILD_COMMAND}
  INSTALL_COMMAND ${_INSTALL_COMMAND}
)

# find_package(MLIR REQUIRED CONFIG)

# message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
# message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
# set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
# set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
# list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
# include(TableGen)
# include(AddLLVM)
# include(AddMLIR)
# include(HandleLLVMOptions)
# include(MLIRDetectPythonEnv)

# if(MLIR_ENABLE_BINDINGS_PYTHON)
#   include(MLIRDetectPythonEnv)
#   mlir_configure_python_dev_packages()
# endif()

# include_directories(${LLVM_INCLUDE_DIRS})
# include_directories(${MLIR_INCLUDE_DIRS})
# include_directories(${PROJECT_SOURCE_DIR}/include)
# include_directories(${PROJECT_BINARY_DIR}/include)
# link_directories(${LLVM_BUILD_LIBRARY_DIR})
# add_definitions(${LLVM_DEFINITIONS})

# # ExternalProject_Get_Property(llvm install_dir)
# # include_directories(${install_dir}/include)

# # # add_dependencies(project_b_exe project_a)
# # # target_link_libraries(project_b_exe ${install_dir}/lib/alib.lib)

# # # Reset CMAKE_BUILD_TYPE to its previous setting.
# # #set(CMAKE_BUILD_TYPE "${_CMAKE_BUILD_TYPE}" CACHE STRING "Build type (default ${DEFAULT_CMAKE_BUILD_TYPE})" FORCE)

# # # Extend module path to allow submodules to use LLVM and MLIR CMake modules.
# # # list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/mlir")
# # # list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_BINARY_DIR}/third_party/llvm-project/llvm/lib/cmake/llvm/")

# # # Add the bundled include directories for cmake files looking for them.
# # list(APPEND LLVM_INCLUDE_DIRS
# #   ${CMAKE_CURRENT_BINARY_DIR}/llvm/src/llvm/llvm/include
# # )
# # list(APPEND MLIR_INCLUDE_DIRS
# #   ${CMAKE_CURRENT_BINARY_DIR}/llvm/src/llvm/mlir/include
# #   ${CMAKE_CURRENT_BINARY_DIR}/llvm/src/llvm-build/tools/mlir/include
# # )

# # function(_hack_llvm_include_paths)
# #   set(_common_include_dirs
# #     $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/llvm/src/llvm/llvm/include>
# #     $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/llvm/src/llvm/mlir/include>
# #     $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/llvm/src/llvm-build/tools/mlir/include>
# #   )
# #   # Avoid globally modifying paths by instead adding the include paths to the
# #   # rules that really should have them in the first place.
# #   target_include_directories(LLVMSupport PUBLIC ${_common_include_dirs})
# #   target_include_directories(MLIRSupport PUBLIC ${_common_include_dirs})
# # endfunction()
# # #_hack_llvm_include_paths()