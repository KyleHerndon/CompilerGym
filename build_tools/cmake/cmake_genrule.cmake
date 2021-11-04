# Copied from https://github.com/google/iree/blob/main/build_tools/cmake/cmake_cc_library.cmake
# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include_guard(GLOBAL)

include(CMakeParseArguments)
include(cmake_macros)

# cmake_genrule()
#
# CMake function to imitate Bazel's genrule rule.
#
# Parameters:
# TODO(kyleherndon)
# NAME: name of target (see Note)
# SRCS: List of source files for the library
# OUTS: Output files generated by this rule
# CMD: Command to run
# PUBLIC: Add this so that this library will be exported under iree::
# Also in IDE, target will appear in IREE folder while non PUBLIC will be in IREE/internal.
# TESTONLY: When added, this target will only be built if user passes -DCMAKE_BUILD_TESTS=ON to CMake.
#
# Note:
# By default, cmake_genrule will always create a library named cmake_${NAME},
# and alias target iree::${NAME}. The iree:: form should always be used.
# This is to reduce namespace pollution.
#
function(cmake_genrule)
  cmake_parse_arguments(
    _RULE
    "PUBLIC;TESTONLY"
    "NAME"
    "SRCS;OUTS;CMD"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT CMAKE_BUILD_TESTS)
    return()
  endif()

  # TODO(boian): remove this renaming when call sites do not include ":" in target dependency names
  rename_bazel_targets("${_RULE_DEPS}" _RULE_DEPS)

  # Prefix the library with the package name, so we get: cmake_package_name.
  cmake_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  set(_SRCS ${_RULE_SRCS})
  # Symlink each file as its own target.
  foreach(SRC_FILE ${_SRCS})
    # SRC_FILE could have other path components in it, so we need to make a
    # directory for it. Ninja does this automatically, but make doesn't. See
    # https://github.com/google/iree/issues/6801
    set(_SRC_BIN_PATH "${CMAKE_CURRENT_BINARY_DIR}/${SRC_FILE}")
    get_filename_component(_SRC_BIN_DIR "${_SRC_BIN_PATH}" DIRECTORY)
    add_custom_command(
      OUTPUT "${_SRC_BIN_PATH}"
      COMMAND
        ${CMAKE_COMMAND} -E make_directory "${_SRC_BIN_DIR}"
      COMMAND ${CMAKE_COMMAND} -E create_symlink
        "${CMAKE_CURRENT_SOURCE_DIR}/${SRC_FILE}" "${_SRC_BIN_PATH}"
      DEPENDS "$${CMAKE_CURRENT_SOURCE_DIR}/{SRC_FILE}"
    )
    list(APPEND _OUTPUT_PATHS "${_SRC_BIN_PATH}")
  endforeach()

  set(_DEPS ${_RULE_DEPS} ${_OUTPUT_PATHS})
  add_custom_target(${_NAME} ALL DEPENDS ${_DEPS})

  add_custom_command(
      OUTPUT "${OUTS}"
      COMMAND "${CMD}"
      DEPENDS "${_SRC_BIN_PATH}"
  )

  # TODO(boian): add install rules

endfunction()
