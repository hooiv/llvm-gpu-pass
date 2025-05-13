cmake_minimum_required(VERSION 3.13.4)
project(LLVMGPUOptimizer)

# When built as an LLVM external project, LLVM's CMake infrastructure is already available.
# The find_package call below is typically for standalone builds of the pass.
# find_package(LLVM REQUIRED CONFIG)
# message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
# message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# LLVM settings - these should be inherited or handled by LLVM targets when built externally.
# include_directories(${LLVM_INCLUDE_DIRS})
# add_definitions(${LLVM_DEFINITIONS})
# link_directories(${LLVM_LIBRARY_DIRS})

# Add path for custom LLVM CMake modules (e.g., AddLLVM.cmake for add_llvm_library)
# LLVM_CMAKE_DIR should be set by the parent LLVM build.
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

add_subdirectory(lib)
