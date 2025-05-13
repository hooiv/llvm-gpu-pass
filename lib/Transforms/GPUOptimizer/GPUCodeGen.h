//===- GPUCodeGen.h - Generate GPU code from LLVM IR --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------===//
//
// This file defines the GPUCodeGen class which generates GPU code from LLVM IR.
//
//===----------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_GPUOPTIMIZER_GPUCODEGEN_H
#define LLVM_TRANSFORMS_GPUOPTIMIZER_GPUCODEGEN_H

#include "GPULoopTransformer.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Target/TargetMachine.h"
#include <string>

namespace llvm {

/// GPUCodeGen - Generate GPU code from LLVM IR
class GPUCodeGen {
public:
  GPUCodeGen(Module &M, GPURuntime Runtime) : M(M), Runtime(Runtime) {}

  /// Generate GPU kernel code for a function
  bool generateGPUCode(Function *F);

  /// Write generated GPU code to files
  bool writeToFile(StringRef Filename);
  
  /// Get the generated GPU code as a string
  std::string getGeneratedCode() const { return GeneratedCode; }

private:
  Module &M;
  GPURuntime Runtime;
  std::string GeneratedCode;

  /// Generate CUDA code
  bool generateCUDACode(Function *F);

  /// Generate OpenCL code
  bool generateOpenCLCode(Function *F);

  /// Generate SYCL code
  bool generateSYCLCode(Function *F);

  /// Generate HIP code
  bool generateHIPCode(Function *F);

  /// Update the module for GPU target
  void prepareModuleForGPU();

  /// Insert runtime API helper functions
  void insertRuntimeHelpers();

  /// Add memory allocation/deallocation code
  void addMemoryManagementCode(Function *F);

  /// Add memory transfer code (host to device, device to host)
  void addMemoryTransferCode(Function *F);

  /// Handle any special operations (atomics, barriers, etc.)
  void handleSpecialOperations(Function *F);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_GPUOPTIMIZER_GPUCODEGEN_H
