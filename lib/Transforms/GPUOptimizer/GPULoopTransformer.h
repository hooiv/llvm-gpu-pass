//===- GPULoopTransformer.h - Transform loops for GPU execution --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------===//
//
// This file defines the GPULoopTransformer class which transforms loops
// to make them suitable for GPU execution.
//
//===----------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_GPUOPTIMIZER_GPULOOPTRANSFORMER_H
#define LLVM_TRANSFORMS_GPUOPTIMIZER_GPULOOPTRANSFORMER_H

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <string>
#include <map>

namespace llvm {

/// Enumeration of supported GPU runtime APIs
enum class GPURuntime {
  CUDA,
  OpenCL,
  SYCL,
  HIP
};

/// Enumeration of common parallelization patterns
enum class ParallelizationPattern {
  MapPattern,            // Each iteration is independent
  ReducePattern,         // Reduction operation (sum, product, etc.)
  StencilPattern,        // Reads from neighboring elements
  TransposePattern,      // Matrix transposition
  ScanPattern,           // Parallel prefix sum (scan)
  HistogramPattern       // Histogram/binning operations
};

/// GPULoopTransformer - Transform loops for GPU execution
class GPULoopTransformer {
public:
  GPULoopTransformer(Module &M, LoopInfo &LI, ScalarEvolution &SE, GPURuntime Runtime)
    : M(M), LI(LI), SE(SE), Runtime(Runtime) {}

  /// Transform a loop for GPU execution
  /// Returns the newly created kernel function if successful
  Function *transformLoopToGPUKernel(Loop *L, ParallelizationPattern Pattern);

  /// Emit the runtime API calls to launch the kernel (CUDA/OpenCL/SYCL)
  bool insertKernelLaunchCode(Loop *L, Function *KernelFunc);

private:
  Module &M;
  LoopInfo &LI;
  ScalarEvolution &SE;
  GPURuntime Runtime;
  
  /// Replace the original loop with a kernel launch
  bool replaceLoopWithKernelLaunch(Loop *L, Function *KernelFunc);

  /// Create a GPU kernel function from a loop body
  Function *extractKernelFunction(Loop *L, ParallelizationPattern Pattern);

  /// Apply optimizations specific to the parallelization pattern
  void optimizeForPattern(Function *F, ParallelizationPattern Pattern);

  /// Add runtime-specific attributes to the kernel
  void addRuntimeAttributes(Function *F);

  /// Analyze and extract kernel parameters from the loop
  std::vector<Value*> extractKernelParameters(Loop *L);

  /// Generate thread indexing code for the kernel
  void insertThreadIndexing(Function *F, Loop *L, IRBuilder<> &Builder);

  /// Transform reduction patterns
  void transformReductionPattern(Function *F, Loop *L);

  /// Transform stencil patterns
  void transformStencilPattern(Function *F, Loop *L);

  /// Cache the loop trip count information
  std::map<const Loop*, const SCEV*> TripCountMap;
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_GPUOPTIMIZER_GPULOOPTRANSFORMER_H
