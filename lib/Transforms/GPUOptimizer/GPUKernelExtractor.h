//===- GPUKernelExtractor.h - Extract kernels for GPU execution --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------===//
//
// This file defines the GPUKernelExtractor class which identifies and extracts
// code regions that are suitable for GPU execution.
//
//===----------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_GPUOPTIMIZER_GPUKERNELEXTRACTOR_H
#define LLVM_TRANSFORMS_GPUOPTIMIZER_GPUKERNELEXTRACTOR_H

#include "GPUPatternAnalyzer.h"
#include "GPULoopTransformer.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include <vector>
#include <map>

namespace llvm {

/// GPUKernelExtractor - Extract code regions suitable for GPU execution
class GPUKernelExtractor {
public:
  GPUKernelExtractor(Module &M, LoopInfo &LI, DependenceInfo &DI, ScalarEvolution &SE, 
                     GPURuntime Runtime = GPURuntime::CUDA)
    : M(M), LI(LI), DI(DI), SE(SE), Runtime(Runtime) {}

  /// Scan a function for GPU-suitable regions and extract them
  bool extractKernels(Function &F);

  /// Get the extracted kernels
  const std::vector<Function*>& getExtractedKernels() const { return ExtractedKernels; }

private:
  Module &M;
  LoopInfo &LI;
  DependenceInfo &DI;
  ScalarEvolution &SE;
  GPURuntime Runtime;
  std::vector<Function*> ExtractedKernels;
  
  /// Check if a loop is suitable for extraction
  bool isSuitableForExtraction(Loop *L);
  
  /// Extract a single loop into a GPU kernel
  Function* extractLoopToKernel(Loop *L);
  
  /// Extract a complex computational region into a GPU kernel
  Function* extractRegionToKernel(Function &F, const std::vector<BasicBlock*> &Region);
  
  /// Identify nested loop regions that are suitable for extraction
  std::vector<Loop*> identifySuitableLoops(Function &F);
  
  /// Identify complex computational regions (not just loops)
  std::vector<std::vector<BasicBlock*>> identifyComputationalRegions(Function &F);
  
  /// Analyze data dependencies between potential kernel regions
  void analyzeDependenciesBetweenRegions(std::vector<Loop*> &SuitableLoops);
  
  /// Determine the parallelization pattern for a loop
  ParallelizationPattern determineParallelizationPattern(Loop *L);
  
  /// Check if a loop has regular memory access patterns
  bool hasRegularMemoryAccess(Loop *L);
  
  /// Is this loop a good candidate for fusion with another loop?
  bool isCandidateForFusion(Loop *L);
  
  /// Fuse multiple loops into a single kernel if profitable
  Loop* fuseLoops(const std::vector<Loop*> &Loops);
  
  /// Score a loop for GPU extraction (higher is better)
  float scoreLoopForExtraction(Loop *L);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_GPUOPTIMIZER_GPUKERNELEXTRACTOR_H
