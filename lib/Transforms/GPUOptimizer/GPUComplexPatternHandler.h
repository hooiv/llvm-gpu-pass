//===- GPUComplexPatternHandler.h - Handle complex GPU patterns --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------===//
//
// This file defines the GPUComplexPatternHandler class which identifies and
// transforms complex GPU parallelization patterns.
//
//===----------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_GPUOPTIMIZER_GPUCOMPLEXPATTERNHANDLER_H
#define LLVM_TRANSFORMS_GPUOPTIMIZER_GPUCOMPLEXPATTERNHANDLER_H

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Support/Debug.h"
#include <vector>
#include <map>

namespace llvm {

/// Enumeration of complex GPU parallelization patterns
enum class ComplexParallelPattern {
  NestedParallelism,    // Nested parallel loops
  Pipeline,             // Pipeline pattern
  TaskParallel,         // Task parallelism
  StreamParallel,       // Stream parallelism
  Wavefront,            // Wavefront or diagonal pattern
  Tiling,               // 2D/3D tiling
  Recursive             // Recursive or divide-and-conquer
};

/// GPUComplexPatternHandler - Handle complex GPU parallelization patterns
class GPUComplexPatternHandler {
public:
  GPUComplexPatternHandler(Module &M, LoopInfo &LI)
    : M(M), LI(LI) {}

  /// Identify complex parallelization patterns in a function
  std::vector<std::pair<Loop*, ComplexParallelPattern>> identifyComplexPatterns(Function &F);

  /// Transform nested parallel loops
  bool transformNestedParallelism(Loop *OuterLoop, Loop *InnerLoop);

  /// Transform pipeline pattern
  bool transformPipelinePattern(const std::vector<BasicBlock*> &Pipeline);

  /// Transform task parallel pattern
  bool transformTaskParallelism(const std::vector<BasicBlock*> &Tasks);

  /// Transform stream parallel pattern
  bool transformStreamParallelism(const std::vector<Loop*> &Stages);

  /// Transform wavefront pattern
  bool transformWavefrontPattern(Loop *OuterLoop, Loop *InnerLoop);

  /// Apply 2D/3D tiling
  bool applyTiling(Loop *OuterLoop, Loop *InnerLoop, unsigned TileSize);

  /// Handle recursive or divide-and-conquer patterns
  bool handleRecursivePattern(Function &F);

private:
  Module &M;
  LoopInfo &LI;
  
  /// Check if two loops can be executed in nested parallel fashion
  bool canApplyNestedParallelism(Loop *OuterLoop, Loop *InnerLoop);
  
  /// Check if a sequence of blocks forms a pipeline
  bool isPipelinePattern(const std::vector<BasicBlock*> &Blocks);
  
  /// Check if a set of blocks can be executed as parallel tasks
  bool isTaskParallelPattern(const std::vector<BasicBlock*> &Blocks);
  
  /// Check if a sequence of loops forms a stream parallel pattern
  bool isStreamParallelPattern(const std::vector<Loop*> &Loops);
  
  /// Check if two loops form a wavefront pattern
  bool isWavefrontPattern(Loop *OuterLoop, Loop *InnerLoop);
  
  /// Check if a function has recursive or divide-and-conquer pattern
  bool isRecursivePattern(Function &F);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_GPUOPTIMIZER_GPUCOMPLEXPATTERNHANDLER_H
