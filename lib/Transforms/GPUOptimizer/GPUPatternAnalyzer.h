//===- GPUPatternAnalyzer.h - Analyze patterns for GPU execution --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------===//
//
// This file defines the GPUPatternAnalyzer class which provides detailed
// analysis of code patterns suitable for GPU execution.
//
//===----------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_GPUOPTIMIZER_GPUPATTERNANALYZER_H
#define LLVM_TRANSFORMS_GPUOPTIMIZER_GPUPATTERNANALYZER_H

#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

namespace llvm {

/// GPUPatternAnalyzer - Analyzes loops and other code patterns to determine
/// their suitability for GPU execution
class GPUPatternAnalyzer {
public:
  GPUPatternAnalyzer(LoopInfo &LI, DependenceInfo &DI, ScalarEvolution &SE)
    : LI(LI), DI(DI), SE(SE) {}

  /// Analyze a loop for GPU execution potential
  bool isLoopSuitableForGPU(Loop *L);

  /// Check if a loop has regular memory access patterns
  bool hasRegularMemoryAccess(Loop *L);

  /// Analyze the data parallelism potential
  float calculateParallelizationPotential(Loop *L);

  /// Analyze memory bandwidth requirements
  bool hasEfficientMemoryBandwidth(Loop *L);

  /// Check if loop has sufficient computation density
  bool hasSufficientComputationDensity(Loop *L);

  /// Identify potential SIMD or SIMT parallelism
  bool identifySIMDPatterns(Loop *L);

  /// Classify potential for thread coarsening
  bool canApplyThreadCoarsening(Loop *L);

  /// Classify reduction patterns
  bool identifyReductionPatterns(Loop *L);

private:
  LoopInfo &LI;
  DependenceInfo &DI;
  ScalarEvolution &SE;

  /// Utility method to analyze memory operations
  void collectMemoryOperations(Loop *L, 
                              std::vector<LoadInst*> &Loads,
                              std::vector<StoreInst*> &Stores);

  /// Helper to analyze access patterns
  bool analyzeAccessPattern(Value *Ptr, Loop *L);

  /// Helper to identify reduction operations
  bool isReductionOperation(PHINode *Phi, Loop *L);

  /// Helper to check that an instruction is compatible with GPU execution
  bool isGPUCompatibleInstruction(Instruction *I);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_GPUOPTIMIZER_GPUPATTERNANALYZER_H
