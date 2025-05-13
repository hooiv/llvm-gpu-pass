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
#include <map>

namespace llvm {

/// Enumeration of supported GPU architectures
enum class GPUArch {
  UNKNOWN,
  NVIDIA_AMPERE,    // NVIDIA A100, A10, RTX 30xx series
  NVIDIA_TURING,    // NVIDIA RTX 20xx series, Quadro RTX
  NVIDIA_VOLTA,     // NVIDIA V100, Titan V
  NVIDIA_PASCAL,    // NVIDIA P100, GTX 10xx series
  NVIDIA_MAXWELL,   // NVIDIA M40, GTX 9xx series
  AMD_RDNA2,        // AMD RX 6000 series
  AMD_RDNA,         // AMD RX 5000 series 
  AMD_CDNA2,        // AMD Instinct MI200 series
  AMD_CDNA,         // AMD Instinct MI100
  AMD_VEGA,         // AMD Vega, Radeon VII
  INTEL_XE_HPC,     // Intel Ponte Vecchio
  INTEL_XE_HPG,     // Intel Arc A-series
  INTEL_XE_LP       // Intel integrated Xe graphics
};

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
  
  /// Calculate comprehensive cost for GPU offloading
  /// Returns a score between 0.0 (not suitable) and 1.0 (highly suitable)
  float calculateGPUOffloadingCost(Loop *L);
  
  /// Estimate memory transfer overhead between CPU and GPU
  float estimateDataTransferOverhead(Loop *L);
  
  /// Estimate the trip count of a loop as a factor in GPU offloading decisions
  const SCEV *estimateTripCount(Loop *L);
    /// Get architecture-specific cost factors for the current target GPU
  /// Returns a map of cost factors used in offloading decisions
  std::map<std::string, float> getGPUArchitectureCostFactors() const;
  
  /// Get the target GPU architecture based on compilation flags and target triple
  /// This is used to customize optimizations for specific GPU architectures
  GPUArch getTargetGPUArchitecture() const;

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
  
  /// Analyze memory access patterns for shared memory optimization opportunities
  /// Returns true if the loop can benefit from shared memory optimizations
  bool analyzeSharedMemoryOptimizationPotential(Loop *L);
  
  /// Get an estimate of the shared memory requirements for this loop
  /// Returns the amount of shared memory in bytes needed for optimizing this loop
  uint64_t estimateSharedMemoryRequirement(Loop *L);
  
  /// Check if an array access pattern is suitable for tiling in shared memory
  /// Analyzes access patterns to determine if they can benefit from tiling
  bool canApplySharedMemoryTiling(Value *ArrayBase, Loop *L);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_GPUOPTIMIZER_GPUPATTERNANALYZER_H
