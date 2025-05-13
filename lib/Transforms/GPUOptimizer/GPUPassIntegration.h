//===- GPUPassIntegration.h - Integration with other LLVM passes --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------===//
//
// This file defines interfaces for integrating GPU optimization passes
// with other standard LLVM optimization passes.
//
//===----------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_GPUOPTIMIZER_GPUPASSINTEGRATION_H
#define LLVM_TRANSFORMS_GPUOPTIMIZER_GPUPASSINTEGRATION_H

#include "GPUPatternAnalyzer.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include <vector>

namespace llvm {

/// GPUPassIntegration - Integrates GPU-specific optimizations with
/// standard LLVM optimization passes
class GPUPassIntegration {
public:
  GPUPassIntegration(Module &M) : M(M) {}

  /// Register dependencies with other passes
  void registerDependencies(PassRegistry &Registry);
  
  /// Integrate with loop optimization passes
  void integrateWithLoopPasses(LoopPassManager &LPM);
  
  /// Integrate with scalar optimization passes
  void integrateWithScalarPasses(FunctionPassManager &FPM);
  
  /// Integrate with vectorization passes
  void integrateWithVectorizationPasses(FunctionPassManager &FPM);
  
  /// Integrate with inlining passes
  void integrateWithInliningPasses(ModulePassManager &MPM);
  
  /// Set up pass pipeline for optimal GPU code generation
  void setupGPUPassPipeline(ModulePassManager &MPM);
  
  /// Check if there might be pass conflicts
  bool checkForPassConflicts(const Pass *P);
  
  /// Get list of recommended passes to run before GPU optimizations
  std::vector<std::string> getRecommendedPrePasses() const;
  
  /// Get list of recommended passes to run after GPU optimizations
  std::vector<std::string> getRecommendedPostPasses() const;
  
  /// Integrate with the Pass Builder infrastructure
  void registerWithPassBuilder(PassBuilder &PB);
  
  /// Query if a specific pass is compatible with GPU optimization
  bool isPassCompatibleWithGPU(StringRef PassName) const;

private:
  Module &M;
  
  /// Create a callback for the PassBuilder
  ModulePassManager createGPUOptimizationCallback(LoopAnalysisManager &LAM,
                                                 FunctionAnalysisManager &FAM,
                                                 CGSCCAnalysisManager &CGAM,
                                                 ModuleAnalysisManager &MAM);
  
  /// Register analysis dependencies
  void registerAnalysisDependencies(AnalysisManager<Module> &AM);
};

/// Class to perform GPU-aware loop optimizations 
class GPUAwareLoopOptimizer {
public:
  GPUAwareLoopOptimizer(LoopInfo &LI, ScalarEvolution &SE, DependenceInfo &DI,
                        GPUPatternAnalyzer &GPA)
    : LI(LI), SE(SE), DI(DI), GPA(GPA) {}

  /// Perform GPU-aware loop unrolling
  bool unrollForGPU(Loop *L);
  
  /// Perform GPU-aware loop distribution
  bool distributeForGPU(Loop *L);
  
  /// Perform GPU-aware loop fusion
  bool fuseLoopsForGPU(Loop *L1, Loop *L2);
  
  /// Analyze loop interchange opportunities for GPU
  bool canInterchangeForGPU(Loop *L);
  
  /// Perform GPU-aware loop interchange
  bool interchangeForGPU(Loop *L);
  
  /// Optimize loop for GPU coalesced memory access
  bool optimizeForCoalescedAccess(Loop *L);
  
  /// Analyze if the loop should have different optimizations for GPU vs CPU
  bool needsDifferentOptimizationForGPU(Loop *L);
  
  /// Check if loop vectorization is beneficial for GPU
  bool isVectorizationBeneficialForGPU(Loop *L);

private:
  LoopInfo &LI;
  ScalarEvolution &SE;
  DependenceInfo &DI;
  GPUPatternAnalyzer &GPA;
  
  /// Calculate unroll factor based on GPU architecture
  unsigned calculateGPUUnrollFactor(Loop *L);
  
  /// Calculate optimal tiling factors for GPU
  void calculateGPUTilingFactors(Loop *L, unsigned &X, unsigned &Y, unsigned &Z);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_GPUOPTIMIZER_GPUPASSINTEGRATION_H
