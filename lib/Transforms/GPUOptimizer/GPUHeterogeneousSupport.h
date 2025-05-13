//===- GPUHeterogeneousSupport.h - Support for heterogeneous execution --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------===//
//
// This file defines support for heterogeneous execution, enabling
// efficient execution of code on both CPU and GPU.
//
//===----------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_GPUOPTIMIZER_GPUHETEROGENEOUSSUPPORT_H
#define LLVM_TRANSFORMS_GPUOPTIMIZER_GPUHETEROGENEOUSSUPPORT_H

#include "GPUPatternAnalyzer.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include <vector>
#include <map>

namespace llvm {

/// Execution modes for heterogeneous code
enum class ExecutionMode {
  CPU_ONLY,           // Execute on CPU only
  GPU_ONLY,           // Execute on GPU only
  CPU_GPU_SPLIT,      // Split work between CPU and GPU
  CPU_GPU_REPLICATE,  // Replicate work on both CPU and GPU (for verification)
  ADAPTIVE            // Decide at runtime based on data size/availability
};

/// Represents a region of code that can be offloaded
struct OffloadableRegion {
  Loop *TheLoop;                // The loop to be offloaded (if it's a loop)
  BasicBlock *EntryBlock;       // Entry block of the region
  std::vector<BasicBlock*> Blocks; // All blocks in the region
  float GPUSuitability;         // Suitability score for GPU execution (0-1)
  float Speedup;                // Estimated speedup on GPU vs CPU
  size_t DataSize;              // Estimated data size to transfer
  bool HasSideEffects;          // Whether the region has side effects
  ExecutionMode PreferredMode;  // Preferred execution mode
  
  OffloadableRegion(Loop *L, float Suitability)
    : TheLoop(L), EntryBlock(L->getHeader()), GPUSuitability(Suitability),
      Speedup(1.0f), DataSize(0), HasSideEffects(false),
      PreferredMode(ExecutionMode::CPU_ONLY) {
    // Collect all blocks in the loop
    for (BasicBlock *BB : L->getBlocks()) {
      Blocks.push_back(BB);
    }
  }
  
  OffloadableRegion(BasicBlock *Entry, std::vector<BasicBlock*> AllBlocks, float Suitability)
    : TheLoop(nullptr), EntryBlock(Entry), Blocks(AllBlocks), 
      GPUSuitability(Suitability), Speedup(1.0f), DataSize(0), 
      HasSideEffects(false), PreferredMode(ExecutionMode::CPU_ONLY) {}
};

/// GPUHeterogeneousSupport - Provides support for heterogeneous execution
class GPUHeterogeneousSupport {
public:
  GPUHeterogeneousSupport(Module &M, LoopInfo &LI, DependenceInfo &DI, 
                        ScalarEvolution &SE, GPUPatternAnalyzer &GPA)
    : M(M), LI(LI), DI(DI), SE(SE), GPA(GPA) {}

  /// Identify regions suitable for offloading
  void identifyOffloadableRegions(Function &F);
  
  /// Get all identified offloadable regions
  const std::vector<OffloadableRegion> &getOffloadableRegions() const {
    return OffloadableRegions;
  }
  
  /// Create a CPU and GPU version of an offloadable region
  bool createHeterogeneousVersions(const OffloadableRegion &Region);
  
  /// Create a runtime dispatch mechanism to choose between CPU and GPU
  bool createRuntimeDispatch(const OffloadableRegion &Region);
  
  /// Analyze data dependencies for heterogeneous execution
  void analyzeDataDependencies(const OffloadableRegion &Region);
  
  /// Generate data transfer code between CPU and GPU
  bool generateDataTransferCode(const OffloadableRegion &Region);
  
  /// Determine the optimal execution mode for a region
  ExecutionMode determineOptimalExecutionMode(const OffloadableRegion &Region);
  
  /// Implement dynamic load balancing between CPU and GPU
  bool implementDynamicLoadBalancing(const OffloadableRegion &Region);
  
  /// Create task scheduling code for heterogeneous execution
  bool createTaskScheduler();
  
  /// Generate CPU fallback for when GPU is unavailable
  bool generateCPUFallback(const OffloadableRegion &Region);
  
  /// Create a heterogeneous execution pipeline for the whole module
  bool createHeterogeneousPipeline();
  
  /// Analyze memory access patterns for efficient data sharing
  void analyzeMemoryAccessPatterns(const OffloadableRegion &Region);
  
  /// Check if the code needs to be replicated for verification
  bool needsReplicationForVerification(const OffloadableRegion &Region);

private:
  Module &M;
  LoopInfo &LI;
  DependenceInfo &DI;
  ScalarEvolution &SE;
  GPUPatternAnalyzer &GPA;
  std::vector<OffloadableRegion> OffloadableRegions;
  
  /// Helper to extract a region into a separate function
  Function *extractRegionToFunction(const OffloadableRegion &Region, 
                                  const std::string &NameSuffix);
  
  /// Helper to create a GPU kernel from a CPU function
  Function *createGPUKernelFromFunction(Function *CPUFunc);
  
  /// Helper to create data movement code
  void createDataMovementCode(const OffloadableRegion &Region, 
                             Function *CPUFunc, 
                             Function *GPUKernel);
  
  /// Helper to create a runtime decision function
  Function *createRuntimeDecisionFunction(const OffloadableRegion &Region,
                                       Function *CPUFunc,
                                       Function *GPUKernel);
  
  /// Helper to analyze data shared between CPU and GPU
  std::map<Value*, size_t> analyzeSharedData(const OffloadableRegion &Region);
  
  /// Helper to estimate execution time on CPU and GPU
  void estimateExecutionTimes(OffloadableRegion &Region);
  
  /// Helper to create GPU kernel launch code
  void createGPULaunchCode(Function *Kernel, BasicBlock *InsertPoint,
                         const std::map<Value*, size_t> &SharedData);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_GPUOPTIMIZER_GPUHETEROGENEOUSSUPPORT_H
