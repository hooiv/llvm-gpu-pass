//===- GPUSyncPrimitives.h - Advanced GPU synchronization primitives --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------===//
//
// This file defines advanced synchronization primitives support for GPU programs.
// It includes analysis and transformation for barrier, warp-level synchronization,
// memory fences, atomic operations, and cooperative group operations.
//
//===----------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_GPUOPTIMIZER_GPUSYNCPRIMITIVES_H
#define LLVM_TRANSFORMS_GPUOPTIMIZER_GPUSYNCPRIMITIVES_H

#include "GPUPatternAnalyzer.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Analysis/LoopInfo.h"
#include <vector>
#include <set>

namespace llvm {

/// Types of GPU synchronization primitives
enum class GPUSyncType {
  BARRIER,                // Block-level synchronization
  WARP_SYNC,              // Warp-level synchronization (e.g., __syncwarp in CUDA)
  MEMORY_FENCE,           // Memory fence operations (e.g., __threadfence)
  ATOMIC_OPERATION,       // Atomic operations (e.g., atomicAdd)
  COOPERATIVE_GROUP,      // CUDA Cooperative Group synchronization
  GRID_SYNC               // Multi-block synchronization (e.g., grid.sync() in CUDA)
};

/// Represents a synchronization point in the code
struct SyncPoint {
  Instruction *Inst;      // The instruction representing the sync point
  GPUSyncType Type;       // Type of synchronization
  Value *Predicate;       // Optional predicate for conditional sync
  bool isConvergent;      // Whether this is a convergent operation

  SyncPoint(Instruction *I, GPUSyncType T, Value *P = nullptr, bool Conv = true) 
    : Inst(I), Type(T), Predicate(P), isConvergent(Conv) {}
};

/// This class analyzes GPU kernel functions to identify synchronization requirements
/// and inserts the appropriate synchronization primitives based on the target GPU
class GPUSynchronizationHandler {
public:
  GPUSynchronizationHandler(Function &F, LoopInfo &LI, GPUPatternAnalyzer &GPA)
    : F(F), LI(LI), GPA(GPA) {}

  /// Analyze the function to identify points where synchronization is needed
  void analyzeSynchronizationPoints();

  /// Insert appropriate synchronization primitives at the identified points
  bool insertSynchronizationPrimitives();

  /// Get all identified synchronization points
  const std::vector<SyncPoint> &getSyncPoints() const { return SyncPoints; }

  /// Check if a given block requires synchronization
  bool blockRequiresSynchronization(BasicBlock *BB) const;

  /// Determine if a memory operation requires synchronization
  bool memoryOperationNeedsSync(Instruction *I);

  /// Check if a loop can use more efficient synchronization patterns
  bool canOptimizeLoopSynchronization(Loop *L);

  /// Get the optimal grid size and synchronization strategy for a kernel
  void determineOptimalGridSyncStrategy(unsigned &BlocksX, unsigned &BlocksY, unsigned &BlocksZ);

  /// Insert host-side synchronization for multiple kernel launches
  void insertHostSideSynchronization(Module &M);

  /// Insert warp-level synchronization primitives
  bool insertWarpSynchronization();

  /// Insert block-level synchronization barriers
  bool insertBlockSynchronization();

  /// Handle CUDA Cooperative Group synchronization
  bool transformForCooperativeGroups();

  /// Transform atomic operations for better performance
  bool optimizeAtomicOperations();

private:
  Function &F;
  LoopInfo &LI;
  GPUPatternAnalyzer &GPA;
  std::vector<SyncPoint> SyncPoints;
  std::set<BasicBlock*> BlocksRequiringSync;

  /// Analyze data dependencies to determine sync requirements
  void analyzeDataDependencies();

  /// Identify places where memory fence operations are needed
  void identifyMemoryFencePoints();

  /// Analyze atomic operations for optimization
  void analyzeAtomicOperations();

  /// Create a new sync point and add it to the list
  void addSyncPoint(Instruction *I, GPUSyncType Type, Value *Predicate = nullptr);

  /// Transform kernel for grid-wide synchronization (cooperative groups)
  bool enableGridWideSynchronization();

  /// Find optimal insertion points for synchronization primitives
  Instruction *findOptimalSyncInsertionPoint(BasicBlock *BB);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_GPUOPTIMIZER_GPUSYNCPRIMITIVES_H
