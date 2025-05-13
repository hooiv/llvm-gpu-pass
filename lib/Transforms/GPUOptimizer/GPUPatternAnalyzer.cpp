//===- GPUPatternAnalyzer.cpp - Analyze patterns for GPU execution --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------===//
//
// This file implements the GPUPatternAnalyzer class which provides detailed
// analysis of code patterns suitable for GPU execution.
//
//===----------------------------------------------------------------===//

#include "GPUPatternAnalyzer.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "gpu-pattern-analyzer"

bool GPUPatternAnalyzer::isLoopSuitableForGPU(Loop *L) {
  // Complex analysis would go here
  // For now, just a basic implementation to start with
  
  // Check the loop has appropriate trip count
  if (!hasSufficientComputationDensity(L))
    return false;
    
  // Check memory access patterns
  if (!hasRegularMemoryAccess(L))
    return false;
    
  // Check reduction patterns
  if (identifyReductionPatterns(L)) {
    // Special handling for reduction patterns
    // They can be efficient on GPUs with proper transformations
    LLVM_DEBUG(dbgs() << "Loop has reduction patterns, may need special handling\n");
  }
  
  // Simple coarse pass - check if we have any loop dependencies
  std::vector<LoadInst*> Loads;
  std::vector<StoreInst*> Stores;
  collectMemoryOperations(L, Loads, Stores);
  
  for (auto *Store : Stores) {
    for (auto *Load : Loads) {
      auto Result = DI.depends(Store, Load, true);
      if (Result && Result->isOrdered())
        return false;
    }
    for (auto *OtherStore : Stores) {
      if (Store == OtherStore)
        continue;
      auto Result = DI.depends(Store, OtherStore, true);
      if (Result && Result->isOrdered())
        return false;
    }
  }
  
  return true;
}

bool GPUPatternAnalyzer::hasRegularMemoryAccess(Loop *L) {
  std::vector<LoadInst*> Loads;
  std::vector<StoreInst*> Stores;
  collectMemoryOperations(L, Loads, Stores);
  
  // Check for stride-1 access patterns, which are ideal for GPU
  for (auto *Load : Loads) {
    if (!analyzeAccessPattern(Load->getPointerOperand(), L))
      return false;
  }
  
  for (auto *Store : Stores) {
    if (!analyzeAccessPattern(Store->getPointerOperand(), L))
      return false;
  }
  
  return true;
}

float GPUPatternAnalyzer::calculateParallelizationPotential(Loop *L) {
  // Count instructions that can be parallelized
  unsigned TotalInsts = 0;
  unsigned ParallelizableInsts = 0;
  
  for (auto *BB : L->getBlocks()) {
    for (auto &I : *BB) {
      TotalInsts++;
      if (isGPUCompatibleInstruction(&I))
        ParallelizableInsts++;
    }
  }
  
  if (TotalInsts == 0)
    return 0.0f;
    
  return static_cast<float>(ParallelizableInsts) / TotalInsts;
}

bool GPUPatternAnalyzer::hasEfficientMemoryBandwidth(Loop *L) {
  // Placeholder for bandwidth analysis
  // In a real implementation, we would:
  // 1. Calculate memory operations per work item
  // 2. Look at coalescing potential
  // 3. Analyze cache behavior
  
  return true;  // Simplified for now
}

bool GPUPatternAnalyzer::hasSufficientComputationDensity(Loop *L) {
  // Calculate arithmetic intensity (ratio of compute to memory operations)
  unsigned ComputeOps = 0;
  unsigned MemoryOps = 0;
  
  for (auto *BB : L->getBlocks()) {
    for (auto &I : *BB) {
      if (isa<LoadInst>(I) || isa<StoreInst>(I))
        MemoryOps++;
      else if (I.getOpcode() >= Instruction::Add && I.getOpcode() <= Instruction::Xor)
        ComputeOps++;
    }
  }
  
  if (MemoryOps == 0)
    return false;  // No memory operations, likely not a good GPU candidate
    
  float ArithmeticIntensity = static_cast<float>(ComputeOps) / MemoryOps;
  
  // Simple heuristic: If we have more compute than memory ops, good candidate
  return ArithmeticIntensity >= 1.0f;
}

bool GPUPatternAnalyzer::identifySIMDPatterns(Loop *L) {
  // Look for patterns where the same operation is applied to different data elements
  // This is a simplified implementation
  
  BasicBlock *Body = L->getLoopLatch();
  if (!Body)
    return false;
    
  // Look for basic arithmetic operations that can benefit from SIMD
  bool HasSIMDOps = false;
  for (auto &I : *Body) {
    if (I.getOpcode() >= Instruction::Add && I.getOpcode() <= Instruction::FRem)
      HasSIMDOps = true;
  }
  
  return HasSIMDOps;
}

bool GPUPatternAnalyzer::canApplyThreadCoarsening(Loop *L) {
  // Thread coarsening is beneficial when each thread would do very little work
  // This is a placeholder for actual analysis
  
  return calculateParallelizationPotential(L) > 0.8f;  // High parallelization potential
}

bool GPUPatternAnalyzer::identifyReductionPatterns(Loop *L) {
  BasicBlock *Header = L->getHeader();
  for (auto &I : *Header) {
    if (auto *Phi = dyn_cast<PHINode>(&I)) {
      if (isReductionOperation(Phi, L))
        return true;
    }
  }
  
  return false;
}

void GPUPatternAnalyzer::collectMemoryOperations(Loop *L, 
                                               std::vector<LoadInst*> &Loads,
                                               std::vector<StoreInst*> &Stores) {
  for (auto *BB : L->getBlocks()) {
    for (auto &I : *BB) {
      if (auto *Load = dyn_cast<LoadInst>(&I))
        Loads.push_back(Load);
      else if (auto *Store = dyn_cast<StoreInst>(&I))
        Stores.push_back(Store);
    }
  }
}

bool GPUPatternAnalyzer::analyzeAccessPattern(Value *Ptr, Loop *L) {
  // Use scalar evolution to determine if this is a strided access
  const SCEV *AccessFunction = SE.getSCEV(Ptr);
  
  // Look for affine expressions that represent good access patterns
  if (auto *AddRec = dyn_cast<SCEVAddRecExpr>(AccessFunction)) {
    // Check if this is an add recurrence in this loop
    if (AddRec->getLoop() == L) {
      // Check for constant stride
      const SCEV *Stride = AddRec->getStepRecurrence(SE);
      if (isa<SCEVConstant>(Stride))
        return true;
    }
  }
  
  // For anything else, we'll assume it's not a regular access pattern
  // A more sophisticated implementation would do deeper analysis
  return false;
}

bool GPUPatternAnalyzer::isReductionOperation(PHINode *Phi, Loop *L) {
  // Basic analysis for reductions like sum, product, min, max
  if (Phi->getNumIncomingValues() != 2)
    return false;
    
  BasicBlock *Header = L->getHeader();
  BasicBlock *Latch = L->getLoopLatch();
  
  // Ensure phi takes values from header and latch
  if (!(Phi->getIncomingBlock(0) == Header && Phi->getIncomingBlock(1) == Latch) &&
      !(Phi->getIncomingBlock(0) == Latch && Phi->getIncomingBlock(1) == Header))
    return false;
    
  // Find the latch value
  Value *LatchVal = nullptr;
  for (unsigned i = 0; i < Phi->getNumIncomingValues(); ++i) {
    if (Phi->getIncomingBlock(i) == Latch) {
      LatchVal = Phi->getIncomingValue(i);
      break;
    }
  }
  
  if (!LatchVal)
    return false;
    
  // Check if latch value is computed using an operation on the phi
  if (auto *BinOp = dyn_cast<BinaryOperator>(LatchVal)) {
    if (BinOp->getOpcode() == Instruction::Add ||
        BinOp->getOpcode() == Instruction::FAdd ||
        BinOp->getOpcode() == Instruction::Mul ||
        BinOp->getOpcode() == Instruction::FMul ||
        BinOp->getOpcode() == Instruction::And ||
        BinOp->getOpcode() == Instruction::Or ||
        BinOp->getOpcode() == Instruction::Xor) {
      // Check if one operand is the phi
      return BinOp->getOperand(0) == Phi || BinOp->getOperand(1) == Phi;
    }
  }
  
  return false;
}

bool GPUPatternAnalyzer::isGPUCompatibleInstruction(Instruction *I) {
  // Most arithmetic operations are compatible
  if (I->getOpcode() >= Instruction::Add && I->getOpcode() <= Instruction::Xor)
    return true;
    
  // Memory operations are compatible if they have regular access patterns
  if (auto *Load = dyn_cast<LoadInst>(I))
    return analyzeAccessPattern(Load->getPointerOperand(), LI.getLoopFor(I->getParent()));
    
  if (auto *Store = dyn_cast<StoreInst>(I))
    return analyzeAccessPattern(Store->getPointerOperand(), LI.getLoopFor(I->getParent()));
    
  // Control flow instructions often need special handling
  if (isa<BranchInst>(I) || isa<SwitchInst>(I) || isa<IndirectBrInst>(I))
    return false;
    
  // Other instructions - defer to more detailed analysis in a real implementation
  return false;
}
