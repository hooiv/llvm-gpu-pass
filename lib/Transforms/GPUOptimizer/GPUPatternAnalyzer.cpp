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
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <cstdlib>
#include <map>
#include <set>
#include <vector>

using namespace llvm;
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <map>

using namespace llvm;

#define DEBUG_TYPE "gpu-pattern-analyzer"

bool GPUPatternAnalyzer::isLoopSuitableForGPU(Loop *L) {
  // First perform a quick check for obvious disqualifications
  
  // Check for loop-carried dependencies which would prevent parallelization
  std::vector<LoadInst*> Loads;
  std::vector<StoreInst*> Stores;
  collectMemoryOperations(L, Loads, Stores);
  
  for (auto *Store : Stores) {
    for (auto *Load : Loads) {
      auto Result = DI.depends(Store, Load);
      if (Result && Result->isOrdered())
        return false;
    }
    for (auto *OtherStore : Stores) {
      if (Store == OtherStore)
        continue;
      auto Result = DI.depends(Store, OtherStore);
      if (Result && Result->isOrdered())
        return false;
    }
  }
  
  // If the loop passed basic dependency checks, use the comprehensive cost model
  // to make a more informed decision about GPU offloading
  float OffloadingScore = calculateGPUOffloadingCost(L);
  
  // Define a threshold for GPU offloading
  constexpr float GPU_OFFLOADING_THRESHOLD = 0.6f;
  
  LLVM_DEBUG(dbgs() << "GPU Offloading Score for Loop: " << OffloadingScore << 
                       " (threshold: " << GPU_OFFLOADING_THRESHOLD << ")\n");
  
  return OffloadingScore >= GPU_OFFLOADING_THRESHOLD;
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
  // Identify reduction patterns in the loop that could be parallelized on a GPU
  // Common reduction patterns include sum, product, min, max, and logical reductions
  
  // Track if we found any reduction patterns
  bool FoundReduction = false;
  
  // First, look for phi nodes in the loop header 
  // (typical reduction pattern is a phi node updated in each iteration)
  BasicBlock *Header = L->getHeader();
  
  for (auto &I : *Header) {
    if (PHINode *Phi = dyn_cast<PHINode>(&I)) {
      // A reduction PHI typically has exactly two incoming values
      if (Phi->getNumIncomingValues() != 2)
        continue;
        
      // One from outside the loop (init value), one from inside
      Value *LoopVal = nullptr;
      Value *InitVal = nullptr;
      
      for (unsigned i = 0; i < 2; ++i) {
        BasicBlock *IncomingBlock = Phi->getIncomingBlock(i);
        if (L->contains(IncomingBlock)) {
          LoopVal = Phi->getIncomingValue(i);
        } else {
          InitVal = Phi->getIncomingValue(i);
        }
      }
      
      if (!LoopVal || !InitVal)
        continue;
        
      // Check if the loop value is from a binary operator 
      // (add, mul, min, max, and, or, xor)
      if (auto *BinOp = dyn_cast<BinaryOperator>(LoopVal)) {
        Value *Op0 = BinOp->getOperand(0);
        Value *Op1 = BinOp->getOperand(1);
        
        // One operand should be the phi itself (for a reduction)
        if (Op0 != Phi && Op1 != Phi)
          continue;
          
        // The other operand is typically something computed in the loop
        Value *ReductionInput = (Op0 == Phi) ? Op1 : Op0;
        
        // Check if the operation is a typical reduction operation
        switch (BinOp->getOpcode()) {
          case Instruction::Add:   // Sum reduction
          case Instruction::FAdd:
          case Instruction::Mul:   // Product reduction
          case Instruction::FMul:
          case Instruction::And:   // Bitwise AND reduction
          case Instruction::Or:    // Bitwise OR reduction
          case Instruction::Xor:   // Bitwise XOR reduction
            FoundReduction = true;
            LLVM_DEBUG(dbgs() << "Found reduction pattern in loop: " 
                              << *BinOp << "\n");
            break;
          default:
            break;
        }
      }
      
      // Also check for min/max patterns using SelectInst
      if (auto *Select = dyn_cast<SelectInst>(LoopVal)) {
        if (auto *Cmp = dyn_cast<CmpInst>(Select->getCondition())) {
          Value *Op0 = Cmp->getOperand(0);
          Value *Op1 = Cmp->getOperand(1);
          
          // One operand should be the phi itself (for a reduction)
          if ((Op0 == Phi || Op1 == Phi) && 
              (Select->getTrueValue() == Phi || Select->getFalseValue() == Phi)) {
            
            // Check for min/max patterns based on comparison predicate
            switch (Cmp->getPredicate()) {
              case CmpInst::ICMP_SLT:
              case CmpInst::ICMP_ULT:
              case CmpInst::FCMP_OLT:
              case CmpInst::ICMP_SLE:
              case CmpInst::ICMP_ULE:
              case CmpInst::FCMP_OLE:
              case CmpInst::ICMP_SGT:
              case CmpInst::ICMP_UGT:
              case CmpInst::FCMP_OGT:
              case CmpInst::ICMP_SGE:
              case CmpInst::ICMP_UGE:
              case CmpInst::FCMP_OGE:
                FoundReduction = true;
                LLVM_DEBUG(dbgs() << "Found min/max reduction pattern in loop: " 
                                  << *Select << "\n");
                break;
              default:
                break;
            }
          }
        }
      }
    }
  }
  
  return FoundReduction;
}

void GPUPatternAnalyzer::collectMemoryOperations(Loop *L, 
                                      std::vector<LoadInst*> &Loads,
                                      std::vector<StoreInst*> &Stores) {
  // Collect all load and store operations within the loop
  for (BasicBlock *BB : L->getBlocks()) {
    for (Instruction &I : *BB) {
      if (auto *Load = dyn_cast<LoadInst>(&I)) {
        Loads.push_back(Load);
      } else if (auto *Store = dyn_cast<StoreInst>(&I)) {
        Stores.push_back(Store);
      }
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
  // Check if this PHI node represents a reduction operation
  // A reduction typically has a pattern like: phi = phi + x
  
  // Phi should have exactly two incoming values
  if (Phi->getNumIncomingValues() != 2)
    return false;
    
  BasicBlock *Header = L->getHeader();
  BasicBlock *Latch = L->getLoopLatch();
  
  // Check if the Phi is in the loop header
  if (Phi->getParent() != Header)
    return false;
    
  // Get the incoming values
  Value *InitVal = nullptr;
  Value *LoopVal = nullptr;
  
  for (unsigned i = 0; i < 2; i++) {
    BasicBlock *IncomingBB = Phi->getIncomingBlock(i);
    
    if (L->contains(IncomingBB)) {
      // This value comes from inside the loop
      LoopVal = Phi->getIncomingValue(i);
    } else {
      // This value comes from outside the loop (initial value)
      InitVal = Phi->getIncomingValue(i);
    }
  }
  
  // We should have both values
  if (!InitVal || !LoopVal)
    return false;
    
  // Check if the loop value is a binary operation (like add, mul, etc.)
  BinaryOperator *BinOp = dyn_cast<BinaryOperator>(LoopVal);
  if (!BinOp)
    return false;
    
  // Check if one of the operands is the Phi itself
  Value *Op0 = BinOp->getOperand(0);
  Value *Op1 = BinOp->getOperand(1);
  
  bool IsPossibleReduction = false;
  
  // Check for the pattern phi = phi op x or phi = x op phi
  if (Op0 == Phi || Op1 == Phi) {
    switch (BinOp->getOpcode()) {
      case Instruction::Add: // Sum reduction
      case Instruction::FAdd:
      case Instruction::Mul: // Product reduction
      case Instruction::FMul:
      case Instruction::And: // Logical AND reduction
      case Instruction::Or:  // Logical OR reduction
      case Instruction::Xor: // Logical XOR reduction
        IsPossibleReduction = true;
        break;
      default:
        break;
    }
  }
  
  // Check for max/min patterns that might use select instead of SMax/SMin/etc.
  if (!IsPossibleReduction) {
    // Look for pattern: phi = select(cmp(phi, x), phi, x) or similar
    if (SelectInst *Select = dyn_cast<SelectInst>(LoopVal)) {
      Value *Condition = Select->getCondition();
      Value *TrueVal = Select->getTrueValue();
      Value *FalseVal = Select->getFalseValue();
      
      if (CmpInst *Cmp = dyn_cast<CmpInst>(Condition)) {
        Value *CmpOp0 = Cmp->getOperand(0);
        Value *CmpOp1 = Cmp->getOperand(1);
        
        // Check if comparison involves Phi
        if ((CmpOp0 == Phi || CmpOp1 == Phi) &&
            ((TrueVal == Phi && FalseVal != Phi) || (TrueVal != Phi && FalseVal == Phi))) {
          IsPossibleReduction = true;
        }
      }
    }
  }
  
  return IsPossibleReduction;
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

float GPUPatternAnalyzer::calculateGPUOffloadingCost(Loop *L) {
  // This is a comprehensive cost model that considers multiple factors
  // Each factor has a weight that can be adjusted based on empirical data
  
  // 1. Computational density - high arithmetic intensity is good for GPUs
  float computeDensityScore = 0.0f;
  unsigned ComputeOps = 0;
  unsigned MemoryOps = 0;
  
  for (auto *BB : L->getBlocks()) {
    for (auto &I : *BB) {
      if (isa<LoadInst>(I) || isa<StoreInst>(I))
        MemoryOps++;
      else if (I.getOpcode() >= Instruction::Add && I.getOpcode() <= Instruction::Xor)
        ComputeOps++;
      // Additional computation types that benefit from GPU execution
      else if (isa<FPMathOperator>(&I))
        ComputeOps += 2; // Weight floating point ops more heavily
    }
  }
  
  if (MemoryOps > 0) {
    float ArithmeticIntensity = static_cast<float>(ComputeOps) / MemoryOps;
    // Scale between 0 and 1 with diminishing returns after 4.0
    computeDensityScore = std::min(ArithmeticIntensity / 4.0f, 1.0f);
  }
  
  // 2. Parallelism potential
  float parallelismScore = calculateParallelizationPotential(L);
  
  // 3. Memory access pattern regularity
  std::vector<LoadInst*> Loads;
  std::vector<StoreInst*> Stores;
  collectMemoryOperations(L, Loads, Stores);
  
  float memoryPatternScore = 0.0f;
  unsigned RegularAccesses = 0;
  unsigned TotalAccesses = Loads.size() + Stores.size();
  
  for (auto *Load : Loads) {
    if (analyzeAccessPattern(Load->getPointerOperand(), L))
      RegularAccesses++;
  }
  
  for (auto *Store : Stores) {
    if (analyzeAccessPattern(Store->getPointerOperand(), L))
      RegularAccesses++;
  }
  
  if (TotalAccesses > 0) {
    memoryPatternScore = static_cast<float>(RegularAccesses) / TotalAccesses;
  }
  
  // 4. Data transfer overhead (lower is better)
  float transferOverheadScore = 1.0f - estimateDataTransferOverhead(L);
  
  // 5. Loop trip count (higher is better, but with diminishing returns)
  const SCEV *TripCount = estimateTripCount(L);
  float tripCountScore = 0.0f;
  
  if (auto *ConstTripCount = dyn_cast<SCEVConstant>(TripCount)) {
    uint64_t TC = ConstTripCount->getAPInt().getZExtValue();
    // Scale between 0 and 1 with diminishing returns after 1024
    tripCountScore = std::min(static_cast<float>(TC) / 1024.0f, 1.0f);
  } else {
    // If we can't determine the exact trip count, use a moderate value
    tripCountScore = 0.5f;
  }
    // 6. Architecture-specific factors
  auto ArchFactors = getGPUArchitectureCostFactors();
  
  // 7. Shared memory optimization potential
  float sharedMemoryScore = 0.0f;
  if (analyzeSharedMemoryOptimizationPotential(L)) {
    // Estimate shared memory requirements
    uint64_t SharedMemBytes = estimateSharedMemoryRequirement(L);
    
    // Check if we have enough shared memory in the target GPU
    float MaxSharedMemoryKB = ArchFactors["sharedMemorySize"];
    uint64_t MaxSharedMemoryBytes = static_cast<uint64_t>(MaxSharedMemoryKB * 1024);
    
    if (SharedMemBytes <= MaxSharedMemoryBytes) {
      // Calculate how efficiently we're using the shared memory
      float UsageRatio = static_cast<float>(SharedMemBytes) / MaxSharedMemoryBytes;
      
      // Best score is when we use a good amount but not too much (25-75% usage)
      if (UsageRatio < 0.25f) {
        sharedMemoryScore = UsageRatio * 4.0f; // Scale up for small usages
      } else if (UsageRatio <= 0.75f) {
        sharedMemoryScore = 1.0f; // Ideal range
      } else {
        sharedMemoryScore = 1.0f - ((UsageRatio - 0.75f) * 4.0f); // Scale down for high usage
      }
      
      // Ensure the score stays between 0 and 1
      sharedMemoryScore = std::max(0.0f, std::min(1.0f, sharedMemoryScore));
    } else {
      // Not enough shared memory, give a lower score
      sharedMemoryScore = 0.2f;
    }
  }
  
  // Apply weights to each factor and compute final score
  float weightCompute = ArchFactors["computeWeight"];
  float weightParallelism = ArchFactors["parallelismWeight"];
  float weightMemory = ArchFactors["memoryPatternWeight"];
  float weightTransfer = ArchFactors["transferWeight"];
  float weightTripCount = ArchFactors["tripCountWeight"];
  float weightSharedMem = 0.15f; // Weight for shared memory optimization potential
  
  float finalScore = 
    (computeDensityScore * weightCompute) +
    (parallelismScore * weightParallelism) +
    (memoryPatternScore * weightMemory) +
    (transferOverheadScore * weightTransfer) +
    (tripCountScore * weightTripCount) +
    (sharedMemoryScore * weightSharedMem);
  
  // Normalize the score between 0 and 1
  float weightSum = weightCompute + weightParallelism + weightMemory + 
                    weightTransfer + weightTripCount + weightSharedMem;
  
  finalScore /= weightSum;
  
  LLVM_DEBUG(dbgs() << "GPU Offloading Cost Analysis for Loop:\n"
                    << "  Compute Density Score: " << computeDensityScore << "\n"
                    << "  Parallelism Score: " << parallelismScore << "\n"
                    << "  Memory Pattern Score: " << memoryPatternScore << "\n"
                    << "  Transfer Overhead Score: " << transferOverheadScore << "\n"
                    << "  Trip Count Score: " << tripCountScore << "\n"
                    << "  Shared Memory Score: " << sharedMemoryScore << "\n"
                    << "  Final Score: " << finalScore << "\n");
  
  return finalScore;
}

float GPUPatternAnalyzer::estimateDataTransferOverhead(Loop *L) {
  // Estimate the overhead of transferring data between CPU and GPU
  // 1. Calculate input data size
  // 2. Calculate output data size
  // 3. Compare to the estimated computation time
  
  unsigned InputBytes = 0;
  unsigned OutputBytes = 0;
  
  std::vector<LoadInst*> Loads;
  std::vector<StoreInst*> Stores;
  collectMemoryOperations(L, Loads, Stores);
  
  // Track memory objects that need to be transferred
  DenseSet<Value*> InputObjects;
  DenseSet<Value*> OutputObjects;
  
  // Gather all unique memory objects accessed
  for (auto *Load : Loads) {
    Value *Ptr = Load->getPointerOperand()->stripPointerCasts();
    InputObjects.insert(Ptr);
  }
  
  for (auto *Store : Stores) {
    Value *Ptr = Store->getPointerOperand()->stripPointerCasts();
    OutputObjects.insert(Ptr);
  }
  
  // Estimate size of each memory object
  for (Value *Obj : InputObjects) {
    if (auto *AI = dyn_cast<AllocaInst>(Obj)) {
      Type *ElemTy = AI->getAllocatedType();
      unsigned ElemSize = ElemTy->getPrimitiveSizeInBits() / 8;
      
      // For arrays, get the size
      if (ElemTy->isArrayTy()) {
        unsigned NumElements = ElemTy->getArrayNumElements();
        ElemSize *= NumElements;
      }
      
      InputBytes += ElemSize;
    } else if (auto *GV = dyn_cast<GlobalVariable>(Obj)) {
      Type *ElemTy = GV->getValueType();
      unsigned ElemSize = ElemTy->getPrimitiveSizeInBits() / 8;
      
      // For arrays, get the size
      if (ElemTy->isArrayTy()) {
        unsigned NumElements = ElemTy->getArrayNumElements();
        ElemSize *= NumElements;
      }
      
      InputBytes += ElemSize;
    }
    // Handle pointer arguments and other cases as well
  }
  
  // Similar calculation for outputs
  for (Value *Obj : OutputObjects) {
    if (auto *AI = dyn_cast<AllocaInst>(Obj)) {
      Type *ElemTy = AI->getAllocatedType();
      unsigned ElemSize = ElemTy->getPrimitiveSizeInBits() / 8;
      
      // For arrays, get the size
      if (ElemTy->isArrayTy()) {
        unsigned NumElements = ElemTy->getArrayNumElements();
        ElemSize *= NumElements;
      }
      
      OutputBytes += ElemSize;
    } else if (auto *GV = dyn_cast<GlobalVariable>(Obj)) {
      Type *ElemTy = GV->getValueType();
      unsigned ElemSize = ElemTy->getPrimitiveSizeInBits() / 8;
      
      // For arrays, get the size
      if (ElemTy->isArrayTy()) {
        unsigned NumElements = ElemTy->getArrayNumElements();
        ElemSize *= NumElements;
      }
      
      OutputBytes += ElemSize;
    }
  }
  
  // Estimate computation operations (already calculated in other methods)
  unsigned ComputeOps = 0;
  for (auto *BB : L->getBlocks()) {
    for (auto &I : *BB) {
      if (I.getOpcode() >= Instruction::Add && I.getOpcode() <= Instruction::Xor)
        ComputeOps++;
      else if (isa<FPMathOperator>(&I))
        ComputeOps += 2; // Weight floating point ops more heavily
    }
  }
  
  // Get trip count if available
  uint64_t TripCount = 1000; // Default assumption
  const SCEV *TripCountSCEV = estimateTripCount(L);
  if (auto *ConstTripCount = dyn_cast<SCEVConstant>(TripCountSCEV)) {
    TripCount = ConstTripCount->getAPInt().getZExtValue();
  }
  
  // Scale by trip count to get total bytes
  uint64_t TotalInputBytes = InputBytes * TripCount;
  uint64_t TotalOutputBytes = OutputBytes * TripCount;
  uint64_t TotalTransferBytes = TotalInputBytes + TotalOutputBytes;
  
  // Assume GPU memory bandwidth and compute throughput from architecture factors
  auto ArchFactors = getGPUArchitectureCostFactors();
  float GPUMemoryBandwidth = ArchFactors["memoryBandwidth"]; // GB/s
  float GPUComputeThroughput = ArchFactors["computeThroughput"]; // GFLOPS
  
  // Calculate transfer time and compute time
  float TransferTimeMs = static_cast<float>(TotalTransferBytes) / (GPUMemoryBandwidth * 1024 * 1024) * 1000;
  float ComputeTimeMs = static_cast<float>(ComputeOps * TripCount) / (GPUComputeThroughput * 1000 * 1000) * 1000;
  
  // Overhead ratio: transfer time as a fraction of compute time
  float OverheadRatio = (ComputeTimeMs > 0) ? TransferTimeMs / ComputeTimeMs : 10.0f;
  
  // Normalize to a score between 0 and 1 (lower is better)
  // A ratio of 0 means no overhead, a ratio of 1 or higher means the overhead exceeds compute time
  float TransferOverhead = std::min(OverheadRatio, 1.0f);
  
  return TransferOverhead;
}

const SCEV *GPUPatternAnalyzer::estimateTripCount(Loop *L) {
  // Try to get an exact trip count from ScalarEvolution
  if (const SCEV *ExactCount = SE.getBackedgeTakenCount(L)) {
    if (isa<SCEVConstant>(ExactCount)) {
      return ExactCount;
    }
  }
  
  // If we can't get an exact count, try to find the loop exit condition
  SmallVector<BasicBlock *, 8> ExitingBlocks;
  L->getExitingBlocks(ExitingBlocks);
  
  if (ExitingBlocks.empty()) {
    // No exiting blocks found, assume a large trip count (infinity loop or complex exit)
    Type *IntPtrTy = Type::getInt64Ty(L->getHeader()->getContext());
    return SE.getConstant(IntPtrTy, 100);
  }
  
  // Look at each exiting block to find a suitable upper bound
  for (BasicBlock *ExitingBlock : ExitingBlocks) {
    Instruction *TermInst = ExitingBlock->getTerminator();
    if (!TermInst) continue;
    
    if (BranchInst *BI = dyn_cast<BranchInst>(TermInst)) {
      if (BI->isConditional()) {
        // Try to analyze the branch condition
        Value *Condition = BI->getCondition();
        
        if (ICmpInst *ICI = dyn_cast<ICmpInst>(Condition)) {
          // Extract operands of the comparison
          Value *Op0 = ICI->getOperand(0);
          Value *Op1 = ICI->getOperand(1);
          
          // Typically one operand is an induction variable and the other is a bound
          const SCEV *Op0SCEV = SE.getSCEV(Op0);
          const SCEV *Op1SCEV = SE.getSCEV(Op1);
          
          // Check if one operand is a constant
          if (const SCEVConstant *ConstSCEV = dyn_cast<SCEVConstant>(Op1SCEV)) {
            uint64_t ConstVal = ConstSCEV->getAPInt().getZExtValue();
            
            // Check if the other operand is an AddRec (induction variable)
            if (const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(Op0SCEV)) {
              if (AddRec->getLoop() == L) {
                // Found an induction variable compared against a constant
                // Now interpret based on the comparison predicate
                switch (ICI->getPredicate()) {
                  case CmpInst::ICMP_ULT:
                  case CmpInst::ICMP_SLT:
                    // iv < const -> trip count is approximately const
                    Type *IntPtrTy = Type::getInt64Ty(L->getHeader()->getContext());
                    return SE.getConstant(IntPtrTy, ConstVal);
                    
                  case CmpInst::ICMP_ULE:
                  case CmpInst::ICMP_SLE:
                    // iv <= const -> trip count is approximately const+1
                    Type *IntPtrTy = Type::getInt64Ty(L->getHeader()->getContext());
                    return SE.getConstant(IntPtrTy, ConstVal + 1);
                    
                  default:
                    // Other comparisons are harder to analyze
                    break;
                }
              }
            }
          }
        }
      }
    }
  }
  
  // Default: assume a moderate trip count if we couldn't determine it
  Type *IntPtrTy = Type::getInt64Ty(L->getHeader()->getContext());
  return SE.getConstant(IntPtrTy, 100);
}

std::map<std::string, float> GPUPatternAnalyzer::getGPUArchitectureCostFactors() const {
  std::map<std::string, float> Factors;
  
  // Default values (used if no specific architecture is detected)
  Factors["computeWeight"] = 0.3f;
  Factors["parallelismWeight"] = 0.25f;
  Factors["memoryPatternWeight"] = 0.2f;
  Factors["transferWeight"] = 0.15f;
  Factors["tripCountWeight"] = 0.1f;
  Factors["sharedMemorySize"] = 48.0f;  // 48 KB
  Factors["warpSize"] = 32.0f;
  Factors["maxThreadsPerBlock"] = 1024.0f;
  Factors["memoryCoalescingImportance"] = 0.8f;
  Factors["computeThroughput"] = 1.0f;
  Factors["memoryBandwidth"] = 1.0f;
  
  // Get the target GPU architecture and adjust values accordingly
  GPUArch Arch = getTargetGPUArchitecture();
  
  switch (Arch) {
    case GPUArch::NVIDIA_AMPERE:
      // NVIDIA Ampere architecture (SM 80, 86)
      Factors["computeWeight"] = 0.35f;
      Factors["parallelismWeight"] = 0.30f;
      Factors["memoryPatternWeight"] = 0.15f;
      Factors["transferWeight"] = 0.1f;
      Factors["tripCountWeight"] = 0.1f;
      Factors["sharedMemorySize"] = 164.0f;  // Up to 164 KB shared memory per SM
      Factors["warpSize"] = 32.0f;
      Factors["maxThreadsPerBlock"] = 1024.0f;
      Factors["memoryCoalescingImportance"] = 0.9f;
      Factors["computeThroughput"] = 2.0f;   // Relative to baseline
      Factors["memoryBandwidth"] = 2.0f;     // Relative to baseline
      break;
      
    case GPUArch::NVIDIA_TURING:
      // NVIDIA Turing architecture (SM 75)
      Factors["computeWeight"] = 0.35f;
      Factors["parallelismWeight"] = 0.25f;
      Factors["memoryPatternWeight"] = 0.15f;
      Factors["transferWeight"] = 0.15f;
      Factors["tripCountWeight"] = 0.1f;
      Factors["sharedMemorySize"] = 64.0f;   // 64 KB shared memory per SM
      Factors["warpSize"] = 32.0f;
      Factors["maxThreadsPerBlock"] = 1024.0f;
      Factors["memoryCoalescingImportance"] = 0.85f;
      Factors["computeThroughput"] = 1.5f;   // Relative to baseline
      Factors["memoryBandwidth"] = 1.5f;     // Relative to baseline
      break;
      
    case GPUArch::NVIDIA_VOLTA:
      // NVIDIA Volta architecture (SM 70)
      Factors["computeWeight"] = 0.30f;
      Factors["parallelismWeight"] = 0.25f;
      Factors["memoryPatternWeight"] = 0.2f;
      Factors["transferWeight"] = 0.15f;
      Factors["tripCountWeight"] = 0.1f;
      Factors["sharedMemorySize"] = 96.0f;   // 96 KB shared memory per SM
      Factors["warpSize"] = 32.0f;
      Factors["maxThreadsPerBlock"] = 1024.0f;
      Factors["memoryCoalescingImportance"] = 0.85f;
      Factors["computeThroughput"] = 1.3f;   // Relative to baseline
      Factors["memoryBandwidth"] = 1.4f;     // Relative to baseline
      break;
      
    case GPUArch::NVIDIA_PASCAL:
      // NVIDIA Pascal architecture (SM 60, 61)
      Factors["computeWeight"] = 0.25f;
      Factors["parallelismWeight"] = 0.25f;
      Factors["memoryPatternWeight"] = 0.2f;
      Factors["transferWeight"] = 0.2f;
      Factors["tripCountWeight"] = 0.1f;
      Factors["sharedMemorySize"] = 48.0f;   // 48 KB shared memory per SM
      Factors["warpSize"] = 32.0f;
      Factors["maxThreadsPerBlock"] = 1024.0f;
      Factors["memoryCoalescingImportance"] = 0.8f;
      Factors["computeThroughput"] = 1.0f;   // Baseline
      Factors["memoryBandwidth"] = 1.0f;     // Baseline
      break;
      
    case GPUArch::AMD_RDNA2:
      // AMD RDNA2 architecture (gfx1030)
      Factors["computeWeight"] = 0.30f;
      Factors["parallelismWeight"] = 0.30f;
      Factors["memoryPatternWeight"] = 0.2f;
      Factors["transferWeight"] = 0.1f;
      Factors["tripCountWeight"] = 0.1f;
      Factors["sharedMemorySize"] = 64.0f;   // 64 KB LDS per workgroup
      Factors["warpSize"] = 64.0f;           // Wavefront size
      Factors["maxThreadsPerBlock"] = 1024.0f;
      Factors["memoryCoalescingImportance"] = 0.85f;
      Factors["computeThroughput"] = 1.7f;   // Relative to baseline
      Factors["memoryBandwidth"] = 1.6f;     // Relative to baseline
      break;
      
    case GPUArch::AMD_CDNA2:
      // AMD CDNA2 architecture (MI200, gfx90a)
      Factors["computeWeight"] = 0.4f;       // More compute focused
      Factors["parallelismWeight"] = 0.3f;
      Factors["memoryPatternWeight"] = 0.15f;
      Factors["transferWeight"] = 0.1f;
      Factors["tripCountWeight"] = 0.05f;
      Factors["sharedMemorySize"] = 64.0f;   // 64 KB LDS
      Factors["warpSize"] = 64.0f;           // Wavefront size
      Factors["maxThreadsPerBlock"] = 1024.0f;
      Factors["memoryCoalescingImportance"] = 0.9f;
      Factors["computeThroughput"] = 2.1f;   // Relative to baseline
      Factors["memoryBandwidth"] = 1.8f;     // Relative to baseline
      break;
      
    case GPUArch::INTEL_XE_HPC:
      // Intel Xe HPC (Ponte Vecchio)
      Factors["computeWeight"] = 0.35f;
      Factors["parallelismWeight"] = 0.3f;
      Factors["memoryPatternWeight"] = 0.15f;
      Factors["transferWeight"] = 0.1f;
      Factors["tripCountWeight"] = 0.1f;
      Factors["sharedMemorySize"] = 64.0f;   // Shared local memory
      Factors["warpSize"] = 32.0f;           // Subgroup size
      Factors["maxThreadsPerBlock"] = 1024.0f;
      Factors["memoryCoalescingImportance"] = 0.85f;
      Factors["computeThroughput"] = 1.8f;   // Relative to baseline
      Factors["memoryBandwidth"] = 1.7f;     // Relative to baseline
      break;
      
    default:
      // Use default values
      break;
  }
  
  return Factors;
}

GPUArch GPUPatternAnalyzer::getTargetGPUArchitecture() const {
  // Determine the target GPU architecture
  // This is used to customize cost models and optimization heuristics
  
  // First check environment variables (allows user override)
  std::string EnvVar;
  if (const char *EnvValue = std::getenv("GPU_TARGET_ARCH")) {
    EnvVar = EnvValue;
    
    // Convert to lowercase for case-insensitive comparison
    std::transform(EnvVar.begin(), EnvVar.end(), EnvVar.begin(), 
                  [](unsigned char c) { return std::tolower(c); });
    
    // Check for NVIDIA architectures
    if (EnvVar == "ampere" || EnvVar == "sm_80" || EnvVar == "sm_86") {
      return GPUArch::NVIDIA_AMPERE;
    } else if (EnvVar == "turing" || EnvVar == "sm_75") {
      return GPUArch::NVIDIA_TURING;
    } else if (EnvVar == "volta" || EnvVar == "sm_70") {
      return GPUArch::NVIDIA_VOLTA;
    } else if (EnvVar == "pascal" || EnvVar == "sm_60" || EnvVar == "sm_61") {
      return GPUArch::NVIDIA_PASCAL;
    } else if (EnvVar == "maxwell" || EnvVar == "sm_50" || EnvVar == "sm_52") {
      return GPUArch::NVIDIA_MAXWELL;
    }
    
    // Check for AMD architectures
    else if (EnvVar == "rdna2" || EnvVar == "gfx1030" || EnvVar == "navi2x") {
      return GPUArch::AMD_RDNA2;
    } else if (EnvVar == "rdna" || EnvVar == "gfx1010" || EnvVar == "navi") {
      return GPUArch::AMD_RDNA;
    } else if (EnvVar == "cdna2" || EnvVar == "gfx90a" || EnvVar == "mi200") {
      return GPUArch::AMD_CDNA2;
    } else if (EnvVar == "cdna" || EnvVar == "gfx908" || EnvVar == "mi100") {
      return GPUArch::AMD_CDNA;
    } else if (EnvVar == "vega" || EnvVar == "gfx900") {
      return GPUArch::AMD_VEGA;
    }
    
    // Check for Intel architectures
    else if (EnvVar == "xe_hpc" || EnvVar == "pvc" || EnvVar == "ponte_vecchio") {
      return GPUArch::INTEL_XE_HPC;
    } else if (EnvVar == "xe_hpg" || EnvVar == "alchemist" || EnvVar == "arc") {
      return GPUArch::INTEL_XE_HPG;
    } else if (EnvVar == "xe_lp" || EnvVar == "gen12" || EnvVar == "tigerlake") {
      return GPUArch::INTEL_XE_LP;
    }
  }
  
  // If no environment variable is set, we could try to detect the GPU
  // through system APIs (CUDA Runtime, ROCm, Level Zero, etc.)
  // This would require linking against the respective APIs
  
  // For now, use a reasonable default
  // Ampere for NVIDIA, RDNA2 for AMD, Xe HPG for Intel
  // NVIDIA is the most common in HPC/AI workloads
  return GPUArch::NVIDIA_AMPERE;
}

bool GPUPatternAnalyzer::analyzeSharedMemoryOptimizationPotential(Loop *L) {
  // This method analyzes the loop to determine if it can benefit from shared memory optimization
  // Several patterns can benefit from shared memory optimization:
  // 1. Reuse of data across threads in a block (spatial locality)
  // 2. Reuse of data over time by the same thread (temporal locality)
  // 3. Reduction patterns that can be optimized with shared memory

  // First, check for basic prerequisites
  if (!isLoopSuitableForGPU(L))
    return false;
    
  // Gather all memory operations in the loop
  std::vector<LoadInst*> Loads;
  std::vector<StoreInst*> Stores;
  collectMemoryOperations(L, Loads, Stores);
  
  if (Loads.empty()) 
    return false; // No loads, no point in shared memory optimizations
  
  // Identify potential array accesses for tiling
  DenseSet<Value*> ArrayBases;
  for (auto *Load : Loads) {
    Value *Ptr = Load->getPointerOperand()->stripPointerCasts();
    
    // Check if this is an array access (pointer arithmetic)
    if (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
      Value *BasePtr = GEP->getPointerOperand();
      ArrayBases.insert(BasePtr);
    }
  }
  
  // Check if any array access pattern is suitable for shared memory tiling
  for (Value *ArrayBase : ArrayBases) {
    if (canApplySharedMemoryTiling(ArrayBase, L))
      return true;
  }
  
  // Check for reduction patterns that can be optimized with shared memory
  if (identifyReductionPatterns(L)) {
    // Analyze if this reduction can benefit from shared memory
    // Typically reductions across threads in a block benefit significantly
    
    // Check the reduction operation type and data size
    for (BasicBlock *BB : L->getBlocks()) {
      for (Instruction &I : *BB) {
        if (auto *Phi = dyn_cast<PHINode>(&I)) {
          if (isReductionOperation(Phi, L)) {
            // Found a reduction pattern
            // Shared memory is beneficial if the reduction is across threads
            return true;
          }
        }
      }
    }
  }
  
  // Check for stencil-like access patterns where neighboring elements are accessed
  bool HasStencilPattern = false;
  DenseMap<Value*, std::set<int64_t>> OffsetsByBase;
  
  for (auto *Load : Loads) {
    Value *Ptr = Load->getPointerOperand()->stripPointerCasts();
    
    if (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
      Value *BasePtr = GEP->getPointerOperand();
      
      // Check if this is a constant offset from the base
      if (GEP->hasAllConstantIndices() && GEP->getNumIndices() >= 1) {
        // Get the last index which is often the most relevant for stencil patterns
        if (auto *ConstIdx = dyn_cast<ConstantInt>(GEP->getOperand(GEP->getNumOperands() - 1))) {
          int64_t Offset = ConstIdx->getSExtValue();
          OffsetsByBase[BasePtr].insert(Offset);
        }
      }
    }
  }
  
  // Check if we have a stencil-like pattern with neighboring accesses
  for (auto &Entry : OffsetsByBase) {
    std::set<int64_t> &Offsets = Entry.second;
    
    // Check if there are multiple close offsets, which indicates a stencil pattern
    if (Offsets.size() > 1) {
      int64_t Min = *Offsets.begin();
      int64_t Max = *Offsets.rbegin();
      
      // If the range of offsets is small and there are multiple accesses,
      // it's likely a stencil pattern that can benefit from shared memory
      if ((Max - Min) <= 16 && Offsets.size() >= 3) {
        HasStencilPattern = true;
        break;
      }
    }
  }
  
  return HasStencilPattern;
}

uint64_t GPUPatternAnalyzer::estimateSharedMemoryRequirement(Loop *L) {
  // Estimate the amount of shared memory needed for optimizing this loop
  // This is a key factor in determining if shared memory optimization is feasible
  
  // If the loop is not suitable for shared memory optimization, return 0
  if (!analyzeSharedMemoryOptimizationPotential(L))
    return 0;
  
  // Gather all memory operations in the loop
  std::vector<LoadInst*> Loads;
  std::vector<StoreInst*> Stores;
  collectMemoryOperations(L, Loads, Stores);
  
  // Identify unique array bases
  DenseMap<Value*, Type*> ArraysWithTypes;
  for (auto *Load : Loads) {
    Value *Ptr = Load->getPointerOperand()->stripPointerCasts();
    
    if (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
      Value *BasePtr = GEP->getPointerOperand();
      Type *ElemTy = GEP->getSourceElementType();
      ArraysWithTypes[BasePtr] = ElemTy;
    }
  }
  
  uint64_t TotalSharedMemory = 0;
  
  // For each array that can be tiled, estimate the tile size
  for (auto &Entry : ArraysWithTypes) {
    Value *ArrayBase = Entry.first;
    Type *ElemType = Entry.second;
    
    if (canApplySharedMemoryTiling(ArrayBase, L)) {
      // Calculate element size
      uint64_t ElemSize = ElemType->getPrimitiveSizeInBits() / 8;
      if (ElemSize == 0) ElemSize = 1; // Minimum of 1 byte
      
      // Estimate tile dimensions
      // For 1D tiling: we typically use blockDim.x + 2*halo elements
      // For 2D tiling: we typically use (blockDim.x + 2*halo) * (blockDim.y + 2*halo) elements
      
      // Determine if we're dealing with 1D or 2D tiling
      bool Is2DTiling = false;
      for (auto *Load : Loads) {
        Value *Ptr = Load->getPointerOperand()->stripPointerCasts();
        
        if (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
          if (GEP->getPointerOperand() == ArrayBase) {
            // Check if this GEP has multiple variable indices
            unsigned NumVarIndices = 0;
            for (unsigned i = 1; i < GEP->getNumOperands(); i++) {
              if (!isa<ConstantInt>(GEP->getOperand(i)))
                NumVarIndices++;
            }
            if (NumVarIndices >= 2) {
              Is2DTiling = true;
              break;
            }
          }
        }
      }
      
      // Estimate tile size based on typical GPU block dimensions
      uint64_t TileSize;
      if (Is2DTiling) {
        // For 2D tiling, use a typical tile size of 16x16 or 32x32 with halo regions
        uint64_t BlockDimX = 16; // Typical block dimension
        uint64_t BlockDimY = 16;
        uint64_t Halo = 1;       // Halo size for stencil operations
        
        TileSize = (BlockDimX + 2 * Halo) * (BlockDimY + 2 * Halo) * ElemSize;
      } else {
        // For 1D tiling, use a typical tile size of 256 with halo regions
        uint64_t BlockDimX = 256; // Typical block dimension
        uint64_t Halo = 1;        // Halo size for stencil operations
        
        TileSize = (BlockDimX + 2 * Halo) * ElemSize;
      }
      
      TotalSharedMemory += TileSize;
    }
  }
  
  // For reduction patterns, estimate shared memory needed for parallel reduction
  if (identifyReductionPatterns(L)) {
    // A parallel reduction typically needs one value per thread in a block
    uint64_t BlockSize = 256; // Typical block size
    uint64_t ReductionElemSize = 4; // Assume 4 bytes (float or int)
    
    // Find actual element size based on the reduction variable
    for (BasicBlock *BB : L->getBlocks()) {
      for (Instruction &I : *BB) {
        if (auto *Phi = dyn_cast<PHINode>(&I)) {
          if (isReductionOperation(Phi, L)) {
            Type *RedType = Phi->getType();
            ReductionElemSize = RedType->getPrimitiveSizeInBits() / 8;
            if (ReductionElemSize == 0) ReductionElemSize = 4; // Minimum of 4 bytes
            break;
          }
        }
      }
    }
    
    TotalSharedMemory += BlockSize * ReductionElemSize;
  }
  
  return TotalSharedMemory;
}

bool GPUPatternAnalyzer::canApplySharedMemoryTiling(Value *ArrayBase, Loop *L) {
  // Determine if an array access pattern is suitable for tiling in shared memory
  // We look for:
  // 1. Regular access patterns (stride-1 or constant stride)
  // 2. Reuse of array elements across loop iterations
  // A perfect case for tiling is when an array is accessed with spatial locality
  
  bool HasRegularAccess = false;
  bool HasLocalityAcrossIterations = false;
  
  // Gather all loads that access this array
  std::vector<GetElementPtrInst*> GEPs;
  
  for (BasicBlock *BB : L->getBlocks()) {
    for (Instruction &I : *BB) {
      if (auto *Load = dyn_cast<LoadInst>(&I)) {
        Value *Ptr = Load->getPointerOperand()->stripPointerCasts();
        
        if (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
          if (GEP->getPointerOperand() == ArrayBase) {
            GEPs.push_back(GEP);
          }
        }
      }
    }
  }
  
  if (GEPs.empty())
    return false;
  
  // Check for regular access patterns
  for (auto *GEP : GEPs) {
    const SCEV *AccessFunction = SE.getSCEV(GEP);
    
    if (auto *AddRec = dyn_cast<SCEVAddRecExpr>(AccessFunction)) {
      if (AddRec->getLoop() == L) {
        // Check for constant stride
        const SCEV *Stride = AddRec->getStepRecurrence(SE);
        if (isa<SCEVConstant>(Stride)) {
          HasRegularAccess = true;
          break;
        }
      }
    }
  }
  
  // Check for locality across iterations
  // This is a bit more complex and would require dependence analysis
  // For simplicity, we'll use a heuristic approach
  
  if (HasRegularAccess) {
    // If we have multiple GEPs accessing the same array with offset patterns,
    // it likely indicates locality that can benefit from shared memory
    
    if (GEPs.size() >= 3) {
      // More than 3 accesses to the same array typically indicates reuse
      HasLocalityAcrossIterations = true;
    } else {
      // Check for accesses with small constant offsets from each other
      // which indicates stencil-like patterns with locality
      std::set<int64_t> ConstOffsets;
      
      for (auto *GEP : GEPs) {
        if (GEP->hasAllConstantIndices()) {
          // For simplicity, just look at the last index
          if (auto *ConstIdx = dyn_cast<ConstantInt>(GEP->getOperand(GEP->getNumOperands() - 1))) {
            ConstOffsets.insert(ConstIdx->getSExtValue());
          }
        }
      }
      
      // If we have multiple close offsets, that indicates spatial locality
      if (ConstOffsets.size() >= 2) {
        int64_t Min = INT64_MAX;
        int64_t Max = INT64_MIN;
        
        for (int64_t Offset : ConstOffsets) {
          Min = std::min(Min, Offset);
          Max = std::max(Max, Offset);
        }
        
        // If the span is small, it indicates spatial locality
        if ((Max - Min) <= 16) {
          HasLocalityAcrossIterations = true;
        }
      }
    }
  }
  
  return HasRegularAccess && HasLocalityAcrossIterations;
}
