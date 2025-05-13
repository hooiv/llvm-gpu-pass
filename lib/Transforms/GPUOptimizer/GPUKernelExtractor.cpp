//===- GPUKernelExtractor.cpp - Extract kernels for GPU execution --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------===//
//
// This file implements the GPUKernelExtractor class which identifies and extracts
// code regions that are suitable for GPU execution.
//
//===----------------------------------------------------------------===//

#include "GPUKernelExtractor.h"
#include "GPUPatternAnalyzer.h"
#include "GPULoopTransformer.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include <algorithm>
#include <queue>

using namespace llvm;

#define DEBUG_TYPE "gpu-kernel-extractor"

bool GPUKernelExtractor::extractKernels(Function &F) {
  LLVM_DEBUG(dbgs() << "GPUKernelExtractor: Scanning function " << F.getName() << " for GPU kernels\n");
  
  // Create analyzer for loop analysis
  GPUPatternAnalyzer Analyzer(LI, DI, SE);
  
  // Find suitable loops for extraction
  std::vector<Loop*> SuitableLoops = identifySuitableLoops(F);
  
  // Find other computational regions (non-loop)
  std::vector<std::vector<BasicBlock*>> ComputationalRegions = identifyComputationalRegions(F);
  
  // Analyze dependencies between regions for potential fusion
  analyzeDependenciesBetweenRegions(SuitableLoops);
  
  // Extract kernels from loops
  bool Modified = false;
  
  for (Loop *L : SuitableLoops) {
    // Skip nested loops that are part of already extracted regions
    if (L->getParentLoop() && 
        std::find(SuitableLoops.begin(), SuitableLoops.end(), L->getParentLoop()) != SuitableLoops.end()) {
      continue;
    }
    
    // Score the loop for extraction
    float Score = scoreLoopForExtraction(L);
    LLVM_DEBUG(dbgs() << "  Loop extraction score: " << Score << "\n");
    
    // Only extract loops with score above threshold
    if (Score > 0.5f) {
      if (Function *KernelFunc = extractLoopToKernel(L)) {
        ExtractedKernels.push_back(KernelFunc);
        Modified = true;
      }
    }
  }
  
  // Extract kernels from other computational regions
  for (const std::vector<BasicBlock*> &Region : ComputationalRegions) {
    if (Function *KernelFunc = extractRegionToKernel(F, Region)) {
      ExtractedKernels.push_back(KernelFunc);
      Modified = true;
    }
  }
  
  return Modified;
}

std::vector<Loop*> GPUKernelExtractor::identifySuitableLoops(Function &F) {
  std::vector<Loop*> SuitableLoops;
  
  // Create a pattern analyzer for detecting suitable loops
  GPUPatternAnalyzer Analyzer(LI, DI, SE);
  
  // Check all loops in the function
  for (Loop *L : LI) {
    // Use recursive helper to check this loop and all its subloops
    std::function<void(Loop*)> CheckLoop = [&](Loop *CurLoop) {
      if (isSuitableForExtraction(CurLoop)) {
        SuitableLoops.push_back(CurLoop);
      }
      
      // Check subloops
      for (Loop *SubLoop : CurLoop->getSubLoops()) {
        CheckLoop(SubLoop);
      }
    };
    
    CheckLoop(L);
  }
  
  return SuitableLoops;
}

std::vector<std::vector<BasicBlock*>> GPUKernelExtractor::identifyComputationalRegions(Function &F) {
  std::vector<std::vector<BasicBlock*>> Regions;
  
  // For now, we'll use a simplified approach to identify computational regions
  // In a real implementation, this would use more sophisticated analysis
  
  // We'll use a simple heuristic: look for basic blocks with high computational density
  // that are not already part of extracted loops
  
  // First, create a set of all blocks in suitable loops
  std::set<BasicBlock*> LoopBlocks;
  for (Loop *L : LI) {
    for (BasicBlock *BB : L->getBlocks()) {
      LoopBlocks.insert(BB);
    }
  }
  
  // Look for connected blocks with high computational density
  std::set<BasicBlock*> Visited;
  for (BasicBlock &BB : F) {
    if (Visited.count(&BB) || LoopBlocks.count(&BB))
      continue;
      
    // Start a new region
    std::vector<BasicBlock*> Region;
    std::queue<BasicBlock*> Queue;
    Queue.push(&BB);
    Visited.insert(&BB);
    
    while (!Queue.empty()) {
      BasicBlock *CurBB = Queue.front();
      Queue.pop();
      
      // Check if this block is computationally intensive
      unsigned ComputeOps = 0;
      unsigned MemoryOps = 0;
      unsigned TotalOps = 0;
      
      for (Instruction &I : *CurBB) {
        TotalOps++;
        if (isa<LoadInst>(I) || isa<StoreInst>(I))
          MemoryOps++;
        else if (I.getOpcode() >= Instruction::Add && I.getOpcode() <= Instruction::Xor)
          ComputeOps++;
      }
      
      // If the block has sufficient computational density, add it to region
      if (TotalOps > 5 && ComputeOps > MemoryOps) {
        Region.push_back(CurBB);
        
        // Add its successors to the queue
        for (BasicBlock *Succ : successors(CurBB)) {
          if (!Visited.count(Succ) && !LoopBlocks.count(Succ)) {
            Queue.push(Succ);
            Visited.insert(Succ);
          }
        }
      }
    }
    
    // If we found a substantial region, add it
    if (Region.size() > 2) {
      Regions.push_back(Region);
    }
  }
  
  return Regions;
}

bool GPUKernelExtractor::isSuitableForExtraction(Loop *L) {
  GPUPatternAnalyzer Analyzer(LI, DI, SE);
  
  // Check if the loop is profitable for GPU execution
  if (!Analyzer.isLoopSuitableForGPU(L)) {
    LLVM_DEBUG(dbgs() << "  Loop not suitable for GPU based on analysis.\n");
    return false;
  }
  
  // Check the trip count
  const SCEV *BackedgeTakenCount = SE.getBackedgeTakenCount(L);
  if (isa<SCEVCouldNotCompute>(BackedgeTakenCount)) {
    LLVM_DEBUG(dbgs() << "  Could not compute trip count.\n");
    return false;
  }
  
  // If the trip count is known at compile time, check if it's large enough
  if (const SCEVConstant *ConstTC = dyn_cast<SCEVConstant>(BackedgeTakenCount)) {
    const APInt &TripCount = ConstTC->getValue()->getValue();
    if (TripCount.getLimitedValue() < 100) {
      LLVM_DEBUG(dbgs() << "  Trip count too small: " << TripCount << "\n");
      return false;
    }
  }
  
  // Check for regular memory access patterns
  if (!hasRegularMemoryAccess(L)) {
    LLVM_DEBUG(dbgs() << "  Irregular memory access patterns.\n");
    return false;
  }
  
  // Check code complexity
  unsigned InstructionCount = 0;
  for (BasicBlock *BB : L->getBlocks()) {
    InstructionCount += BB->size();
  }
  
  if (InstructionCount < 10) {
    LLVM_DEBUG(dbgs() << "  Loop body too small, not worth extracting.\n");
    return false;
  }
  
  // Check for unsupported operations
  for (BasicBlock *BB : L->getBlocks()) {
    for (Instruction &I : *BB) {
      // Check for operations that are typically not supported on GPUs
      if (isa<CallInst>(I) || isa<InvokeInst>(I)) {
        CallBase *CB = dyn_cast<CallBase>(&I);
        Function *Callee = CB->getCalledFunction();
        
        // Skip intrinsics
        if (Callee && Callee->isIntrinsic())
          continue;
          
        LLVM_DEBUG(dbgs() << "  Loop contains function calls, not extracting.\n");
        return false;
      }
    }
  }
  
  return true;
}

Function* GPUKernelExtractor::extractLoopToKernel(Loop *L) {
  LLVM_DEBUG(dbgs() << "Extracting loop to GPU kernel\n");
  
  // Determine the parallelization pattern for this loop
  ParallelizationPattern Pattern = determineParallelizationPattern(L);
  
  // Create a loop transformer for the actual transformation
  GPULoopTransformer Transformer(M, LI, SE, Runtime);
  
  // Transform the loop into a GPU kernel
  Function *KernelFunc = Transformer.transformLoopToGPUKernel(L, Pattern);
  
  return KernelFunc;
}

Function* GPUKernelExtractor::extractRegionToKernel(Function &F, const std::vector<BasicBlock*> &Region) {
  LLVM_DEBUG(dbgs() << "Extracting region to GPU kernel\n");
  
  // Simplified implementation - we'll use CodeExtractor
  DominatorTree DT(F);
  BlockFrequencyInfo BFI(F, DT, LI);
  
  // Using the CodeExtractor utility
  CodeExtractor CE(Region, &DT, false, &BFI, &LI);
  
  // Check if extraction is possible
  if (!CE.isEligible()) {
    LLVM_DEBUG(dbgs() << "  Region not eligible for extraction\n");
    return nullptr;
  }
  
  // Perform the extraction
  Function *ExtractedFunc = CE.extractCodeRegion();
  if (!ExtractedFunc) {
    LLVM_DEBUG(dbgs() << "  Failed to extract region\n");
    return nullptr;
  }
  
  // Add GPU attributes to the extracted function
  switch (Runtime) {
    case GPURuntime::CUDA:
      ExtractedFunc->addFnAttr("nvvm.annotations", "{\"kernel\", i32 1}");
      break;
    case GPURuntime::OpenCL:
      ExtractedFunc->addFnAttr("opencl.kernels", "kernel");
      break;
    case GPURuntime::SYCL:
      // SYCL uses templates instead of attributes
      break;
    case GPURuntime::HIP:
      ExtractedFunc->addFnAttr("amdgpu-flat-work-group-size", "1, 1024");
      break;
  }
  
  return ExtractedFunc;
}

void GPUKernelExtractor::analyzeDependenciesBetweenRegions(std::vector<Loop*> &SuitableLoops) {
  // Analyze dependencies between loops for potential fusion
  
  // For each pair of loops, check if they can be fused
  for (size_t i = 0; i < SuitableLoops.size(); ++i) {
    for (size_t j = i + 1; j < SuitableLoops.size(); ++j) {
      Loop *L1 = SuitableLoops[i];
      Loop *L2 = SuitableLoops[j];
      
      // Check if the loops are siblings (no nesting)
      if (L1->getParentLoop() != L2->getParentLoop())
        continue;
        
      // Check if the loops are adjacent in the CFG
      if (L1->getExitBlock() && L1->getExitBlock() == L2->getHeader()) {
        // Check for dependencies between the loops
        bool HasDependencies = false;
        
        // Get all memory operations in both loops
        std::vector<LoadInst*> Loads1, Loads2;
        std::vector<StoreInst*> Stores1, Stores2;
        
        for (BasicBlock *BB : L1->getBlocks()) {
          for (Instruction &I : *BB) {
            if (auto *Load = dyn_cast<LoadInst>(&I))
              Loads1.push_back(Load);
            else if (auto *Store = dyn_cast<StoreInst>(&I))
              Stores1.push_back(Store);
          }
        }
        
        for (BasicBlock *BB : L2->getBlocks()) {
          for (Instruction &I : *BB) {
            if (auto *Load = dyn_cast<LoadInst>(&I))
              Loads2.push_back(Load);
            else if (auto *Store = dyn_cast<StoreInst>(&I))
              Stores2.push_back(Store);
          }
        }
        
        // Check for RAW, WAR, WAW dependencies
        for (auto *Store1 : Stores1) {
          for (auto *Load2 : Loads2) {
            auto Result = DI.depends(Store1, Load2, true);
            if (Result && Result->isOrdered()) {
              HasDependencies = true;
              break;
            }
          }
          
          if (HasDependencies)
            break;
            
          for (auto *Store2 : Stores2) {
            auto Result = DI.depends(Store1, Store2, true);
            if (Result && Result->isOrdered()) {
              HasDependencies = true;
              break;
            }
          }
          
          if (HasDependencies)
            break;
        }
        
        for (auto *Load1 : Loads1) {
          for (auto *Store2 : Stores2) {
            auto Result = DI.depends(Store2, Load1, true);
            if (Result && Result->isOrdered()) {
              HasDependencies = true;
              break;
            }
          }
          
          if (HasDependencies)
            break;
        }
        
        // If no dependencies, we can potentially fuse the loops
        if (!HasDependencies) {
          LLVM_DEBUG(dbgs() << "Loops can be fused: " << i << " and " << j << "\n");
          // In a real implementation, we would mark these loops for fusion
        }
      }
    }
  }
}

ParallelizationPattern GPUKernelExtractor::determineParallelizationPattern(Loop *L) {
  GPUPatternAnalyzer Analyzer(LI, DI, SE);
  
  // Check for reduction patterns
  if (Analyzer.identifyReductionPatterns(L)) {
    return ParallelizationPattern::ReducePattern;
  }
  
  // Check for stencil patterns
  BasicBlock *Header = L->getHeader();
  for (Instruction &I : *Header) {
    if (auto *Phi = dyn_cast<PHINode>(&I)) {
      continue; // Skip PHI nodes
    }
    
    for (BasicBlock *BB : L->getBlocks()) {
      for (Instruction &I : *BB) {
        if (auto *Load = dyn_cast<LoadInst>(&I)) {
          // Look for array access with offset
          GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Load->getPointerOperand());
          if (GEP) {
            // Check if any of the indices is a constant offset other than 0
            for (unsigned i = 1; i < GEP->getNumOperands(); ++i) {
              if (ConstantInt *CI = dyn_cast<ConstantInt>(GEP->getOperand(i))) {
                if (CI->getSExtValue() != 0) {
                  return ParallelizationPattern::StencilPattern;
                }
              }
            }
          }
        }
      }
    }
  }
  
  // Check for histogram patterns
  for (BasicBlock *BB : L->getBlocks()) {
    for (Instruction &I : *BB) {
      if (auto *Store = dyn_cast<StoreInst>(&I)) {
        Value *Val = Store->getValueOperand();
        if (auto *BinOp = dyn_cast<BinaryOperator>(Val)) {
          if (BinOp->getOpcode() == Instruction::Add || 
              BinOp->getOpcode() == Instruction::Or) {
            // Check if one operand is a load from the same pointer
            for (unsigned i = 0; i < BinOp->getNumOperands(); ++i) {
              if (auto *Load = dyn_cast<LoadInst>(BinOp->getOperand(i))) {
                if (Load->getPointerOperand() == Store->getPointerOperand()) {
                  return ParallelizationPattern::HistogramPattern;
                }
              }
            }
          }
        }
      }
    }
  }
  
  // Default to map pattern
  return ParallelizationPattern::MapPattern;
}

bool GPUKernelExtractor::hasRegularMemoryAccess(Loop *L) {
  for (BasicBlock *BB : L->getBlocks()) {
    for (Instruction &I : *BB) {
      if (auto *Load = dyn_cast<LoadInst>(&I)) {
        Value *Ptr = Load->getPointerOperand();
        
        // Use SCEV to analyze the pointer
        const SCEV *PtrSCEV = SE.getSCEV(Ptr);
        
        // Check if it's an add recurrence (strided access)
        if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(PtrSCEV)) {
          if (AR->getLoop() == L) {
            // Check if it has a constant stride
            const SCEV *Stride = AR->getStepRecurrence(SE);
            if (!isa<SCEVConstant>(Stride)) {
              return false;
            }
          }
        } else {
          // Non-strided access might not be efficient
          return false;
        }
      } else if (auto *Store = dyn_cast<StoreInst>(&I)) {
        Value *Ptr = Store->getPointerOperand();
        
        // Use SCEV to analyze the pointer
        const SCEV *PtrSCEV = SE.getSCEV(Ptr);
        
        // Check if it's an add recurrence (strided access)
        if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(PtrSCEV)) {
          if (AR->getLoop() == L) {
            // Check if it has a constant stride
            const SCEV *Stride = AR->getStepRecurrence(SE);
            if (!isa<SCEVConstant>(Stride)) {
              return false;
            }
          }
        } else {
          // Non-strided access might not be efficient
          return false;
        }
      }
    }
  }
  
  return true;
}

bool GPUKernelExtractor::isCandidateForFusion(Loop *L) {
  // Check if this loop is a candidate for fusion with another loop
  
  // Simplified implementation - check for simple loops with few dependencies
  
  // Count the number of memory operations
  unsigned Loads = 0, Stores = 0;
  for (BasicBlock *BB : L->getBlocks()) {
    for (Instruction &I : *BB) {
      if (isa<LoadInst>(I))
        Loads++;
      else if (isa<StoreInst>(I))
        Stores++;
    }
  }
  
  // Loops with few memory operations are good candidates for fusion
  return (Loads + Stores) < 10;
}

Loop* GPUKernelExtractor::fuseLoops(const std::vector<Loop*> &Loops) {
  // Implement loop fusion for GPU kernels
  // This is a complex transformation - simplified placeholder
  
  // In a real implementation, this would merge the bodies of compatible loops
  
  return Loops[0]; // Return the first loop for now
}

float GPUKernelExtractor::scoreLoopForExtraction(Loop *L) {
  // Compute a score for how suitable this loop is for GPU extraction
  float Score = 0.0f;
  
  // Factor 1: Loop trip count
  const SCEV *BackedgeTakenCount = SE.getBackedgeTakenCount(L);
  if (!isa<SCEVCouldNotCompute>(BackedgeTakenCount)) {
    if (const SCEVConstant *ConstTC = dyn_cast<SCEVConstant>(BackedgeTakenCount)) {
      const APInt &TripCount = ConstTC->getValue()->getValue();
      uint64_t TC = TripCount.getLimitedValue();
      
      // Higher trip counts are better
      if (TC > 1000)
        Score += 0.3f;
      else if (TC > 100)
        Score += 0.2f;
      else if (TC > 10)
        Score += 0.1f;
    } else {
      // Variable trip count, potentially good
      Score += 0.15f;
    }
  }
  
  // Factor 2: Compute intensity
  unsigned ComputeOps = 0;
  unsigned MemoryOps = 0;
  
  for (BasicBlock *BB : L->getBlocks()) {
    for (Instruction &I : *BB) {
      if (isa<LoadInst>(I) || isa<StoreInst>(I))
        MemoryOps++;
      else if (I.getOpcode() >= Instruction::Add && I.getOpcode() <= Instruction::Xor)
        ComputeOps++;
    }
  }
  
  if (MemoryOps > 0) {
    float ComputeRatio = (float)ComputeOps / MemoryOps;
    if (ComputeRatio > 5.0f)
      Score += 0.3f;
    else if (ComputeRatio > 2.0f)
      Score += 0.2f;
    else if (ComputeRatio > 1.0f)
      Score += 0.1f;
  }
  
  // Factor 3: Loop structure
  if (L->getLoopLatch() && L->getExitBlock())
    Score += 0.1f; // Well-formed loop
  
  // Factor 4: Data parallelism
  GPUPatternAnalyzer Analyzer(LI, DI, SE);
  float ParallelizationPotential = Analyzer.calculateParallelizationPotential(L);
  Score += 0.3f * ParallelizationPotential;
  
  return Score;
}
