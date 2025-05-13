//===- GPUOptimizerPass.cpp - Optimize code for GPU execution --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------===//
//
// This file implements a pass to optimize code for GPU execution,
// including determining when code should be offloaded to a GPU,
// transforming code for better GPU performance, and managing
// heterogeneous execution.
//
//===----------------------------------------------------------------===//

#include "GPUPatternAnalyzer.h"
#include "GPUSyncPrimitives.h"
#include "GPUPassIntegration.h"
#include "GPUHeterogeneousSupport.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

#define DEBUG_TYPE "gpu-optimizer"

STATISTIC(LoopsAnalyzed, "Number of loops analyzed for GPU execution");
STATISTIC(LoopsOffloaded, "Number of loops offloaded to GPU");
STATISTIC(RegionsOptimized, "Number of regions optimized for GPU execution");
STATISTIC(SharedMemoryOpts, "Number of shared memory optimizations applied");
STATISTIC(SyncOptimizations, "Number of synchronization optimizations applied");
STATISTIC(HeterogeneousRegions, "Number of regions with heterogeneous execution");

namespace {

// New Pass Manager implementation
struct GPUOptimizerPass : public PassInfoMixin<GPUOptimizerPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  
  // This is needed by the LLVM pass infrastructure
  static bool isRequired() { return true; }
};

// Legacy Pass Manager implementation
struct LegacyGPUOptimizerPass : public FunctionPass {
  static char ID;
  
  LegacyGPUOptimizerPass() : FunctionPass(ID) {}
  
  bool runOnFunction(Function &F) override;
  
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  
  StringRef getPassName() const override { 
    return "GPU Optimizer Pass";
  }
};

} // end anonymous namespace

// Implementation for new Pass Manager
PreservedAnalyses GPUOptimizerPass::run(Function &F, FunctionAnalysisManager &AM) {
  // Get necessary analyses
  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  auto &DI = AM.getResult<DependenceAnalysis>(F);
  auto &TTI = AM.getResult<TargetIRAnalysis>(F);
  
  // Create our pattern analyzer
  GPUPatternAnalyzer GPA(LI, DI, SE);
  
  // Create synchronization handler
  GPUSynchronizationHandler SyncHandler(F, LI, GPA);
  
  // Create heterogeneous execution support
  Module *M = F.getParent();
  GPUHeterogeneousSupport HeteroSupport(*M, LI, DI, SE, GPA);
  
  bool Modified = false;
  
  // Step 1: Analyze loops for GPU execution potential
  LoopsAnalyzed = 0;
  for (Loop *L : LI) {
    LoopsAnalyzed++;
    
    // Calculate comprehensive cost for GPU offloading
    float OffloadCost = GPA.calculateGPUOffloadingCost(L);
    
    LLVM_DEBUG(dbgs() << "Loop at " << L->getHeader()->getName()
                     << " has GPU offloading cost: " << OffloadCost << "\n");
                     
    // If the cost is above threshold, consider for GPU offloading
    if (OffloadCost > 0.6) {
      // Analyze for shared memory optimization
      if (GPA.analyzeSharedMemoryOptimizationPotential(L)) {
        LLVM_DEBUG(dbgs() << "Loop can benefit from shared memory optimization\n");
        SharedMemoryOpts++;
      }
      
      RegionsOptimized++;
    }
  }
  
  // Step 2: Analyze synchronization needs
  SyncHandler.analyzeSynchronizationPoints();
  if (SyncHandler.getSyncPoints().size() > 0) {
    Modified |= SyncHandler.insertSynchronizationPrimitives();
    SyncOptimizations += SyncHandler.getSyncPoints().size();
  }
  
  // Step 3: Identify offloadable regions for heterogeneous execution
  HeteroSupport.identifyOffloadableRegions(F);
  
  const std::vector<OffloadableRegion> &Regions = HeteroSupport.getOffloadableRegions();
  HeterogeneousRegions += Regions.size();
  
  if (!Regions.empty()) {
    // Process regions for heterogeneous execution
    for (const OffloadableRegion &Region : Regions) {
      if (Region.PreferredMode != ExecutionMode::CPU_ONLY) {
        Modified |= HeteroSupport.createHeterogeneousVersions(Region);
        LoopsOffloaded++;
      }
    }
  }
  
  if (Modified) {
    // We modified the IR, so invalidate appropriate analyses
    return PreservedAnalyses::none();
  }
  
  return PreservedAnalyses::all();
}

// Implementation for legacy Pass Manager
bool LegacyGPUOptimizerPass::runOnFunction(Function &F) {
  // Get necessary analyses
  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  auto &DI = getAnalysis<DependenceAnalysisWrapperPass>().getDI();
  auto &TTI = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
  
  // Create our pattern analyzer
  GPUPatternAnalyzer GPA(LI, DI, SE);
  
  // Create synchronization handler
  GPUSynchronizationHandler SyncHandler(F, LI, GPA);
  
  // Create heterogeneous execution support
  Module *M = F.getParent();
  GPUHeterogeneousSupport HeteroSupport(*M, LI, DI, SE, GPA);
  
  bool Modified = false;
  
  // Step 1: Analyze loops for GPU execution potential
  LoopsAnalyzed = 0;
  for (Loop *L : LI) {
    LoopsAnalyzed++;
    
    // Calculate comprehensive cost for GPU offloading
    float OffloadCost = GPA.calculateGPUOffloadingCost(L);
    
    LLVM_DEBUG(dbgs() << "Loop at " << L->getHeader()->getName()
                     << " has GPU offloading cost: " << OffloadCost << "\n");
                     
    // If the cost is above threshold, consider for GPU offloading
    if (OffloadCost > 0.6) {
      // Analyze for shared memory optimization
      if (GPA.analyzeSharedMemoryOptimizationPotential(L)) {
        LLVM_DEBUG(dbgs() << "Loop can benefit from shared memory optimization\n");
        SharedMemoryOpts++;
      }
      
      RegionsOptimized++;
    }
  }
  
  // Step 2: Analyze synchronization needs
  SyncHandler.analyzeSynchronizationPoints();
  if (SyncHandler.getSyncPoints().size() > 0) {
    Modified |= SyncHandler.insertSynchronizationPrimitives();
    SyncOptimizations += SyncHandler.getSyncPoints().size();
  }
  
  // Step 3: Identify offloadable regions for heterogeneous execution
  HeteroSupport.identifyOffloadableRegions(F);
  
  const std::vector<OffloadableRegion> &Regions = HeteroSupport.getOffloadableRegions();
  HeterogeneousRegions += Regions.size();
  
  if (!Regions.empty()) {
    // Process regions for heterogeneous execution
    for (const OffloadableRegion &Region : Regions) {
      if (Region.PreferredMode != ExecutionMode::CPU_ONLY) {
        Modified |= HeteroSupport.createHeterogeneousVersions(Region);
        LoopsOffloaded++;
      }
    }
  }
  
  return Modified;
}

void LegacyGPUOptimizerPass::getAnalysisUsage(AnalysisUsage &AU) const {
  // We require these analyses
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequired<ScalarEvolutionWrapperPass>();
  AU.addRequired<DependenceAnalysisWrapperPass>();
  AU.addRequired<TargetTransformInfoWrapperPass>();
  AU.addRequired<DominatorTreeWrapperPass>();
  
  // We preserve these analyses
  AU.addPreserved<LoopInfoWrapperPass>();
  AU.addPreserved<DominatorTreeWrapperPass>();
}

char LegacyGPUOptimizerPass::ID = 0;
static RegisterPass<LegacyGPUOptimizerPass> X("gpu-optimizer", "GPU Optimizer Pass",
                                             false /* Only looks at CFG */,
                                             false /* Analysis Pass */);

// New pass manager registration
llvm::PassPluginLibraryInfo getGPUOptimizerPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "GPUOptimizer", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "gpu-optimizer") {
                    FPM.addPass(GPUOptimizerPass());
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getGPUOptimizerPassPluginInfo();
}
