//===- GPUParallelizer.cpp - GPU Parallelization Pass ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------===//
//
// This file implements a pass that identifies parallelizable patterns in code
// and transforms them to be more suitable for GPU execution.
//
//===----------------------------------------------------------------===//

#include "GPUPatternAnalyzer.h" // Include the new analyzer
#include "llvm/Analysis/DependenceAnalysis.h" // Required by GPUPatternAnalyzer
#include "llvm/Analysis/ScalarEvolution.h" // Required by GPUPatternAnalyzer
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include <vector>

#define DEBUG_TYPE "gpu-parallelizer"

using namespace llvm;

namespace {

// Define a pass that inherits from FunctionPass
struct GPUParallelizer : public FunctionPass {
  static char ID;
  GPUParallelizer() : FunctionPass(ID) {}

  // getAnalysisUsage - This function declares which other passes are required by this pass
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<DependenceAnalysisWrapperPass>(); // Add DependenceAnalysis
    AU.addRequired<ScalarEvolutionWrapperPass>(); // Add ScalarEvolution
    AU.setPreservesCFG();
  }

  // runOnFunction - Implement the actual pass functionality
  bool runOnFunction(Function &F) override {
    LLVM_DEBUG(dbgs() << "GPUParallelizer: Processing function: " << F.getName() << "\n");
    errs() << "GPUParallelizer: Processing function: " << F.getName() << "\n";
    
    // Get analysis passes
    LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    DependenceInfo &DI = getAnalysis<DependenceAnalysisWrapperPass>().getDI();
    ScalarEvolution &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    
    // Create our pattern analyzer
    GPUPatternAnalyzer Analyzer(LI, DI, SE);
    
    // Analyze loops for parallelization opportunities
    bool Modified = false;
    for (Loop *L : LI) {
      Modified |= analyzeLoopForParallelization(L, Analyzer);
    }
    
    return Modified;
  }

  // Analyze a loop for parallelization potential
  bool analyzeLoopForParallelization(Loop *L, GPUPatternAnalyzer &Analyzer) {
    errs() << "  Analyzing loop for parallelization\n";
    
    // Use the GPUPatternAnalyzer to check if the loop is suitable
    if (Analyzer.isLoopSuitableForGPU(L)) {
      errs() << "  Loop is suitable for GPU, would transform for GPU\n";
      // Placeholder: Actual transformation logic would go here.
      // For example, outlining the loop body into a new function,
      // generating GPU kernel launch code, etc.
      // For now, we just identify candidates without transformation.
    } else {
      errs() << "  Loop is NOT suitable for GPU based on analysis.\n";
    }
    
    // Recursively analyze nested loops
    for (Loop *SubL : L->getSubLoops()) {
      analyzeLoopForParallelization(SubL, Analyzer);
    }
    
    return false; // No actual modifications yet, so return false
  }
};

} // end of anonymous namespace

// Pass ID registration
char GPUParallelizer::ID = 0;
static RegisterPass<GPUParallelizer> X("gpu-parallelize", "GPU Parallelization Pass",
                                       false /* Only looks at CFG */,
                                       false /* Analysis Pass */);

// Pass registration for opt tool
static RegisterStandardPasses RegisterGPUParallelizerPass(
    PassManagerBuilder::EP_EarlyAsPossible,
    [](const PassManagerBuilder &Builder, legacy::PassManagerBase &PM) {
      PM.add(new GPUParallelizer());
    });
