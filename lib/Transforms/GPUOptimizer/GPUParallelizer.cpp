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

#include "GPUPatternAnalyzer.h"
#include "GPULoopTransformer.h"
#include "GPUKernelExtractor.h"
#include "GPUCodeGen.h"
#include "GPUComplexPatternHandler.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
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
    
    // Create our analysis and transformation components
    GPUPatternAnalyzer Analyzer(LI, DI, SE);
    GPUKernelExtractor Extractor(*(F.getParent()), LI, DI, SE, GPURuntime::CUDA);
    GPUComplexPatternHandler ComplexHandler(*(F.getParent()), LI);
    
    // Step 1: Analyze loops for basic GPU parallelization
    bool Modified = false;
    for (Loop *L : LI) {
      Modified |= analyzeLoopForParallelization(L, Analyzer);
    }
    
    // Step 2: Automatically extract kernels from suitable regions
    if (Extractor.extractKernels(F)) {
      Modified = true;
      errs() << "Successfully extracted " << Extractor.getExtractedKernels().size() 
             << " GPU kernels from function " << F.getName() << "\n";
             
      // Generate code for each extracted kernel
      for (Function *KernelFunc : Extractor.getExtractedKernels()) {
        GPUCodeGen CodeGenerator(*(F.getParent()), GPURuntime::CUDA);
        if (CodeGenerator.generateGPUCode(KernelFunc)) {
          errs() << "Generated GPU code for kernel " << KernelFunc->getName() << "\n";
          
          // Write generated code to file (optional)
          std::string OutFileName = KernelFunc->getName().str() + ".cu";
          if (CodeGenerator.writeToFile(OutFileName)) {
            errs() << "Wrote generated code to " << OutFileName << "\n";
          }
        }
      }
    }
    
    // Step 3: Handle complex parallelization patterns
    auto ComplexPatterns = ComplexHandler.identifyComplexPatterns(F);
    for (auto &Pattern : ComplexPatterns) {
      Loop *L = Pattern.first;
      ComplexParallelPattern PatternType = Pattern.second;
      
      switch (PatternType) {
        case ComplexParallelPattern::NestedParallelism:
          if (L && !L->getSubLoops().empty()) {
            if (ComplexHandler.transformNestedParallelism(L, L->getSubLoops()[0])) {
              Modified = true;
              errs() << "Applied nested parallelism transformation\n";
            }
          }
          break;
          
        case ComplexParallelPattern::Wavefront:
          if (L && !L->getSubLoops().empty()) {
            if (ComplexHandler.transformWavefrontPattern(L, L->getSubLoops()[0])) {
              Modified = true;
              errs() << "Applied wavefront pattern transformation\n";
            }
          }
          break;
          
        case ComplexParallelPattern::Recursive:
          if (ComplexHandler.handleRecursivePattern(F)) {
            Modified = true;
            errs() << "Applied recursive pattern transformation\n";
          }
          break;
          
        default:
          // Other patterns
          break;
      }
    }
    
    return Modified;
  }
  // Analyze a loop for parallelization potential
  bool analyzeLoopForParallelization(Loop *L, GPUPatternAnalyzer &Analyzer) {
    errs() << "  Analyzing loop for parallelization\n";
    
    // Use the GPUPatternAnalyzer to check if the loop is suitable
    if (Analyzer.isLoopSuitableForGPU(L)) {
      errs() << "  Loop is suitable for GPU, transforming for GPU\n";
      
      // Create a loop transformer for the actual transformation
      GPULoopTransformer Transformer(*(L->getHeader()->getParent()->getParent()), 
                                   getAnalysis<LoopInfoWrapperPass>().getLoopInfo(),
                                   getAnalysis<ScalarEvolutionWrapperPass>().getSE(),
                                   GPURuntime::CUDA);
      
      // Determine parallelization pattern
      ParallelizationPattern Pattern = ParallelizationPattern::MapPattern;
      
      // Check for reduction patterns
      if (Analyzer.identifyReductionPatterns(L)) {
        Pattern = ParallelizationPattern::ReducePattern;
        errs() << "  Detected reduction pattern\n";
      }
      
      // Transform the loop into a GPU kernel
      Function *KernelFunc = Transformer.transformLoopToGPUKernel(L, Pattern);
      if (KernelFunc) {
        errs() << "  Successfully transformed loop to GPU kernel: " << KernelFunc->getName() << "\n";
        return true;
      } else {
        errs() << "  Failed to transform loop to GPU kernel\n";
      }
    } else {
      errs() << "  Loop is NOT suitable for GPU based on analysis.\n";
    }
    
    // Recursively analyze nested loops
    for (Loop *SubL : L->getSubLoops()) {
      if (analyzeLoopForParallelization(SubL, Analyzer)) {
        return true;
      }
    }
    
    return false;
  }
};

} // end of anonymous namespace

// Pass ID registration
char GPUParallelizer::ID = 0;
static RegisterPass<GPUParallelizer> X("gpu-parallelize", "GPU Parallelization Pass",
                                       false /* Only looks at CFG */,
                                       false /* Analysis Pass */);

// Pass registration for the new pass manager
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "GPUParallelizer", LLVM_VERSION_STRING,
    [](PassBuilder &PB) {
      PB.registerPipelineParsingCallback(
        [](StringRef Name, FunctionPassManager &FPM,
           ArrayRef<PassBuilder::PipelineElement>) {
          if (Name == "gpu-parallelize") {
            FPM.addPass(GPUParallelizer());
            return true;
          }
          return false;
        });
    }
  };
}
