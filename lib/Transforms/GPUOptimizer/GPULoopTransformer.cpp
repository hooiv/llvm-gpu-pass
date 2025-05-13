//===- GPULoopTransformer.cpp - Transform loops for GPU execution --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------===//
//
// This file implements the GPULoopTransformer class which transforms loops
// to make them suitable for GPU execution.
//
//===----------------------------------------------------------------===//

#include "GPULoopTransformer.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "gpu-loop-transformer"

Function *GPULoopTransformer::transformLoopToGPUKernel(Loop *L, ParallelizationPattern Pattern) {
  LLVM_DEBUG(dbgs() << "GPULoopTransformer: Transforming loop to GPU kernel\n");

  // Extract a kernel function from the loop body
  Function *KernelFunc = extractKernelFunction(L, Pattern);
  if (!KernelFunc)
    return nullptr;

  LLVM_DEBUG(dbgs() << "Successfully extracted kernel function: " << KernelFunc->getName() << "\n");

  // Apply pattern-specific optimizations
  optimizeForPattern(KernelFunc, Pattern);

  // Add runtime-specific attributes
  addRuntimeAttributes(KernelFunc);

  // Insert kernel launch code that replaces the original loop
  if (!insertKernelLaunchCode(L, KernelFunc)) {
    LLVM_DEBUG(dbgs() << "Failed to insert kernel launch code\n");
    return nullptr;
  }

  LLVM_DEBUG(dbgs() << "Successfully transformed loop to GPU kernel\n");
  return KernelFunc;
}

Function *GPULoopTransformer::extractKernelFunction(Loop *L, ParallelizationPattern Pattern) {
  BasicBlock *Header = L->getHeader();
  Function *OrigFunc = Header->getParent();
  
  // Create a name for the new kernel function
  std::string KernelName = OrigFunc->getName().str() + "_gpu_kernel";
  
  // Extract kernel parameters from the loop
  std::vector<Value*> Params = extractKernelParameters(L);
  
  // Create function type for the kernel
  std::vector<Type*> ParamTypes;
  for (Value *V : Params)
    ParamTypes.push_back(V->getType());
  
  FunctionType *FT = FunctionType::get(Type::getVoidTy(M.getContext()), ParamTypes, false);
  
  // Create the kernel function
  Function *KernelFunc = Function::Create(FT, GlobalValue::ExternalLinkage, KernelName, &M);
  
  // Set kernel function attributes based on the GPU runtime
  switch (Runtime) {
    case GPURuntime::CUDA:
      KernelFunc->addFnAttr("cuda-device");
      KernelFunc->addFnAttr("nvvm.annotations", "{\"kernel\", i32 1}");
      break;
    case GPURuntime::OpenCL:
      KernelFunc->addFnAttr("opencl.kernel");
      break;
    case GPURuntime::SYCL:
      // SYCL uses specific templates instead of attributes
      break;
    case GPURuntime::HIP:
      KernelFunc->addFnAttr("amdgpu-flat-work-group-size", "1, 1024");
      break;
  }
  
  // Create the entry basic block
  BasicBlock *Entry = BasicBlock::Create(M.getContext(), "entry", KernelFunc);
  IRBuilder<> Builder(Entry);
  
  // Map the parameter values to function arguments
  ValueToValueMapTy VMap;
  Function::arg_iterator AI = KernelFunc->arg_begin();
  for (Value *V : Params) {
    AI->setName(V->getName());
    VMap[V] = &*AI;
    ++AI;
  }
  
  // Insert thread indexing code at the start of the kernel
  insertThreadIndexing(KernelFunc, L, Builder);
  
  // Clone the loop body into the kernel
  SmallVector<BasicBlock*, 8> LoopBlocks;
  for (BasicBlock *BB : L->getBlocks())
    LoopBlocks.push_back(BB);
  
  // Clone all blocks from the loop
  for (BasicBlock *BB : LoopBlocks) {
    BasicBlock *NewBB = CloneBasicBlock(BB, VMap, ".kernel", KernelFunc);
    VMap[BB] = NewBB;
  }
  
  // Update PHI nodes and fix references in the cloned blocks
  for (BasicBlock &BB : *KernelFunc) {
    if (&BB == Entry)
      continue;
      
    for (Instruction &I : BB) {
      RemapInstruction(&I, VMap, RF_NoModuleLevelChanges);
    }
  }
  
  // Create branch from entry to the first basic block of the loop body
  BasicBlock *FirstLoopBlock = cast<BasicBlock>(VMap[L->getHeader()]);
  Builder.CreateBr(FirstLoopBlock);
  
  // Simplify the cloned code and correct control flow
  for (BasicBlock &BB : *KernelFunc) {
    if (isa<ReturnInst>(BB.getTerminator()))
      continue;
      
    // Replace loop exits with returns 
    if (isa<BranchInst>(BB.getTerminator())) {
      BranchInst *Br = cast<BranchInst>(BB.getTerminator());
      for (unsigned i = 0; i < Br->getNumSuccessors(); ++i) {
        BasicBlock *Succ = Br->getSuccessor(i);
        if (Succ->getParent() != KernelFunc) {
          // Replace branch to original function's block with return void
          IRBuilder<> Builder(&BB, std::prev(BB.end()));
          Builder.CreateRetVoid();
          Br->eraseFromParent();
          break;
        }
      }
    }
  }
  
  LLVM_DEBUG(dbgs() << "Kernel function extraction complete\n");
  return KernelFunc;
}

std::vector<Value*> GPULoopTransformer::extractKernelParameters(Loop *L) {
  std::vector<Value*> Params;
  std::set<Value*> ParamSet;
  
  // Find all memory references and scalar values used in the loop body
  for (BasicBlock *BB : L->getBlocks()) {
    for (Instruction &I : *BB) {
      // Add memory operations
      if (auto *Load = dyn_cast<LoadInst>(&I)) {
        Value *Ptr = Load->getPointerOperand();
        if (Instruction *PtrInst = dyn_cast<Instruction>(Ptr)) {
          if (!L->contains(PtrInst->getParent())) {
            ParamSet.insert(Ptr);
          }
        } else if (isa<Argument>(Ptr) || isa<GlobalValue>(Ptr)) {
          ParamSet.insert(Ptr);
        }
      } else if (auto *Store = dyn_cast<StoreInst>(&I)) {
        Value *Ptr = Store->getPointerOperand();
        if (Instruction *PtrInst = dyn_cast<Instruction>(Ptr)) {
          if (!L->contains(PtrInst->getParent())) {
            ParamSet.insert(Ptr);
          }
        } else if (isa<Argument>(Ptr) || isa<GlobalValue>(Ptr)) {
          ParamSet.insert(Ptr);
        }
        
        // Also consider stored value if it's from outside the loop
        Value *Val = Store->getValueOperand();
        if (Instruction *ValInst = dyn_cast<Instruction>(Val)) {
          if (!L->contains(ValInst->getParent())) {
            ParamSet.insert(Val);
          }
        } else if (!isa<Constant>(Val)) {
          ParamSet.insert(Val);
        }
      } else {
        // Add scalar operands from outside the loop
        for (unsigned i = 0; i < I.getNumOperands(); ++i) {
          Value *Op = I.getOperand(i);
          if (Instruction *OpInst = dyn_cast<Instruction>(Op)) {
            if (!L->contains(OpInst->getParent())) {
              ParamSet.insert(Op);
            }
          } else if (isa<Argument>(Op) && !isa<Constant>(Op)) {
            ParamSet.insert(Op);
          }
        }
      }
    }
  }
  
  // Convert to vector
  Params.assign(ParamSet.begin(), ParamSet.end());

  // Add the loop bounds if this is a canonical loop
  if (const SCEV *TripCount = SE.getBackedgeTakenCount(L)) {
    if (const SCEVConstant *ConstTripCount = dyn_cast<SCEVConstant>(TripCount)) {
      Type *Int32Ty = Type::getInt32Ty(M.getContext());
      // Add trip count as parameter
      Constant *TripCountVal = ConstantInt::get(Int32Ty, ConstTripCount->getValue()->getZExtValue());
      Params.push_back(TripCountVal);
    }
  }
  
  return Params;
}

void GPULoopTransformer::insertThreadIndexing(Function *F, Loop *L, IRBuilder<> &Builder) {
  // Create thread index variables based on the GPU runtime
  LLVMContext &Ctx = M.getContext();
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  
  // Get loop dimensions to determine grid size
  const SCEV *TripCount = SE.getBackedgeTakenCount(L);
  Value *TripCountVal = nullptr;
  
  if (const SCEVConstant *ConstTripCount = dyn_cast<SCEVConstant>(TripCount)) {
    TripCountVal = ConstantInt::get(Int32Ty, ConstTripCount->getValue()->getZExtValue());
  } else {
    // For variable trip count, we'll need a parameter
    TripCountVal = F->arg_begin(); // Assuming trip count is first arg
  }
  
  Value *ThreadIdx = nullptr;  switch (Runtime) {
    case GPURuntime::CUDA: {
      // Create functions for CUDA intrinsics
      FunctionType *Int32RetTy = FunctionType::get(Type::getInt32Ty(Ctx), false);
      
      // Define the CUDA intrinsics for getting thread and block indices
      Function *GetThreadIdxX = Function::Create(
          Int32RetTy, 
          GlobalValue::ExternalLinkage, 
          "llvm.nvvm.read.ptx.sreg.tid.x", 
          &M);
          
      Function *GetBlockIdxX = Function::Create(
          Int32RetTy, 
          GlobalValue::ExternalLinkage, 
          "llvm.nvvm.read.ptx.sreg.ctaid.x", 
          &M);
          
      Function *GetBlockDimX = Function::Create(
          Int32RetTy, 
          GlobalValue::ExternalLinkage, 
          "llvm.nvvm.read.ptx.sreg.ntid.x", 
          &M);
      
      Value *Tid = Builder.CreateCall(GetThreadIdxX);
      Value *Bid = Builder.CreateCall(GetBlockIdxX);
      Value *Bdim = Builder.CreateCall(GetBlockDimX);
      
      // Calculate global thread ID: threadIdx.x + blockIdx.x * blockDim.x
      Value *BlockOffset = Builder.CreateMul(Bid, Bdim);
      ThreadIdx = Builder.CreateAdd(Tid, BlockOffset, "global_tid");
      break;
    }    case GPURuntime::OpenCL: {
      // For OpenCL, use get_global_id(0)
      Function *GetGlobalId = Function::Create(
        FunctionType::get(Int32Ty, Int32Ty, false),
        GlobalValue::ExternalLinkage,
        "get_global_id",
        &M);
      
      Value *ZeroIndex = ConstantInt::get(Int32Ty, 0);
      ThreadIdx = Builder.CreateCall(GetGlobalId, ZeroIndex, "global_id");
      break;
    }
    case GPURuntime::SYCL: {
      // SYCL has specific templates and accessors - simplified version here
      // In a real implementation, you'd use the SYCL structures
      F->addFnAttr("sycl-module-id", "kernel_module");
      ThreadIdx = F->arg_begin(); // Assuming SYCL passes the ID as first arg
      break;
    }    case GPURuntime::HIP: {
      // HIP is similar to CUDA
      FunctionType *Int32RetTy = FunctionType::get(Type::getInt32Ty(Ctx), false);
      
      // Define HIP intrinsics for thread/block indices
      Function *GetThreadIdxX = Function::Create(
        Int32RetTy,
        GlobalValue::ExternalLinkage,
        "__hip_get_thread_idx_x",
        &M);
      Function *GetBlockIdxX = Function::Create(
        Int32RetTy,
        GlobalValue::ExternalLinkage,
        "__hip_get_block_idx_x",
        &M);
      Function *GetBlockDimX = Function::Create(
        Int32RetTy,
        GlobalValue::ExternalLinkage,
        "__hip_get_block_dim_x",
        &M);
      
      Value *Tid = Builder.CreateCall(GetThreadIdxX);
      Value *Bid = Builder.CreateCall(GetBlockIdxX);
      Value *Bdim = Builder.CreateCall(GetBlockDimX);
      
      Value *BlockOffset = Builder.CreateMul(Bid, Bdim);
      ThreadIdx = Builder.CreateAdd(Tid, BlockOffset, "global_tid");
      break;
    }
  }
  
  // Create a bounds check
  Value *InBounds = Builder.CreateICmpULT(ThreadIdx, TripCountVal, "in_bounds");
  BasicBlock *LoopEntry = cast<BasicBlock>(F->begin()->getNextNode());
  BasicBlock *ExitBlock = BasicBlock::Create(Ctx, "thread_out_of_bounds", F);
  
  // Early return if thread is out of bounds
  Builder.CreateCondBr(InBounds, LoopEntry, ExitBlock);
  
  // Set up the exit block
  Builder.SetInsertPoint(ExitBlock);
  Builder.CreateRetVoid();
  
  // Store thread ID and trip count for use in the kernel body
  new GlobalVariable(M, Int32Ty, false, GlobalValue::ExternalLinkage, 
                    nullptr, "thread_idx_var");
}

void GPULoopTransformer::optimizeForPattern(Function *F, ParallelizationPattern Pattern) {
  // Apply optimizations based on the pattern
  switch (Pattern) {
    case ParallelizationPattern::MapPattern:
      // Simple map pattern already handled by basic transformation
      break;
    case ParallelizationPattern::ReducePattern:
      // Need special handling for reductions
      // Typically use shared memory and parallel reduction algorithms
      // This would add atomic operations or tree-based reduction
      break;
    case ParallelizationPattern::StencilPattern:
      // For stencil patterns, add shared memory to cache neighboring elements
      break;
    case ParallelizationPattern::TransposePattern:
      // For matrix transpose, optimize memory access patterns
      break;
    case ParallelizationPattern::ScanPattern:
      // Implement parallel scan algorithm
      break;
    case ParallelizationPattern::HistogramPattern:
      // Add atomic operations for histogram updates
      break;
  }
}

void GPULoopTransformer::addRuntimeAttributes(Function *F) {
  // Add attributes specific to the GPU runtime
  switch (Runtime) {
    case GPURuntime::CUDA:
      F->addFnAttr("nvvm.annotations", "{\"kernel\", i32 1}");
      break;
    case GPURuntime::OpenCL:
      F->addFnAttr("opencl.kernels", "kernel");
      F->addFnAttr("reqd_work_group_size", "256");
      break;
    case GPURuntime::SYCL:
      // SYCL uses special templates rather than attributes
      break;
    case GPURuntime::HIP:
      F->addFnAttr("amdgpu-flat-work-group-size", "1, 1024");
      break;
  }
}

bool GPULoopTransformer::insertKernelLaunchCode(Loop *L, Function *KernelFunc) {
  BasicBlock *Preheader = L->getLoopPreheader();
  if (!Preheader) {
    LLVM_DEBUG(dbgs() << "Loop doesn't have a preheader\n");
    return false;
  }
  
  BasicBlock *Header = L->getHeader();
  BasicBlock *ExitBlock = L->getExitBlock();
  if (!ExitBlock) {
    LLVM_DEBUG(dbgs() << "Loop has multiple exits, not currently supported\n");
    return false;
  }
  
  LLVMContext &Ctx = M.getContext();
  IRBuilder<> Builder(Preheader->getTerminator());
  
  // Create a new basic block for the kernel launch
  BasicBlock *LaunchBlock = BasicBlock::Create(Ctx, "kernel_launch", 
                                              Header->getParent(),
                                              Header);
  
  // Redirect the preheader's terminator to the launch block
  Preheader->getTerminator()->setSuccessor(0, LaunchBlock);
  
  // Move to the launch block for code insertion
  Builder.SetInsertPoint(LaunchBlock);
  
  // Get the loop trip count
  const SCEV *TripCount = SE.getBackedgeTakenCount(L);
  Value *TripCountVal = nullptr;
  
  if (const SCEVConstant *ConstTripCount = dyn_cast<SCEVConstant>(TripCount)) {
    TripCountVal = ConstantInt::get(Type::getInt32Ty(Ctx), 
                                   ConstTripCount->getValue()->getZExtValue());
  } else {
    // For variable trip count, we need to emit the calculation
    SCEVExpander Expander(SE, M.getDataLayout(), "tripcount");
    TripCountVal = Expander.expandCodeFor(TripCount, Type::getInt32Ty(Ctx), 
                                         Builder.GetInsertPoint());
  }
  
  // Gather the kernel arguments
  std::vector<Value*> Args = extractKernelParameters(L);
  
  // Different runtime APIs have different launch syntax
  switch (Runtime) {
    case GPURuntime::CUDA: {
      // For CUDA, we need to set up grid and block dimensions
      // This is simplified - a real implementation would be more complex
      Type *Int32Ty = Type::getInt32Ty(Ctx);
      Value *BlockSize = ConstantInt::get(Int32Ty, 256);  // Default 256 threads per block
      
      // Calculate grid size: (N + 255) / 256 to handle non-multiples of 256
      Value *BlockSizeMinus1 = ConstantInt::get(Int32Ty, 255);
      Value *Numerator = Builder.CreateAdd(TripCountVal, BlockSizeMinus1);
      Value *GridSize = Builder.CreateUDiv(Numerator, BlockSize);
      
      // Create external declarations for CUDA runtime functions
      FunctionType *LaunchKernelTy = FunctionType::get(
          Type::getVoidTy(Ctx),
          {Type::getInt8PtrTy(Ctx), Int32Ty, Int32Ty, Type::getInt8PtrTy(Ctx)},
          true); // varargs for kernel args
      
      // Create or get the cudaLaunchKernel function
      Function *CudaLaunchKernel = Function::Create(
          LaunchKernelTy,
          GlobalValue::ExternalLinkage,
          "cudaLaunchKernel",
          &M);
      
      // Get function pointer to kernel
      Value *KernelPtr = Builder.CreateBitCast(KernelFunc, Type::getInt8PtrTy(Ctx));
      
      // Call cudaLaunchKernel with the kernel arguments
      std::vector<Value*> LaunchArgs = {KernelPtr, GridSize, BlockSize};
      
      // Create an array for kernel arguments
      ArrayType *ArgsArrayTy = ArrayType::get(Type::getInt8PtrTy(Ctx), Args.size());
      AllocaInst *ArgsArray = Builder.CreateAlloca(ArgsArrayTy);
      
      // Store each argument in the array
      for (unsigned i = 0; i < Args.size(); i++) {
        Value *ArgPtr = Builder.CreateBitCast(Args[i], Type::getInt8PtrTy(Ctx));
        Value *ArrayIdx = Builder.CreateConstGEP2_32(ArgsArrayTy, ArgsArray, 0, i);
        Builder.CreateStore(ArgPtr, ArrayIdx);
      }
      
      // Pass the array to cudaLaunchKernel
      LaunchArgs.push_back(Builder.CreateBitCast(ArgsArray, Type::getInt8PtrTy(Ctx)));
      
      Builder.CreateCall(CudaLaunchKernel, LaunchArgs);
      
      break;
    }
    case GPURuntime::OpenCL: {
      // For OpenCL, we'd use the OpenCL runtime API
      // This is simplified - a real implementation would be more complex
      Type *Int32Ty = Type::getInt32Ty(Ctx);
      Type *VoidPtrTy = Type::getInt8PtrTy(Ctx);
      
      // Create or get OpenCL runtime functions
      FunctionType *SetKernelArgTy = FunctionType::get(
          Int32Ty, {VoidPtrTy, Int32Ty, Int32Ty, VoidPtrTy}, false);
      Function *SetKernelArg = Function::Create(
          SetKernelArgTy,
          GlobalValue::ExternalLinkage,
          "clSetKernelArg",
          &M);
      
      FunctionType *EnqueueNDRangeKernelTy = FunctionType::get(
          Int32Ty, 
          {VoidPtrTy, VoidPtrTy, Int32Ty, VoidPtrTy, VoidPtrTy, Int32Ty, VoidPtrTy, VoidPtrTy},
          false);
      Function *EnqueueNDRangeKernel = Function::Create(
          EnqueueNDRangeKernelTy,
          GlobalValue::ExternalLinkage,
          "clEnqueueNDRangeKernel",
          &M);
      
      // Assume we have a kernel object and queue (would be created earlier)
      GlobalVariable *KernelObj = new GlobalVariable(
          M, VoidPtrTy, false, GlobalValue::ExternalLinkage, 
          nullptr, "cl_kernel_object");
      GlobalVariable *QueueObj = new GlobalVariable(
          M, VoidPtrTy, false, GlobalValue::ExternalLinkage, 
          nullptr, "cl_command_queue");
      
      // Set kernel arguments
      for (unsigned i = 0; i < Args.size(); i++) {
        Value *IdxVal = ConstantInt::get(Int32Ty, i);
        Value *SizeVal = ConstantInt::get(Int32Ty, M.getDataLayout().getTypeAllocSize(Args[i]->getType()));
        Value *ArgPtr = Builder.CreateBitCast(Args[i], VoidPtrTy);
        Builder.CreateCall(SetKernelArg, {KernelObj, IdxVal, SizeVal, ArgPtr});
      }
      
      // Calculate work sizes
      Value *WorkDim = ConstantInt::get(Int32Ty, 1);  // 1D kernel
      
      // Local work size (block size in CUDA terms)
      Value *LocalWorkSize = Builder.CreateAlloca(Int32Ty);
      Builder.CreateStore(ConstantInt::get(Int32Ty, 256), LocalWorkSize);
      
      // Global work size (grid size * block size in CUDA terms)
      Value *GlobalWorkSize = Builder.CreateAlloca(Int32Ty);
      Value *RoundedSize = Builder.CreateAdd(TripCountVal, 
                                            ConstantInt::get(Int32Ty, 255));
      Value *BlockCount = Builder.CreateUDiv(RoundedSize, 
                                           ConstantInt::get(Int32Ty, 256));
      Value *TotalSize = Builder.CreateMul(BlockCount, 
                                          ConstantInt::get(Int32Ty, 256));
      Builder.CreateStore(TotalSize, GlobalWorkSize);
      
      // Launch the kernel
      Value *NullPtr = ConstantPointerNull::get(VoidPtrTy);
      Value *ZeroEvents = ConstantInt::get(Int32Ty, 0);
      Builder.CreateCall(EnqueueNDRangeKernel, 
                        {QueueObj, KernelObj, WorkDim, NullPtr, 
                         GlobalWorkSize, LocalWorkSize, ZeroEvents, 
                         NullPtr, NullPtr});
                         
      break;
    }
    case GPURuntime::SYCL: {
      // SYCL uses C++ templates - this would typically be at source level
      // We'll just add placeholder code here
      break;
    }
    case GPURuntime::HIP: {
      // HIP is similar to CUDA
      // Similar implementation to CUDA with HIP API functions
      break;
    }
  }
  
  // Add a branch to the loop exit block
  Builder.CreateBr(ExitBlock);
  
  // Now we need to delete the original loop
  // This is simplified - a real implementation would handle complex CFGs
  for (BasicBlock *BB : L->getBlocks()) {
    if (BB != Header) {
      BB->dropAllReferences();
    }
  }
  
  for (BasicBlock *BB : L->getBlocks()) {
    if (BB != Header) {
      BB->eraseFromParent();
    }
  }
  
  // Update the header to unconditionally branch to the exit
  Header->dropAllReferences();
  Builder.SetInsertPoint(Header);
  Builder.CreateBr(ExitBlock);
  
  return true;
}

bool GPULoopTransformer::replaceLoopWithKernelLaunch(Loop *L, Function *KernelFunc) {
  // This functionality is included in insertKernelLaunchCode
  return true;
}

void GPULoopTransformer::transformReductionPattern(Function *F, Loop *L) {
  // Implement reduction transformation
  // This would add code to perform a parallel reduction
  // For example, using shared memory and tree-based reduction
}

void GPULoopTransformer::transformStencilPattern(Function *F, Loop *L) {
  // Implement stencil pattern transformation
  // This would add shared memory for caching neighboring elements
}
