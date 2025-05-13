add_llvm_library(LLVMGPUOptimizer MODULE
  GPUParallelizer.cpp
  GPUPatternAnalyzer.cpp
  GPULoopTransformer.cpp
  GPUKernelExtractor.cpp
  GPUCodeGen.cpp
  GPUComplexPatternHandler.cpp

  DEPENDS
  intrinsics_gen

  PLUGIN_TOOL
  opt
  )
