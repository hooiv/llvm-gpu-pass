add_llvm_library(LLVMGPUOptimizer MODULE
  GPUParallelizer.cpp
  GPUPatternAnalyzer.cpp

  DEPENDS
  intrinsics_gen

  PLUGIN_TOOL
  opt
  )
