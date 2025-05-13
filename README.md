# LLVM GPU-Specific Optimization Pass

This project implements a custom LLVM optimization pass that targets GPU architectures by identifying and transforming parallelizable code patterns.

## Project Structure

- `lib/Transforms/GPUOptimizer/`: Contains the implementation of the GPU parallelization pass
- `test_sample.c`: Sample C code with different patterns to test the optimization pass

## Features

The GPU Parallelizer Pass does the following:

1. Identifies loops that are potential candidates for GPU execution
2. Analyzes loop-carried dependencies
3. Examines memory access patterns for GPU compatibility
4. Evaluates compute-to-memory ratio
5. Determines if loops have sufficient iterations to benefit from GPU execution

## Building the Project

### Prerequisites

- LLVM development environment (LLVM 12.0.0 or newer recommended)
- CMake 3.13.4 or newer
- C++ compiler supporting C++14 or newer

### Build Instructions

1. Clone or download the LLVM source code:
   ```
   git clone https://github.com/llvm/llvm-project.git
   cd llvm-project
   ```

2. Create a build directory:
   ```
   mkdir build
   cd build
   ```

3. Configure the build with CMake:
   ```
   cmake -DLLVM_ENABLE_PROJECTS=clang -DLLVM_TARGETS_TO_BUILD=X86;NVPTX -DLLVM_EXTERNAL_PROJECTS=GPUOptimizer -DLLVM_EXTERNAL_GPUOPTIMIZER_SOURCE_DIR=/path/to/this/project -G "Ninja" ../llvm
   ```

4. Build LLVM with your custom pass:
   ```
   ninja
   ```

## Using the Pass

To use the pass on a C/C++ file:

1. Generate LLVM IR:
   ```
   clang -O0 -emit-llvm -c test_sample.c -o test_sample.bc
   ```

2. Run the pass using opt:
   ```
   opt -load /path/to/build/lib/LLVMGPUOptimizer.so -gpu-parallelize test_sample.bc -o optimized.bc
   ```

3. Generate machine code:
   ```
   llc optimized.bc -o optimized.s
   ```

4. Compile to executable:
   ```
   clang optimized.s -o optimized
   ```

## Benchmarking

The pass performance can be evaluated by:

1. Running the original code and measuring execution time
2. Running the optimized code and measuring execution time
3. Comparing the results

## Future Improvements

- Implement actual loop transformations for GPU execution
- Add support for more complex parallelization patterns
- Integrate with CUDA/OpenCL/SYCL code generation
- Add automatic kernel extraction capability
