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
6. Transforms suitable loops into GPU kernels
7. Supports complex parallelization patterns (nested parallelism, wavefront, etc.)
8. Generates CUDA/OpenCL/SYCL/HIP code
9. Automatically extracts kernels from compute-intensive regions

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

### Using GPU Code Generation

The pass can also generate CUDA/OpenCL/SYCL/HIP code for identified kernels:

1. Set the desired GPU runtime in the pass:
   ```cpp
   GPURuntime Runtime = GPURuntime::CUDA; // Or OpenCL, SYCL, HIP
   ```

2. After running the pass, kernel source files will be generated (e.g., `kernel_name.cu` for CUDA).

3. Compile the generated kernels with the appropriate toolchain:
   - For CUDA: `nvcc kernel_name.cu -o kernel_name.o`
   - For OpenCL: Include with your OpenCL host code
   - For SYCL: Compile with a SYCL compiler like DPC++
   - For HIP: Compile with the HIP compiler

### Complex Parallelization Patterns

The pass identifies and optimizes several complex parallelization patterns:

1. Nested Parallelism - Uses 2D/3D thread blocks
2. Wavefront Pattern - Optimizes diagonal dependency traversal
3. Reduction Operations - Uses tree-based or atomic reduction
4. Stencil Computations - Optimizes shared memory usage
5. Histogram Operations - Uses atomic operations where needed

## Benchmarking

The pass performance can be evaluated by:

1. Running the original code and measuring execution time
2. Running the optimized code and measuring execution time
3. Comparing the results

## Future Improvements

- ~~Implement actual loop transformations for GPU execution~~ ✓
- ~~Add support for more complex parallelization patterns~~ ✓
- ~~Integrate with CUDA/OpenCL/SYCL code generation~~ ✓
- ~~Add automatic kernel extraction capability~~ ✓
- Improve cost model for GPU offloading decisions
- Add support for more diverse GPU architectures
- Implement automatic shared memory optimization
- Support more advanced synchronization primitives
- Add cooperative groups support for CUDA 
- Integrate with other LLVM optimization passes
- Support for heterogeneous execution (CPU+GPU)
