# LLVM GPU-Specific Optimization Pass

This project implements a custom LLVM optimization pass that targets GPU architectures by identifying and transforming parallelizable code patterns, with extensive optimizations for better GPU performance.

## Project Structure

- `lib/Transforms/GPUOptimizer/`: Contains the implementation of the GPU optimization pass
- `test_samples/`: Sample C code with different patterns to test the optimization pass
- `build_and_test.ps1`: PowerShell script to build and test the pass automatically

## Features

The GPU Optimizer Pass implements the following key features:

### 1. Improved Cost Model for GPU Offloading Decisions
- Architecture-specific cost models for major GPU vendors (NVIDIA, AMD, Intel)
- Comprehensive analysis of computational intensity, parallelism, memory access patterns
- Data transfer overhead estimation
- Trip count and loop structure analysis
- Weighted decision model that adapts based on the target architecture

### 2. Support for Diverse GPU Architectures
- Optimizations tailored for:
  - NVIDIA: Ampere, Turing, Volta, Pascal, Maxwell
  - AMD: RDNA2, RDNA, CDNA2, CDNA, Vega
  - Intel: Xe HPC, Xe HPG, Xe LP
- Architecture detection via environment variables and platform queries
- Architecture-specific parameters for cost models
- Hardware-specific optimizations based on capabilities

### 3. Automatic Shared Memory Optimization
- Detection of code patterns that can benefit from shared memory
- Analysis of array access patterns for tiling opportunities
- Support for stencil patterns and reduction operations
- Memory requirement estimation to ensure shared memory constraints are met
- Tiling transformations based on memory access patterns

### 4. Advanced Synchronization Primitives
- Block-level synchronization (`__syncthreads()` in CUDA, `barrier()` in OpenCL)
- Warp-level synchronization (`__syncwarp()` in CUDA)
- Memory fences for different memory scopes
- Atomic operations for concurrent updates
- Cooperative groups for more flexible synchronization
- Race condition detection and resolution

### 5. Cooperative Groups Support (for CUDA)
- Support for grid-wide synchronization
- Thread block clusters for multi-block cooperation
- Subgroup synchronization with various sizes
- Dynamic group formation based on workload
- More flexible synchronization models than traditional barriers

### 6. Integration with Other LLVM Optimization Passes
- Coordination with loop optimization passes
- Integration with scalar optimization passes
- Interaction with vectorization passes
- Support for both new and legacy pass managers
- Pass pipeline ordering optimization

### 7. Support for Heterogeneous Execution (CPU+GPU)
- Region identification for offloading
- Cost analysis for CPU vs. GPU execution
- Support for runtime dispatch decisions
- Data transfer optimization between CPU and GPU
- Verification of correctness across devices
- Hybrid execution strategies

### 8. Complex Parallelization Patterns
- Nested Parallelism - Uses 2D/3D thread blocks
- Wavefront Pattern - Optimizes diagonal dependency traversal
- Reduction Operations - Uses tree-based or atomic reduction
- Stencil Computations - Optimizes shared memory usage
- Histogram Operations - Uses atomic operations where needed
- Pipeline patterns for producer-consumer workflows
- Task parallelism and stream parallelism patterns

### 9. Automatic Kernel Extraction
- Automatic identification of GPU-suitable code regions
- Extraction of loops with sufficient computational density
- Identification and handling of reduction patterns
- Analysis of data dependencies between potential kernels
- Scoring of regions based on GPU suitability metrics

## Building the Project

### Prerequisites

- LLVM development environment (LLVM 12.0.0 or newer recommended)
- CMake 3.13.4 or newer
- C++ compiler supporting C++14 or newer
- Ninja build system (recommended)

### Build Instructions

#### Option 1: Using the build script (recommended)

The easiest way to build and test the project is to use the provided PowerShell script:

```powershell
.\build_and_test.ps1
```

This will:
1. Build the GPU optimizer pass
2. Compile the test samples to LLVM IR
3. Run the pass on the test IR
4. Compile the optimized IR to an executable
5. Run the optimized executable

#### Option 2: Manual build

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

### Basic Usage

To use the pass on a C/C++ file:

1. Generate LLVM IR:
   ```
   clang -O0 -emit-llvm -c test_sample.c -o test_sample.bc
   ```

2. Run the pass using opt:
   ```
   opt -load-pass-plugin=build/lib/LLVMGPUOptimizer.dll -passes="gpu-optimizer" test_sample.bc -o optimized.bc
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

The pass can generate CUDA/OpenCL/SYCL/HIP code for identified kernels:

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

## Testing

The project includes sample code for testing in the `test_samples/` directory:

- `comprehensive_test.c`: Showcases various patterns that the optimizer can handle
- `complex_patterns.c`: Demonstrates advanced patterns including:
  - Matrix operations suitable for GPU
  - Reduction patterns
  - Stencil computations
  - Histogram operations
  - Nested parallelism
  - Wavefront patterns
  - Recursive functions

## Benchmarking

The pass performance can be evaluated by:

1. Running the original code and measuring execution time
2. Running the optimized code and measuring execution time
3. Comparing the results

## Future Improvements

Though all major features have been implemented, there are still areas for future improvement:

- Refine the cost model for offloading decisions
- Add more architecture-specific optimizations for newer GPUs
- Enhance shared memory optimization techniques
- Improve error handling in code generation
- Develop more sophisticated heuristics for kernel extraction
- Support for more specialized GPU computation patterns
