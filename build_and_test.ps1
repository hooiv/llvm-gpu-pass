# Build and test script for LLVM GPU Optimizer pass
# Run from the root of the project

# Ensure test_samples directory exists
if (-not (Test-Path -Path "test_samples")) {
    New-Item -Path "test_samples" -ItemType Directory
}

# Set paths
$LLVM_BUILD = Join-Path (Get-Location) "llvm-project\build"
$OPT = Join-Path $LLVM_BUILD "bin\opt.exe"
$CLANG = Join-Path $LLVM_BUILD "bin\clang.exe"
$LIB_DIR = Join-Path (Get-Location) "lib\Transforms\GPUOptimizer"

# Build the pass
Write-Host "Building GPU Optimizer pass..." -ForegroundColor Green
if (-not (Test-Path -Path "build")) {
    New-Item -Path "build" -ItemType Directory
}
Set-Location build
cmake -G "Ninja" ..
ninja
Set-Location ..

# Compile the test sample to IR
Write-Host "Compiling test sample to LLVM IR..." -ForegroundColor Green
& $CLANG -O0 -S -emit-llvm -o test_samples/comprehensive_test.ll test_samples/comprehensive_test.c

# Run the pass on the test IR
Write-Host "Running GPU Optimizer pass..." -ForegroundColor Green
& $OPT -load-pass-plugin=build/lib/LLVMGPUOptimizer.dll -passes="gpu-optimizer" test_samples/comprehensive_test.ll -o test_samples/comprehensive_test.opt.ll

# Compile optimized IR to executable
Write-Host "Compiling optimized IR to executable..." -ForegroundColor Green
& $CLANG test_samples/comprehensive_test.opt.ll -o test_samples/comprehensive_test.exe

# Run the executable
Write-Host "Running optimized executable..." -ForegroundColor Green
& test_samples/comprehensive_test.exe

Write-Host "All done!" -ForegroundColor Green
