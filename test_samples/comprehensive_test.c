#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// Matrix sizes for testing
#define SIZE_SMALL 64
#define SIZE_MEDIUM 512
#define SIZE_LARGE 2048

// =============================================================================
// Test Case 1: Standard Matrix Multiplication - Perfect for GPU
// - Regular access patterns
// - High arithmetic intensity
// - Good parallelism
// =============================================================================
void matrix_multiply(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// =============================================================================
// Test Case 2: Stencil Operation - Benefits from shared memory optimization
// - Regular access pattern with neighborhood access
// - Good for tiling with shared memory
// =============================================================================
void stencil_2d(float* input, float* output, int width, int height) {
    // 2D stencil with 5-point star pattern
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            // Apply 5-point stencil
            float result = 0.2f * (
                input[y * width + x] +          // Center
                input[(y-1) * width + x] +      // North
                input[(y+1) * width + x] +      // South
                input[y * width + (x-1)] +      // West
                input[y * width + (x+1)]        // East
            );
            output[y * width + x] = result;
        }
    }
}

// =============================================================================
// Test Case 3: Reduction - Needs atomic operations or parallel reduction
// - Sequential accumulation
// - Benefits from cooperative groups
// =============================================================================
float vector_reduction_sum(float* vec, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += vec[i];
    }
    return sum;
}

float vector_reduction_max(float* vec, int N) {
    float max_val = vec[0]; 
    for (int i = 1; i < N; i++) {
        if (vec[i] > max_val) {
            max_val = vec[i];
        }
    }
    return max_val;
}

// =============================================================================
// Test Case 4: Scan (Prefix Sum) - Requires thread synchronization
// - Sequential operation with dependence
// - Needs advanced synchronization
// =============================================================================
void inclusive_scan(float* input, float* output, int N) {
    output[0] = input[0];
    for (int i = 1; i < N; i++) {
        output[i] = output[i-1] + input[i];
    }
}

// =============================================================================
// Test Case 5: Wave Propagation - Complex pattern with data dependencies
// - Temporal locality
// - Multiple synchronization points needed
// - Benefits from asynchronous execution
// =============================================================================
void wave_propagation(float* current, float* previous, float* next, 
                     int width, int height, float dt, float dx, float c) {
    float coef = (c * c * dt * dt) / (dx * dx);
    
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * width + x;
            // Wave equation (simplified): 
            // next = 2*current - previous + c^2 * dt^2 * (spatial Laplacian of current)
            next[idx] = 2.0f * current[idx] - previous[idx] + 
                        coef * (
                            current[(y-1)*width + x] + 
                            current[(y+1)*width + x] + 
                            current[y*width + (x-1)] + 
                            current[y*width + (x+1)] - 
                            4.0f * current[idx]
                        );
        }
    }
}

// =============================================================================
// Test Case 6: Heterogeneous Mix - Tests CPU/GPU decision logic
// - Contains both GPU-suitable and CPU-suitable sections
// - Tests heterogeneous execution support
// =============================================================================
void heterogeneous_workload(float* A, float* B, float* C, int N, float threshold) {
    // Complex conditional logic (better on CPU)
    for (int i = 0; i < N; i++) {
        if (A[i] > threshold) {
            B[i] = sinf(A[i]) * cosf(A[i]) / sqrtf(fabs(A[i]));
        } else if (A[i] < -threshold) {
            B[i] = logf(fabs(A[i])) + expf(-A[i] * A[i]);
        } else {
            B[i] = A[i] * A[i] * A[i];
        }
    }
    
    // Regular computation (better on GPU)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 16; k++) {
                sum += B[(i+k) % N] * B[(j+k) % N];
            }
            C[i * N + j] = sum;
        }
    }
}

// =============================================================================
// Test Case 7: Matrix Transpose - Memory access pattern optimization
// - Non-coalesced access pattern
// - Benefits from shared memory and memory layout changes
// =============================================================================
void matrix_transpose(float* input, float* output, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            output[x * height + y] = input[y * width + x];
        }
    }
}

// =============================================================================
// Test Case 8: Histogram - Atomic operations required
// - Random access pattern
// - Race conditions without atomic operations
// =============================================================================
void histogram(unsigned char* image, int* hist, int width, int height) {
    // Clear histogram
    for (int i = 0; i < 256; i++) {
        hist[i] = 0;
    }
    
    // Compute histogram
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            unsigned char pixel = image[y * width + x];
            hist[pixel]++;
        }
    }
}

// =============================================================================
// Main function to execute test cases
// =============================================================================
int main() {
    printf("LLVM GPU Optimizer Test Suite\n");
    printf("=============================\n\n");
    
    // Allocate and initialize test data
    int N = SIZE_MEDIUM;
    float *A = (float*)malloc(N * N * sizeof(float));
    float *B = (float*)malloc(N * N * sizeof(float));
    float *C = (float*)malloc(N * N * sizeof(float));
    float *vec = (float*)malloc(N * sizeof(float));
    unsigned char *image = (unsigned char*)malloc(N * N * sizeof(unsigned char));
    int *hist = (int*)malloc(256 * sizeof(int));
    
    // Initialize data
    srand(time(NULL));
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
        if (i < N) vec[i] = (float)rand() / RAND_MAX;
        image[i] = rand() % 256;
    }
    
    // Measure execution time
    clock_t start, end;
    double cpu_time_used;
    
    // Test Case 1: Matrix Multiplication
    printf("Running Test Case 1: Matrix Multiplication\n");
    start = clock();
    matrix_multiply(A, B, C, N);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("  Time: %f seconds\n\n", cpu_time_used);
    
    // Test Case 2: Stencil Operation
    printf("Running Test Case 2: Stencil Operation\n");
    start = clock();
    stencil_2d(A, C, N, N);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("  Time: %f seconds\n\n", cpu_time_used);
    
    // Test Case 3: Reduction
    printf("Running Test Case 3: Reduction Operations\n");
    start = clock();
    float sum = vector_reduction_sum(vec, N);
    float max_val = vector_reduction_max(vec, N);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("  Time: %f seconds\n", cpu_time_used);
    printf("  Sum: %f, Max: %f\n\n", sum, max_val);
    
    // Test Case 4: Scan
    printf("Running Test Case 4: Scan (Prefix Sum)\n");
    start = clock();
    inclusive_scan(vec, A, N);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("  Time: %f seconds\n\n", cpu_time_used);
    
    // Test Case 5: Wave Propagation
    printf("Running Test Case 5: Wave Propagation\n");
    start = clock();
    // Use existing arrays as current, previous, and next states
    for (int t = 0; t < 5; t++) {
        wave_propagation(B, A, C, N, N, 0.1f, 1.0f, 1.0f);
        
        // Cycle the arrays for next iteration
        float *temp = A;
        A = B;
        B = C;
        C = temp;
    }
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("  Time: %f seconds\n\n", cpu_time_used);
    
    // Test Case 6: Heterogeneous Mix
    printf("Running Test Case 6: Heterogeneous Workload\n");
    start = clock();
    heterogeneous_workload(A, B, C, N, 0.5f);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("  Time: %f seconds\n\n", cpu_time_used);
    
    // Test Case 7: Matrix Transpose
    printf("Running Test Case 7: Matrix Transpose\n");
    start = clock();
    matrix_transpose(A, C, N, N);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("  Time: %f seconds\n\n", cpu_time_used);
    
    // Test Case 8: Histogram
    printf("Running Test Case 8: Histogram\n");
    start = clock();
    histogram(image, hist, N, N);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("  Time: %f seconds\n\n", cpu_time_used);
    
    // Clean up
    free(A);
    free(B);
    free(C);
    free(vec);
    free(image);
    free(hist);
    
    printf("All tests completed successfully!\n");
    
    return 0;
}
