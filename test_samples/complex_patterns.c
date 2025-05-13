#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Matrix transpose - good for GPU (memory access patterns can be optimized)
void matrix_transpose(float* in, float* out, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            out[j * N + i] = in[i * N + j];
        }
    }
}

// Matrix multiplication - perfect for GPU (high compute intensity)
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

// Vector reduction - requires special handling on GPU
float vector_sum(float* vec, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += vec[i];
    }
    return sum;
}

// Stencil computation - good for GPU with shared memory
void stencil_1d(float* in, float* out, int N) {
    for (int i = 1; i < N-1; i++) {
        out[i] = 0.3333f * (in[i-1] + in[i] + in[i+1]);
    }
}

// Histogram computation - requires atomic operations on GPU
void histogram(int* data, int* hist, int N, int bins) {
    for (int i = 0; i < bins; i++) {
        hist[i] = 0;
    }
    
    for (int i = 0; i < N; i++) {
        int bin = data[i] % bins;
        hist[bin]++;
    }
}

// Nested parallelism - can use 2D thread blocks on GPU
void nested_parallel(float* A, float* B, int N, int M) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            B[i * M + j] = A[i * M + j] * 2.0f;
        }
    }
}

// Wavefront pattern - diagonal traversal beneficial for GPU
void wavefront(float* A, int N) {
    for (int i = 1; i < N; i++) {
        for (int j = 1; j < N; j++) {
            A[i * N + j] = A[(i-1) * N + j] + A[i * N + (j-1)];
        }
    }
}

// Recursive pattern - might use dynamic parallelism on GPU
int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n-1) + fibonacci(n-2);
}

int main() {
    const int N = 1024;
    const int BINS = 256;
    
    // Allocate and initialize data
    float *A = (float*)malloc(N * N * sizeof(float));
    float *B = (float*)malloc(N * N * sizeof(float));
    float *C = (float*)malloc(N * N * sizeof(float));
    float *vec = (float*)malloc(N * sizeof(float));
    int *data = (int*)malloc(N * sizeof(int));
    int *hist = (int*)malloc(BINS * sizeof(int));
    
    // Initialize data
    srand(time(NULL));
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }
    
    for (int i = 0; i < N; i++) {
        vec[i] = (float)rand() / RAND_MAX;
        data[i] = rand() % BINS;
    }
    
    // Run the functions
    clock_t start, end;
    double cpu_time_used;
    
    // Matrix transpose
    start = clock();
    matrix_transpose(A, B, N);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Matrix transpose time: %f seconds\n", cpu_time_used);
    
    // Matrix multiplication
    start = clock();
    matrix_multiply(A, B, C, N);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Matrix multiplication time: %f seconds\n", cpu_time_used);
    
    // Vector reduction
    start = clock();
    float sum = vector_sum(vec, N);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Vector sum: %f, time: %f seconds\n", sum, cpu_time_used);
    
    // Stencil computation
    start = clock();
    stencil_1d(vec, vec, N);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Stencil computation time: %f seconds\n", cpu_time_used);
    
    // Histogram
    start = clock();
    histogram(data, hist, N, BINS);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Histogram computation time: %f seconds\n", cpu_time_used);
    
    // Nested parallelism
    start = clock();
    nested_parallel(A, B, N, N);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Nested parallel time: %f seconds\n", cpu_time_used);
    
    // Wavefront
    start = clock();
    wavefront(A, N);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Wavefront time: %f seconds\n", cpu_time_used);
    
    // Recursive pattern
    start = clock();
    int fib = fibonacci(20);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Fibonacci(20) = %d, time: %f seconds\n", fib, cpu_time_used);
    
    // Free memory
    free(A);
    free(B);
    free(C);
    free(vec);
    free(data);
    free(hist);
    
    return 0;
}
