#include <stdio.h>

// A simple matrix multiplication function that could benefit from GPU parallelization
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

// A function with reduction - potentially harder to parallelize fully
float vector_sum(float* vec, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += vec[i];
    }
    return sum;
}

// A function with loop-carried dependencies - harder to parallelize
void sequential_update(float* vec, int N) {
    for (int i = 1; i < N; i++) {
        vec[i] = vec[i] + vec[i-1];
    }
}

int main() {
    const int N = 1024;
    float A[N*N], B[N*N], C[N*N];
    float vec[N];
    
    // Initialize arrays (in real code, you would have actual data)
    for (int i = 0; i < N*N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }
    
    for (int i = 0; i < N; i++) {
        vec[i] = i;
    }
    
    // Run the functions
    matrix_multiply(A, B, C, N);
    float sum = vector_sum(vec, N);
    sequential_update(vec, N);
    
    printf("Sum: %f\n", sum);
    printf("Matrix multiplication result sample: %f\n", C[0]);
    printf("Sequential update result sample: %f\n", vec[N-1]);
    
    return 0;
}
