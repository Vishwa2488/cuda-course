#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define M 3
#define K 4
#define N 2

#define CHECK_CUDA(call) {\
    cudaError_t err = call;\
    if (err != cudaSuccess)\
    {\
        fprintf(stderr, "[CUDA ERROR] in %s %d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err));\
        exit(EXIT_FAILURE);\
    }\
}

#define CHECK_CUBLAS(call) {\
    cublasStatus_t status = call;\
    if (status != CUBLAS_STATUS_SUCCESS)\
    {\
        fprintf(stderr, "[CUBLAS ERROR] in %s %d - %d\n", __FILE__, __LINE__, status);\
        exit(EXIT_FAILURE);\
    }\
}


#undef PRINT_MATRIX
#define PRINT_MATRIX(mat, rows, cols) \
    for (int i = 0; i < rows; i++) { \
        for (int j = 0; j < cols; j++) \
            printf("%8.3f ", mat[i * cols + j]); \
        printf("\n"); \
    } \
    printf("\n");

void cpu_matmul(float *A, float *B, float *C) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

int main() {
    float A[M * K] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    float B[K * N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    float C_cpu[M * N], C_cublas_s[M * N], C_cublas_h[M * N];

    cpu_matmul(A, B, C_cpu);

    cublasHandle_t handle;

    CHECK_CUBLAS(cublasCreate(&handle));

    float * d_A, * d_B, * d_C;

    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    float alpha = 1.0, beta = 0.0;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
  

    CHECK_CUDA(cudaMemcpy(C_cublas_s, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    half * d_A_h, * d_B_h, * d_C_h;

    CHECK_CUDA(cudaMalloc(&d_A_h, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B_h, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C_h, sizeof(half) * M * N));


    half A_h[M * K], B_h[K * N];

    for (int i = 0; i < M * K; i++) {
        A_h[i] = __float2half(A[i]);
    }
    for (int i = 0; i < K * N; i++) {
        B_h[i] = __float2half(B[i]);
    }

    // Copy half precision data to device
    CHECK_CUDA(cudaMemcpy(d_A_h, A_h, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_h, B_h, K * N * sizeof(half), cudaMemcpyHostToDevice));

    half alpha_h = __float2half(1.0), beta_h = __float2half(0.0);
    CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_h, d_B_h, N, d_A_h, K, &beta_h, d_C_h, N));

    half C_h[M * N];
    CHECK_CUDA(cudaMemcpy(C_h, d_C_h, M * N * sizeof(half), cudaMemcpyDeviceToHost));

    for (int i = 0; i < M * N; i++) {
        C_cublas_h[i] = __half2float(C_h[i]);
    }

    // Print results
    printf("Matrix A (%dx%d):\n", M, K);
    PRINT_MATRIX(A, M, K);
    printf("Matrix B (%dx%d):\n", K, N);
    PRINT_MATRIX(B, K, N);
    printf("CPU Result (%dx%d):\n", M, N);
    PRINT_MATRIX(C_cpu, M, N);
    printf("cuBLAS SGEMM Result (%dx%d):\n", M, N);
    PRINT_MATRIX(C_cublas_s, M, N);
    printf("cuBLAS HGEMM Result (%dx%d):\n", M, N);
    PRINT_MATRIX(C_cublas_h, M, N);

    // Clean up
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_A_h));
    CHECK_CUDA(cudaFree(d_B_h));
    CHECK_CUDA(cudaFree(d_C_h));
    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}


