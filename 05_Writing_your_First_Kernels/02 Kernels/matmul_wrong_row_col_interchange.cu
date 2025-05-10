#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

#define M 256  // Number of rows in A and C
#define K 512   // Number of columns in A and rows in B
#define N 256  // Number of columns in B and C
#define BLOCK_SIZE 32

// Example 3x2 @ 2x4 = 3x4 -> (M x K) @ (K x N) = (M x N)
// A = [[1, 2], 
//      [3, 4], 
//      [5, 6]]

// B = [[7, 8, 9, 10],
//      [11, 12, 13, 14]]

// C = A * B = [[1*7 + 2*11, 1*8 + 2*12, 1*9 + 2*13, 1*10 + 2*14],
//              [3*7 + 4*11, 3*8 + 4*12, 3*9 + 4*13, 3*10 + 4*14],
//              [5*7 + 6*11, 5*8 + 6*12, 5*9 + 6*13, 5*10 + 6*14]]

// C = [[29, 32, 35, 38],
//      [65, 72, 79, 86],
//      [101, 112, 123, 134]]

void initarray(float * A, int size)
{
    for (int i=0;i<size;i++)
    {
        A[i] = (float)rand()/RAND_MAX;
    }
    return;
}

void cpu_matmul(float * A, float * B, float * C, int m, int n, int k)
{
    for (int i=0;i<m;i++)
    {
        for (int j=0;j<n;j++)
        {
            double sum = 0.0;
            for (int l=0;l<k;l++)
            {
                sum+=A[i * k + l] * B[l * n + j];
            }

            C[i * n + j] = sum;
        }
    }
}

__global__
void matmul(float * A, float * B, float * C, int m, int n, int k)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n)
    {
        float sum = 0.0;

        for (int i=0;i<k;i++)
        {
            sum += A[row * k + i] * B[i * n + col];
        }

        C[row * n + col] = sum;
    }
}

// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}


int main()
{
    float * h_A, * h_B, * h_C, * h_GPU_C;
    float * d_A, * d_B, * d_C;

    h_A = (float *)malloc(sizeof(float) * M * K);
    h_B = (float *)malloc(sizeof(float) * K * N);
    h_C = (float *)malloc(sizeof(float) * M * N);
    h_GPU_C = (float *)malloc(sizeof(float) * M * N);


    srand(time(NULL));
    initarray(h_A, M * K);
    initarray(h_B, K * N);

    cudaMalloc(&d_A, sizeof(float) * M * K);
    cudaMalloc(&d_B, sizeof(float) * K * N);
    cudaMalloc(&d_C, sizeof(float) * M * N);

    cudaMemcpy(d_A, h_A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    dim3 gridDimensions((M + BLOCK_SIZE -1)/ BLOCK_SIZE, (N + BLOCK_SIZE-1)/BLOCK_SIZE);
    dim3 threadDimensions(BLOCK_SIZE,BLOCK_SIZE);

    // warmup
    for (int i=0;i<3;i++)
    {
        matmul <<<gridDimensions, threadDimensions>>> (d_A, d_B, d_C, M, N, K);  
        cudaDeviceSynchronize();
    }

    double cpu_time = 0.0, gpu_time = 0.0;

    // benchmarking
    for (int i=0;i<20;i++)
    {
        double start = get_time();
        cpu_matmul(h_A, h_B, h_C, M, N, K);
        double end = get_time();

        cpu_time += end - start;

        start = get_time();
        matmul <<<gridDimensions, threadDimensions>>> (d_A, d_B, d_C, M, N, K);  
        cudaDeviceSynchronize();
        end = get_time();
        gpu_time+=end - start;
    }

    cudaMemcpy(d_B, h_B, sizeof(float) * K * N, cudaMemcpyDeviceToHost);


    bool correct = true;
    for (int i=0;i<M;i++)
    {
        for (int j=0;j<N;j++)
        {
            if (fabs(h_C[i * N + j] - h_GPU_C[i*N + j])>1e-5)
            {
                correct = false;
                break;
            } 
        }
    }

    printf("Output is %s\n", (correct)? "correct" : "in correct");

    printf("CPU average time: %f microseconds\n", (cpu_time/20 * 1e6f));
    printf("GPU average time: %f microseconds\n", (gpu_time/20 * 1e6f));
    printf("Speedup: %fx\n", cpu_time / gpu_time);
    return 0;
}