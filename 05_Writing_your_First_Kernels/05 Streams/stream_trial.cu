#include <cuda_runtime.h>
#include <stdio.h>

#include <iostream>

#define CUDA_ERROR_CHECK(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char * const filename, const int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "[ERROR] %d, %s %s - %s\n", line, filename, func, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


__global__
void kernel1(float * A, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        A[idx] *= 2;
    }
    return;
}

__global__
void kernel2(float * A, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        A[idx] += 1;
    }

    return;
}



void CUDART_CB myStreamCallback(cudaStream_t stream1,cudaError_t status, void *userData)
{
    printf("Stream completed successfully\n");
    return;
}

int main()
{
    float * d_data, * h_data;
    int N = 256;
    int size = N * sizeof(float);

    cudaStream_t stream1, stream2;
    cudaEvent_t event;

    CUDA_ERROR_CHECK(cudaStreamCreate(&stream1));
    CUDA_ERROR_CHECK(cudaStreamCreate(&stream2));
    CUDA_ERROR_CHECK(cudaEventCreate(&event));
    CUDA_ERROR_CHECK(cudaMalloc(&d_data, size));
    CUDA_ERROR_CHECK(cudaMallocHost(&h_data, size));
    for (int i=0;i<N;i++)
    {
        h_data[i] = static_cast <float> (i);
    }

    CUDA_ERROR_CHECK(cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream1));
    kernel1 <<<(N + 255)/256, 256, 0, stream1>>> (d_data, N);

    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream1)); // not reqd as the event recording for stream1 ensures this

    CUDA_ERROR_CHECK(cudaEventRecord(event, stream1)); // waits till all threads of stream1 completes

    CUDA_ERROR_CHECK(cudaStreamWaitEvent(stream2, event, 0)); // waits for the event

    kernel2 <<<(N+255)/256, 256, 0, stream2>>> (d_data, N);

    CUDA_ERROR_CHECK(cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream2));
    
    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream1));
    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream2));

    CUDA_ERROR_CHECK(cudaStreamAddCallback(stream2, myStreamCallback, NULL, 0));
    bool correct = true;

    for (int i=0;i<N;i++)
    {
        if (fabs(h_data[i] - (static_cast <float> (i) * 2 + 1)) > 1e-5)
        {
            correct = false;
            break;
        }
    }

    printf("Answer %s\n", (correct)? "correct": "incorrect");

    CUDA_ERROR_CHECK(cudaFree(d_data));
    CUDA_ERROR_CHECK(cudaFreeHost(h_data));
    CUDA_ERROR_CHECK(cudaStreamDestroy(stream1));
    CUDA_ERROR_CHECK(cudaStreamDestroy(stream2));
    CUDA_ERROR_CHECK(cudaEventDestroy(event));

    return 0;
}