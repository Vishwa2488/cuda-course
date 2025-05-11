#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_ERROR_CHECK(val) check(val, #val, __FILE__, __LINE__)

template <typename T>
void check(T err, char* func, char const * filename, int line)
{
    if (err != cudaSuccess)
    {
        
    }

    return;
}
