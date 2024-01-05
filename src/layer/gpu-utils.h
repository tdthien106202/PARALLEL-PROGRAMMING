#ifndef SRC_LAYER_GPU_UTILS_H
#define SRC_LAYER_GPU_UTILS_H
#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(1);\
    }\
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

void printDeviceInfo();
#endif 
