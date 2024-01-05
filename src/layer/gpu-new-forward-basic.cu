#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define cudaErrChk(stmt) \
  { cudaAssert((stmt), __FILE__, __LINE__); }

inline void cudaAssert(cudaError_t error,
                       const char* file,
                       int line,
                       bool abort = true) {
  if (error != cudaSuccess) {
    std::cerr << "CUDA error: "
              << cudaGetErrorString(error) << ' ' << file << ':' << line << std::endl;
    if (abort) {
      exit(error);
    }
  }
}

#define TILE_WIDTH 16

__global__ void conv_forward_kernel(float *y, 
                                    const float *x, 
                                    const float *k, 
                                    const int B, 
                                    const int M, 
                                    const int C, 
                                    const int H, 
                                    const int W, 
                                    const int K)
{
	// Calculate output dimensions
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]


	// Calculate grid dimensions for parallelization
    int H_grid = ceil(1.0*H_out / TILE_WIDTH);
    int W_grid = ceil(1.0*W_out / TILE_WIDTH); 
    
	// Extract thread indices
    int b = blockIdx.x;                 // batch number
    int m = blockIdx.y;                 // output feature map
    int h = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y; // row of image matrix
    int w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x; // col of image matrix
    
	// Param for convolution result
    float accum = 0.0f;
	// Perform convolution by summing over input features and applying the filter
    if ((h < (H_out)) && (w < (W_out))) 
    {
        float accum = 0.0f;
        for(int c=0; c<C; c++)             // sum over all input features
        {
            for(int p=0; p<K; p++)         // KxK filter 
                for(int q=0; q<K; q++)
                    accum += x4d(b, c, h+p, w+q) * k4d(m, c, p, q); // 4 dimensions macro resolve thread index
        }
		// Store the result in the output array
        y4d(b,m,h,w) = accum;
    }
	// Undefine macros to avoid potential conflicts
#undef y4d
#undef x4d
#undef k4d
}
	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_y, 
                                                    const float *host_x, 
                                                    const float *host_k, 
                                                    float **device_y_ptr, 
                                                    float **device_x_ptr, 
                                                    float **device_k_ptr, 
                                                    const int B, 
                                                    const int M, 
                                                    const int C, 
                                                    const int H, 
                                                    const int W, 
                                                    const int K)
{
    // Allocate memory and copy data to GPU(device)
    printf("(B=%d, M=%d, C=%d, H=%d, W=%d, K=%d)\n", B, M, C, H, W, K);

    // We pass double pointers for you to initialize the relevant device pointers,
    // which are passed to the other two functions.

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    const int inputSize  = B * C * H * W * sizeof(float);  // input features map C
    const int outputSize = B * M * H_out * W_out * sizeof(float); // output feature map M
    const int filterSize = M * C * K * K * sizeof(float); // C * M filter Maps of size K*K

    cudaMalloc((void **) device_x_ptr, inputSize);
    cudaMalloc((void **) device_y_ptr, outputSize);
    cudaMalloc((void **) device_k_ptr, filterSize);

    // Copy input data to device
    cudaMemcpy(*device_x_ptr, host_x, inputSize, cudaMemcpyHostToDevice);
    // Copy filter size to device
    cudaMemcpy(*device_k_ptr, host_k, filterSize, cudaMemcpyHostToDevice);

}


__host__ void GPUInterface::conv_forward_gpu(float *device_y, 
                                             const float *device_x, 
                                             const float *device_k, 
                                             const int B, 
                                             const int M, 
                                             const int C, 
                                             const int H, 
                                             const int W, 
                                             const int K)
{
    // Set the kernel dimensions and call the kernel

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    const int H_grid = ceil(1.0*H_out / TILE_WIDTH);
    int W_grid = ceil(1.0*W_out / TILE_WIDTH);
    int Z = H_grid * W_grid;

    // Block dimensions = #of threads in the block
    dim3 numThreadsPerBlock(TILE_WIDTH, TILE_WIDTH, 1);

    // Grid Dimension = #of Blocks: Batch Size(B) * Number of Output Features(M) * Calculated gridsize(Z)
    dim3 numBlocksInGrid(B, M, Z);


    // launch the kernel
    conv_forward_kernel<<<numBlocksInGrid, numThreadsPerBlock>>>(device_y, device_x, device_k, B, M, C, H, W, K);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, 
                                                    float *device_y, 
                                                    float *device_x, 
                                                    float *device_k, 
                                                    const int B, 
                                                    const int M, 
                                                    const int C, 
                                                    const int H, 
                                                    const int W, 
                                                    const int K)
{
    // Copy the output from device back to host
    
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int outputSize = B * M * H_out * W_out * sizeof(float);

    cudaMemcpy(host_y, device_y, outputSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_k);
}

