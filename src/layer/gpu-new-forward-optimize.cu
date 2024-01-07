#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

# define TILE_WIDTH 8
# define new_TILE_WIDTH 16

__constant__ float MASK[7*7*16*4];

/*
Modify this function to implement the forward pass described in Chapter 16.
We have added an additional dimension to the tensors to support an entire mini-batch
The goal here is to be correct AND fast.

Function paramter definitions:
y - output
x - input
k - kernel
B - batch_size (number of images in x)      100/1000/10000
M - number of output feature maps           4   16
C - number of input feature maps            1   4
H - input height dimension                  86  40
W - input width dimension                   86  40
K - kernel height and width (K x K)         7   7
*/

// A new version which uses kernel in constant memory
__global__ void conv_forward_kernel_constant(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    //(void)H_out; // silence declared but never referenced warning. remove this line when you start working
    //(void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) MASK[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int w_grid = ceil(1.*W_out/TILE_WIDTH);
    int h_grid = ceil(1.*H_out/TILE_WIDTH);

    int n = blockIdx.x;
    int m = blockIdx.y;
    int h = blockIdx.z/w_grid*TILE_WIDTH + threadIdx.y;
    int w = blockIdx.z%w_grid*TILE_WIDTH + threadIdx.x;


    if ((w < (W_out)) && (h < (H_out))) {
        float acc = 0.0f;
        for (int c = 0; c<C; c++){          // loop all input channels
            for (int p = 0; p<K; p++){      // loop over k*k filter
                for (int q = 0; q<K; q++){
                    acc += x4d(n,c,h+p,w+q)*k4d(m,c,p,q);
                }
            }
        }
        y4d(n,m,h,w) = acc;
    }
    #undef y4d
    #undef x4d
    #undef k4d
}

// A new version which uses shared memory tiling 
// and kernel in shared memory
__global__ void conv_forward_kernel_shared(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int w_grid = ceil(1.*W_out/TILE_WIDTH);

    int n = blockIdx.x;
    int m = blockIdx.y;

    int X_tile_width = TILE_WIDTH + K - 1; 
    extern __shared__ float shmem[];
    float* X_shared = &shmem[0];
    float* K_shared = &shmem[X_tile_width * X_tile_width];

    int w0 = threadIdx.x;
    int h0 = threadIdx.y; // h0 and w0 used as shorthand for threadIdx.x and threadIdx.y 

    int h_base = (blockIdx.z / w_grid) * TILE_WIDTH; // vertical base out data index for the block 
    int w_base = (blockIdx.z % w_grid) * TILE_WIDTH; // horizontal base out data index for the block 

    int h = h_base + h0;
    int w = w_base + w0;

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    float acc = 0.;
    for (int c = 0; c < C; c++) {
        // sum over all input channels
        if (( h0 < K) && ( w0 < K)){
            K_shared[h0*K + w0] = k4d(m,c,h0,w0);
        }
        __syncthreads();

        for (int i = h; i < h_base + X_tile_width; i += TILE_WIDTH) { 
            for (int j = w; j < w_base + X_tile_width; j += TILE_WIDTH) {
                X_shared[(i-h_base)*(X_tile_width)+(j-w_base)]=x4d(n,c,i,j);
            }
        }
        __syncthreads();

        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++)
                acc += X_shared[(h0 + p) * X_tile_width + w0 + q] * K_shared[p*K+q];
        }
        __syncthreads(); 
    }
    if (n<B && m<M && h<H_out && w<W_out){
        y4d(n,m,h,w)=acc;
    }

    #undef y4d
    #undef x4d
    #undef k4d
}

// A new version which uses shared memory tiling 
// and kernel in constant memory
__global__ void conv_forward_kernel_shared_constant(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int w_grid = ceil(1.*W_out/TILE_WIDTH);

    int n = blockIdx.x;
    int m = blockIdx.y;

    int X_tile_width = TILE_WIDTH + K - 1; 
    extern __shared__ float shmem[];
    float* X_shared = &shmem[0];
    //float* K_shared = &shmem[X_tile_width * X_tile_width];

    int w0 = threadIdx.x;
    int h0 = threadIdx.y; // h0 and w0 used as shorthand for threadIdx.x and threadIdx.y 

    int h_base = (blockIdx.z / w_grid) * TILE_WIDTH; // vertical base out data index for the block 
    int w_base = (blockIdx.z % w_grid) * TILE_WIDTH; // horizontal base out data index for the block 

    int h = h_base + h0;
    int w = w_base + w0;

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) MASK[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    float acc = 0.;
    for (int c = 0; c < C; c++) {
        // sum over all input channels
        for (int i = h; i < h_base + X_tile_width; i += TILE_WIDTH) { 
            for (int j = w; j < w_base + X_tile_width; j += TILE_WIDTH) {
                X_shared[(i-h_base)*(X_tile_width)+(j-w_base)]=x4d(n,c,i,j);
            }
        }
        __syncthreads();

        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++)
                acc += X_shared[(h0 + p) * X_tile_width + w0 + q] * k4d(m,c,p,q);
        }
        __syncthreads(); 
    }
    if (n<B && m<M && h<H_out && w<W_out){
        y4d(n,m,h,w)=acc;
    }

    #undef y4d
    #undef x4d
    #undef k4d
}

// A new version which uses shared memory tiling 
// and kernel in constant memory
// and use different tile size for different input image size (Add another input, also modify the host code)
__global__ void conv_forward_kernel_shared_constant_multikernel(int tile_size, float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int w_grid = ceil(1.*W_out/tile_size);

    int n = blockIdx.x;
    int m = blockIdx.y;

    int X_tile_width = tile_size + K - 1; 
    extern __shared__ float shmem[];
    float* X_shared = &shmem[0];
    //float* K_shared = &shmem[X_tile_width * X_tile_width];

    int w0 = threadIdx.x;
    int h0 = threadIdx.y; // h0 and w0 used as shorthand for threadIdx.x and threadIdx.y 

    int h_base = (blockIdx.z / w_grid) * tile_size; // vertical base out data index for the block 
    int w_base = (blockIdx.z % w_grid) * tile_size; // horizontal base out data index for the block 

    int h = h_base + h0;
    int w = w_base + w0;

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) MASK[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    float acc = 0.;
    for (int c = 0; c < C; c++) {
        // sum over all input channels
        for (int i = h; i < h_base + X_tile_width; i += tile_size) { 
            for (int j = w; j < w_base + X_tile_width; j += tile_size) {
                X_shared[(i-h_base)*(X_tile_width)+(j-w_base)]=x4d(n,c,i,j);
            }
        }
        __syncthreads();

        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++)
                acc += X_shared[(h0 + p) * X_tile_width + w0 + q] * k4d(m,c,p,q);
        }
        __syncthreads(); 
    }
    if (n<B && m<M && h<H_out && w<W_out){
        y4d(n,m,h,w)=acc;
    }

    #undef y4d
    #undef x4d
    #undef k4d
}


// A new version which uses shared memory tiling 
// and kernel in constant memory
// and use different tile size for different input image size (Add another input, also modify the host code)
// and use Input channel reduction: atomics
__global__ void conv_forward_kernel_shared_constant_multikernel_atomic(int tile_size, float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    //__shared__ float acc[4];

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int w_grid = ceil(1.*W_out/tile_size);

    int n = blockIdx.x;
    int m = blockIdx.y;

    int X_tile_width = tile_size + K - 1; 
    extern __shared__ float shmem[];
    float* X_shared = &shmem[0];
    float* Y_shared = &shmem[X_tile_width * X_tile_width * C];

    int w0 = threadIdx.x;
    int h0 = threadIdx.y; // h0 and w0 used as shorthand for threadIdx.x and threadIdx.y 
    int c0 = threadIdx.z;

    int h_base = (blockIdx.z / w_grid) * tile_size; // vertical base out data index for the block 
    int w_base = (blockIdx.z % w_grid) * tile_size; // horizontal base out data index for the block 

    int h = h_base + h0;
    int w = w_base + w0;

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) MASK[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define xs3d(i2, i1, i0) X_shared[(i2) * (X_tile_width * X_tile_width) + (i1) * (X_tile_width) + i0]
    #define ys3d(i2, i1, i0) Y_shared[(i2) * (tile_size * tile_size) + (i1) * (tile_size) + i0] 

    if(c0<C){
        // sum over all input channels
        for (int i = h; i < h_base + X_tile_width; i += tile_size) { 
            for (int j = w; j < w_base + X_tile_width; j += tile_size) {
                xs3d(c0, i-h_base, j-w_base) = x4d(n,c0,i,j);
            }
        }
        __syncthreads();

        if (h0<tile_size && w0<tile_size){
            ys3d(m, h0, w0) = 0.;
        }
        __syncthreads();
        float temp = 0.;
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++)
                temp += xs3d(c0, h0+p, w0+q) * k4d(m,c0,p,q);
        }
        atomicAdd( &ys3d(m,h0,w0), temp);
        __syncthreads(); 
    }

    if (n<B && m<M && h<H_out && w<W_out){
        y4d(n,m,h,w) = ys3d(m,h0,w0);
    }

    #undef y4d
    #undef x4d
    #undef k4d
    #undef xs3d
    #undef ys3d
}


// Shared memory matrix multiplication and input matrix unrolling


__host__ void GPUInterface2::conv_forward_gpu_prolog(float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    cudaMalloc((void**) device_x_ptr, sizeof(float)*B*C*H*W);
    cudaMalloc((void**) device_y_ptr, sizeof(float)*B*M*H_out*W_out);
    cudaMalloc((void**) device_k_ptr, sizeof(float)*K*K*C*M);

    cudaMemcpy(*device_x_ptr,host_x,sizeof(float)*B*C*H*W,cudaMemcpyHostToDevice);
    cudaMemcpy(*device_k_ptr,host_k,sizeof(float)*K*K*C*M,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(MASK,host_k,K*K*C*M*sizeof(float));
    
    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    //Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

}


__host__ void GPUInterface2::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int w_grid = ceil(1.*W_out/TILE_WIDTH);
    int h_grid = ceil(1.*H_out/TILE_WIDTH);
    int Z = w_grid*h_grid;

    dim3 blockDim(TILE_WIDTH,TILE_WIDTH,1);
    dim3 gridDim(B,M,Z);

    // used for shared memory
    size_t shemem_size = sizeof(float) * ((TILE_WIDTH+K-1) * (TILE_WIDTH+K-1) + K*K);
    //conv_forward_kernel<<<gridDim,blockDim,shemem_size>>>(device_y,device_x,device_k,B,M,C,H,W,K);


    // for the kernel: conv_forward_kernel_shared_constant_multikernel
    int tile_size;
    if (H>64){
        tile_size = new_TILE_WIDTH;    // use big tile if the niput image is large
    } else{
        tile_size = TILE_WIDTH;
    }
    int Z_choice = ceil(1.*W_out/tile_size) * ceil(1.*H_out/tile_size);
    size_t shemem_size_choice = sizeof(float) * ((tile_size+K-1) * (tile_size+K-1) + K*K);
    dim3 blockDim_choice(tile_size,tile_size,1);
    dim3 gridDim_choice(B,M,Z_choice);
    //conv_forward_kernel_shared_constant_multikernel<<<gridDim_choice,blockDim_choice,shemem_size_choice>>>(tile_size,device_y,device_x,device_k,B,M,C,H,W,K);

    //for the kernel: conv_forward_kernel_shared_constant_multikernel_atomic
    int atomic_tile_size = tile_size;
    int Z_atomic = ceil(1.*W_out/atomic_tile_size) * ceil(1.*H_out/atomic_tile_size);
    size_t shemem_size_atomic = sizeof(float) * (((atomic_tile_size+K-1) * (atomic_tile_size+K-1)*C) + (atomic_tile_size) * (atomic_tile_size)*M);
    dim3 blockDim_atomic(atomic_tile_size,atomic_tile_size,C);
    dim3 gridDim_atomic(B,M,Z_atomic);
    conv_forward_kernel_shared_constant_multikernel_atomic<<<gridDim_atomic,blockDim_atomic,shemem_size_atomic>>>(atomic_tile_size,device_y,device_x,device_k,B,M,C,H,W,K);



}


__host__ void GPUInterface2::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Copy the output back to host
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    cudaMemcpy(host_y, device_y, sizeof(float)*B*M*H_out*W_out,cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_y);
    cudaFree(device_x);
    cudaFree(device_k);
}

