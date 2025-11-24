#include "include/gpu_engine/gpu/hash_kernel.h"
#include <cuda_runtime.h>

__device__ void sha256_block(const uint8_t* input,uint8_t* output)
{
    //todo: implement  block-level hashing
    //will go with xxHash64 or similar this needs to be studied first , so in order: test , then choose&&implement describing tradeoffs
}

__global__ void hash_kernel(HashJob* jobs,size_t count)
{
    size_t idx=blockIdx.x*blockDim+threadIdx.x;
    if(idx>=count) return;

    sha256_block(jobs[idx].input,jobs[idx].output);
}
void launch_hash_kernel(const HashJob* jobs,size_t count)
{
    dim3 block(128); 
    dim3 grid((count+block.x-1)/block.x);
    hash_kernel<<<grid,block>>>(jobs,count);
    cudaDeviceSynchronize();
}