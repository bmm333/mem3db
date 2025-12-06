#include "include/gpu_engine/gpu/vector_ops.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace gpu{
    //128bytes=32floats
    //using float4 so 32 floats/4 = 8 iterations
    static constexpr int FLOATS_PER_VECTOR=8;

    //constants from layout
    static constexpr uint32_t SLOT_FILLED=2u;

    //KERNEL DOT PRODUCT
    __global__ void kernel_dot_product(SoALayout table, 
                                   const float* __restrict__ query, 
                                   float* __restrict__ scores, 
                                   size_t n)
    {
        //1 grid stride loop
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        //2. Check if slot is occupied
        if (table.flags[idx] != 2) { // 2 = SLOT_FILLED
            scores[idx] = -1e30f; // Negative infinity for empty slots
            return;
        } 
        // 3. Pointers Setyo (Vectorized)
        // Cast the byte pointer to float4* to load 16 bytes at a time
        const float4* my_vec = reinterpret_cast<const float4*>(table.values + (idx * VALUE_SIZE));
        const float4* q_vec  = reinterpret_cast<const float4*>(query);

        // 4. Compute Dot Product
        float dot = 0.0f;
        
        // Unroll for speed. 
        #pragma unroll
        for (int i = 0; i < VECTORS_PER_ITEM; ++i) {
            float4 v = my_vec[i]; // Load 16 bytes from VRAM
            float4 q = q_vec[i];  // Load 16 bytes from Global (Cached in L1)
            
            dot += v.x * q.x;
            dot += v.y * q.y;
            dot += v.z * q.z;
            dot += v.w * q.w;
        }
        // 5. Store Score
        scores[idx] = dot;
    }

    
    / HOST WRAPPER
    void vector_search_dev(SoALayout table, 
                        const float* query_vector, 
                        float* scores_out, 
                        size_t count, 
                        cudaStream_t stream) 
    {
        if (!table.valid()) return;

        dim3 block(256);
        dim3 grid((count + block.x - 1) / block.x);

        // Launch the scan
        kernel_dot_product<<<grid, block, 0, stream>>>(table, query, scores_out, count);
        
        // Error check is good practice
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Vector Kernel Error: %s\n", cudaGetErrorString(err));
        }
    }
}