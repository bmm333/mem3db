#include "include/gpu_engine/gpu/vector_ops.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace gpu{
    //128bytes=32floats
    static constexpr int FLOATS_PER_VECTOR=32;
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
        // We only compute scores for valid data
        // Note: table.flags is uint32_t now based on previous steps
        if (table.flags[idx] != 2) { // 2 = SLOT_FILLED
            scores[idx] = -1000000.0f; // Negative infinity for empty slots
            return;
        } 
        // 3. Pointers to data
        // table.values is uint8_t*, cast to float*
        const float* my_vector = reinterpret_cast<const float*>(table.values + (idx * VALUE_SIZE));

        // 4. Compute Dot Product
        float dot = 0.0f;
        
        // Unroll for speed. 
        // Optimization: In production, we use float4 vector types for 128-bit loads.
        #pragma unroll
        for (int i = 0; i < FLOATS_PER_VECTOR; ++i) {
            dot += my_vector[i] * query[i];
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