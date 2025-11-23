#include "include/storage/layout.h"
#include <cuda_runtime.h>
#include <stdexcept>

static inline void check(cudaError_t err, const char* msg)
{
    if(err!=cudaSuccess)
        throw std::runtime_error(msg);
}
SoALayout allocate_soa(size_t capacity,uint32_t max_probes,uint32_t batch_size)
{
    if((capacity&(capacity-1))!=0)
    {
        throw std::runtime_error("Capacity must be a power of 2");
    }
    SoALayout t;
    t.capacity=capacity;
    t.mask=capacity-1;
    t.max_probes=max_probes;
    t.batch_size=batch_size;

    check(cudaMalloc(&t.keys,keys_bytes(capacity)),"cudaMalloc keys failed");
    check(cudaMalloc(&t.values,values_bytes(capacity)),"cudaMalloc values failed");
    check(cudaMalloc(&t.hashes,hashes_bytes(capacity)),"cudaMalloc hashes failed");
    check(cudaMalloc(&t.flags,flags_bytes(capacity)),"cudaMalloc flags failed");
    check(cudaMemset(t.flags,EntryFlag::EMPTY,flags_bytes(capacity)),"cudaMemset flags failed");

    return t;
}

void free_soa(SoALayout& t)
{
    cudaFree(t.keys);
    cudaFree(t.values);
    cudaFree(t.hashes);
    cudaFree(t.flags);

    t={};
}