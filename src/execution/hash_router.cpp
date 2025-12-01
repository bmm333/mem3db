#include "include/gpu_engine/execution/hash_router.h"
#include "include/gpu_engine/gpu/device_context.h"
#include "include/gpu_engine/gpu/insert_kernel.h"
#include "include/gpu_engine/gpu/lookup_kernel.h"
#include <cstring>
#include <stdexcept>
#include <iostream>

static void check(cudaError_t err,const char* msg)
{
    if(e!=cudaSuccess) throw std::runtime_error(std::string(msg)+": "+cudaGetErrorString(err));
}
HashRouter::HashRouter(SoALayout table,size_t max_batch_size)
:table_(table),max_batch_(max_batch_size)
{
    stream_=DeviceContext::instance().stream();
    size_t keys_sz=max_batch_*KEY_SIZE;
    size_t vals_sz=max_batch_*VALUES_SIZE;
    size_t status_sz=max_batch_*sizeof(uint8_t);

    check(cudaMallocHost(&h_stage_keys,keys_sz),"Host alloc keys");
    check(cudaMallocHost(&h_stage_values,vals_sz),"Host alloc values");
    check(cudaMallocHost(&h_stage_status,status_sz),"Host alloc status");

    check(cudaMalloc(&d_stage_keys,keys_sz),"Device alloc keys");
    check(cudaMalloc(&d_stage_vals_,vals_sz),"Device alloc values");
    check(cudaMalloc(&d_stage_status_,status_sz),"Device alloc status");
}

HashRouter::~HashRouter()
{
    cudaFreeHost(h_stage_keys_);
    cudaFreeHost(h_stage_vals_);
    cudaFreeHost(h_stage_status_);
    cudaFree(d_stage_keys_);
    cudaFree(d_stage_vals_);
    cudaFree(d_stage_status_);
}

size_t HashRouter::put_batch(const uint8_t* keys, const uint8_t* values, size_t count)
{
    //will continue asap
}