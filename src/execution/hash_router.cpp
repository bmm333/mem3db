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
    if(count>max_batch_) throw std::runtime_error("batch too large");
    if(count==0)return 0;

    size_t keys_bytes=count*KEY_SIZE;
    size_t vals_bytes=count*VALUES_SIZE;

    //1 copy user->Pinned
    std::memcpy(h_stage_keys_,keys,keys_bytes);
    std::memcpy(h_stage_vals_,values,vals_bytes);
    //2 DMA pinned -> device
    check(cudaMemcpyAsync(d_stage_keys_,h_stage_keys_,keys_bytes,cudaMemecpyHostToDevice,stream_),"DMA keys");
    check(cudaMemcpyAsync(d_stage_keys_,h_stage_keys_,keys_bytes,cudaMemcpyHostToDevice,stream_),"DMA vals");

    //3 Kernel
    gpu::batch_insert_dev(table_,d_stage_keys_,d_stage_vals_,nullptr,count,d_stage_status_,stream_);
    //4 status back
    check(cudaMemcpyAsync(h_stage_status_, d_stage_status_, count * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream_), "DMA status");
    //5 sync (will be blocking for now , safe for mvp just to get things working)
    check(cudaStreamSynchronize(stream_),"Sync put");
    //6 Count
    size_t success=0;
    for(size_t i=0;i<count;++i)
    {
        if(h_stage_status_[i]==gpu::INSERT_OK||h_stage_status_[i]==gpu::INSERT_OVERWRITE)
        {++success;}
    }return success;
}
void HashRouter::get_batch(const uint8_t* keys,size_t count,uint8_t* results,bool* found)
{
    if(count>max_batch_)throw std::runtime_error("Batch too large");
    if(count==0)return;
    size_t keys_bytes=count*KEY_SIZE;
    //1 copy keys->pinned
    std::memcpy(h_stage_keys_,keys,keys_bytes);
    //2 DMA pinned->device
    check(cudaMemcpyAsync(d_stage_keys_, h_stage_keys_, keys_bytes, cudaMemcpyHostToDevice, stream_), "DMA keys");
    //3 Kernel
    gpu::batch_lookup_dev(table_,d_stage_keys_,nullptr,d_stage_vals_,d_stage_status_,count,stream_);
    //4 DMA results -> pinned
    check(cudaMemcpyAsync(h_stage_vals_, d_stage_vals_, count * VALUE_SIZE, cudaMemcpyDeviceToHost, stream_), "DMA vals back");
    check(cudaMemcpyAsync(h_stage_status_, d_stage_status_, count * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream_), "DMA status back");
    check(cudaStreamSynchronize(stream_),"Sync get");
    //5 copy pinned->user ouput
    std::memcpy(results,h_stage_vals_,count*VALUES_SIZE);
    for(size_t i=0;i<count;++i)
    {
        found[i]=(h_stage_status_[i]==gpu::LOOKUP_FOUND);
    }
}
