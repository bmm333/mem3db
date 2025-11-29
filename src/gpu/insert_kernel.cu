#include "gpu/insert_kernel.h"
#include "gpu/device_context.h"
#include "gpu/hash_kernel.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>

//Slot flag states *uint32_t
static constexpr uint32_t SLOT_EMPTY = 0u;
static constexpr uint32_t SLOT_WRITING = 1u;
static constexpr uint32_t SLOT_FILLED = 2u;

#if defined(__CUDA_ARCH__)
  #define HOST_DEVICE __host__ __device__
#else
    #define HOST_DEVICE
#endif

//fast 64-bit loader for aligned memory (assumes guaranteed alignment)
HOST_DEVICE static inline uint64_t load_u64_aligned(const uint8_t* p)
{
    return *reinterpret_cast<const uint64_t*>(p);
}
//fast 64bit store for aligned memory
HOST_DEVICE static inline void store_u64_aligned(uint8_t* p,uint64_t v)
{
    *reinterpret_cast<uint64_t*>(p)=v;
}
//copy key_size bytes from src to dst using 8-bytes stores 
HOST_DEVICE static inline void copy_key_aligned(uint8_t* dst,const uint8_t v)
{
    constexpr size_t W=8;
    constexpr size_t WORDS=KEY_SIZE/W;
    const uint64_t* s=reinterpret_cast<const uint64_t*>(src);
    uint64_t* d=reinterpret_cast<uint64_t*>

#pragma unroll
    for(size_t i=0;i<WORDS;++i)
    {
        d[i]=s[i];
    }
    //if key_sie is not multiple of 8 , handle tail (not excpeted when ks=128)
    size_t tail=KEY_SIZE%W;
    if(tail)
    {
        size_t base=WORDS*W;
        for(size_t i=0;i<tail;++i) dst[base+i]=src[base+i];
    }
}
//copy VALUE_SIZE bytes 
HOST_DEVICE static inline void copy_value_aligned(uint8_t* dst,const uint8_t* src)
{
    constexpr size_t W=8;
    constexpr size_t WORDS=VALUE_SIZE/W;
    const uint64_t* s=reinterpret_cast<const uint64_t*>(src);
    uint64_t* d=reinterpret_cast<uint64_t*>(dst);
#pragma unroll
    for(size_t i=0;i<WORDS;++i)
    {
        d[i]=s[i];
    }
    size_t tail=VALUE_SIZE%W;
    if(tail)
    {
        size_t base=WORDS*W;
        for(size_t i=0;i<tail;++i) dst[base+i]=src[base+i];
    }
}

HOST_DEVICE static inline bool dev_key_equals_aligned(const uint8_t* a,const uint8_t* b)
{
    constexpr size_t W=8;
    const uint64_t* aa=reinterpret_cast<const uint64_t*>(a);
    const uint64_t* bb=reinterpret_cast<const uint64_t*>(b);
    constexpr size_t WORDS=KEY_SIZE/W;
#pragma unroll
    for(size_t i=0;i<WORDS;++i)
    {
        if(aa[i]!=bb[i]) return false;
    }
    //tail if any
    size_t tail=KEY_SIZE%W;
    if(tail)
    {
        size_t base=WORDS*W;
        for(size_t i=0;i<tail;++i)
        {
            if(a[base+i]!=b[base+i]) return false;
        }
    }
    return true;
}

__global__ static void kernel__insert__batch(SoALayout table,
                                             const uint8_t* keys_dev,
                                             const uint8_t* vals_dev,
                                             const uint64_t* hashes_dev,
                                             uint8_t* status_dev,
                                             size_t n)
{
    size_t tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=n)return;
    const uint8_t* key_ptr=keys_dev+tid*KEY_SIZE;
    const uint8_t* val_ptr=vals_dev+tid*VALUE_SIZE;

    uint64_t h;
    if(hashes_dev)
    {
        h=hashes_dev[tid];
    }
    else{
        //compute hash on device if host didn't provide it
        h=xxhash64_device(key_ptr,KEY_SIZE,0);
    }
    const size_t mask=table.mask;
    size_t idx=(size_t)(h&mask);
    uint8_t local_status=INSERT_FAIL;
    for(uint32_t probe=0;probe<table.max_probes;++probe)
    {
        //try to acquire lock
        uint32_t prev = atomicCAS(&table.flags[idx], SLOT_EMPTY, SLOT_WRITING);
        if(prev==SLOT_EMPTY)
        {
            //we got the slot
            uint8_t* dst_key=table.keys+idx*KEY_SIZE;
            uint8_t* dst_val=table.vals+idx*VALUE_SIZE;

            copy_key_aligned(dst_key,key_ptr);
            copy_value_aligned(dst_val,val_ptr);
            table.hashes[idx]=h;
            //fence before unlocking
            __threadfence();
            table.flags[idx]=SLOT_FILLED;
            local_status=INSERT_OK;
            break;
        }
        //2nd Case sloty busy - check duplicate/update
        //we check if matches our hash. even if writing we can speculatively check hash
        //but strictly we should only check key if filled
        if(prve==SLOT_FILLED)
        {
            uint64_t stored_hash=table.hashes[idx];
            if(stored_hash==h)
            {
                const uint8_t* stored_key=table.keys+idx*KEY_SIZE;
                if(dev_key_equals_aligned(stored_key,key_ptr))
                {
                    //update(last-writer-wins, potential tearing on concurrent update to same key)
                    uint8_t* dst_val=table.values+idx*VALUE_SIZE;
                    copy_value_aligned(dst_val,val_ptr);
                    //Ensure write visible
                    __threadfence();
                    local_status=INSERT_UPDATED;
                    break;
                }
            }
        }
        //3rd case collision - linear probe
        idx=(idx+1)&mask;
    }
    if(status_dev) status_dev[tid]=local_status;
}
//host side helpers
static inline void cuda_check(cudaError_t e,const char* msg)
{
    if(e!=cudaSuccess)
    {
        throw std::runtime_error(msg);
    }
}
//host wrapper: excepts device buffers already allocated nd populated 
//this avoids malloc/free per cal and is recc hot path
namespace gpu{
    void batch_insert_dev(SoALayout table,
                      const uint8_t* keys_dev,
                      const uint8_t* vals_dev,
                      const uint64_t* hashes_dev,
                      size_t n,
                      uint8_t* status_dev,
                      cudaStream_t stream) 
{
    if(!table.valid())throw std::runtime_error("SoALayout invalid in batch_insert_dev");
    if(n==0)return;
    const int block=256;
    int grid=(int)((n+block-1)/block);
    kernel__insert__batch<<<grid,block,0,stream>>>(table,keys_dev,vals_dev,hashes_dev,status_dev,n);
    cudaError_t err=cudaGetLastError();
    if(err!=cudaSuccess)
    {
        throw std::runtime_error(std::string("kernel__insert__batch launch failed: ")+cudaGetErrorString(err));
    }
    //opt sync here or leave to caller
    cuda_check(cudaStreamSynchronize(stream),"batch_insert_dev stream sync failed");
}
//convenience host wrapper taht copies host buffers to device with staging buffers,
//not ideal for hot path; prefer using preallocated device staging and batch_insert_dev

void batch_insert_host(SoALayout table,
                       const uint8_t* host_keys_packed,
                       const uint8_t* host_vals_packed,
                       const uint64_t* host_hashes,
                       size_t n,
                       uint8_t* host_status,
                       cudaStream_t stream)
{
    if(!table.valid())throw std::runtime_error("SoALayout invalid");
    if(n==0)return;

    size_t keys_b = n*KEY_SIZE;
    size_t vals_b = n*VALUE_SIZE;
    size_t hashes_b = n*sizeof(uint64_t);
    size_t status_b=n*sizeof(uint8_t);
    uint8_t* keys_dev=nullptr;
    uint8_t* vals_dev=nullptr;
    uint64_t* hashes_dev=nullptr;
    uint8_t* status_dev=nullptr;
    //llocate temp device buffers (caller should avoid this in hot path really)
    cuda_check(cudaMalloc(&keys_dev,keys_b),"cudaMalloc keys_dev failed");
    cuda_check(cudaMalloc(&vals_dev,vals_b),"cudaMalloc vals_dev failed");
    if(host_hashes) cuda_check(cudaMalloc(&hashes_dev,hashes_b),"cudaMalloc hashes_dev failed");
    if(host_status) cuda_check(cudaMalloc(&status_dev,status_b),"cudaMalloc status_dev failed");
    //copy host -> device
    cuda_check(cudaMemcpyAsync(keys_dev, host_keys_packed, keys_b, cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync keys -> dev failed");
    cuda_check(cudaMemcpyAsync(vals_dev, host_vals_packed, vals_b, cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync vals -> dev failed");
    if (host_hashes) cuda_check(cudaMemcpyAsync(hashes_dev, host_hashes, hashes_b, cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync hashes -> dev failed");
    //launch kernel
    batch_insert_dev(table,keys_dev,vals_dev,hashes_dev,n,status_dev,stream);
    //copy staus back if requested
    if(host_status&&status_dev)
    {
        cuda_check(cudaMemcpyAsync(host_status, status_dev, status_b, cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync status -> host failed");
    }
    //synchronize (hhost wrapper simplifies semantics)
    cuda_check(cudaStreamSynchronize(stream),"batch_insert_host stream sync failed");
    //free temps
    cudaFree(keys_dev);
    cudaFree(vals_dev);
    if(hashes_dev) cudaFree(hashes_dev);
    if(status_dev) cudaFree(status_dev);
}
}//namespace gpu
                                             