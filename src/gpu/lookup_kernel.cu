#include "gpu/lookup_kernel.h"
#include "gpu/hash_kernel.h"
#include "gpu/device_context.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace gpu{
    //constants matching insert_kernel
    static constexpr uint32_t SLOT_EMPTY=0u;
    static constexpr uint32_t SLOT_WRITING=1u;
    static constexpr uint32_t SLOT_FILLED=2u;

    //reuse helpers (should move to a common utils.cuh later)
    __device__ __forceinline__ bool equals_128b_read(const uint8_t* a,const uint8_t* b)
    {
        const uint64_t* a64=reinterpret_cast<const uint64_t*>(a);
        const uint64_t* b64=reinterpret_cast<const uint64_t*>(b);
        #pragma unroll
        for(int i=0;i<16;i++)
        {
            if(a64[i]!=b64[i]) return false;
        }
        return true;
    }
    __device__ __forceinline__ void copy_128b_read(uint8_t* dst,const uint8_t* src)
    {
        uint64_t* d64=reinterpret_cast<uint64_t*>(dst);
        const uint64_t* s64=reinterpret_cast<const uint64_t*>(src);
        #pragma unroll
        for(int i=0;i<16;i++) d64[i]=s64[i];
    }
    //Kernel: Batch lookup
    __global__ void lookup_kernel__soa(SoALayout table,
    const uint8_t* keys_in,
    const uint64_t* hashes_in,
    uint8_t* vals_out,
    uint8_t* found_out,
    size_t n)
    {
        size_t tid=blockIdx.x*blockDim.x+threadIdx.x;
        if(tid>=n) return;
        const uint8_t* my_key=keys_in+(tid*KEY_SIZE);
        //hash
        uint64_t hash;
        if(hashes_in) hash=hashes_int[tid];
        else hash=xxhash64_device(my_key,KEY_SIZE,0);
        size_t idx=hash&table.mask;
        //linear probing
        for(uint32_t i=0;i<table.max_probes;++i)
        {
            uint32_t flag=table.flags[idx];
            //1 emtpy ? key def dose not exist
            if(flag==SLOT_EMPTY)
            {
                found_out[tid]=LOOKUP_NOT_FOUND;
                return;
            }
            //2 filled ? check key
            if(flag==SLOT_FILLED)
            {
                if(table.hashes[idx]==hash)
                {
                    const uint8_t* stored_key=table.keys+(idx*KEY_SIZE);
                    if(equals_128b_read(stored_key,my_key))
                    {
                        //found copy value to output buffer
                        uint8_t* my_out_val=vals_out+(tid*VALUE_SIZE);
                        const uint8_t* stored_val=table.values+(idx*VALUE_SIZE);

                        copy_128b_read(my_out_val,stored_val);
                        found_out[tid]=LOOKUP_FOUND;
                        return;
                    }
                }
            }
            //3. Writing (SLOT_WRITING) or Collision?
            //if its writing the data isn't ready. we assume our key might be further down the chain
            //or this slot is beign taken by someone else
            //We just continue probing.
            idx=(idx+1)&table.mask;
        }
        found_out[tid]=LOOKUP_NOT_FOUND;
    }
    //host wrappers
    static inline void check(cudaError_t e,const char* msg)
    {
        if(e!=cudaSuccess) throw std::runtime_error(msg);
    }
    void batch_lookup_dev(SoALayout table,
    const uint8_t* keys_dev,
    const uint64_t* hashes_dev,
    uint8_t* vals_out,
    uint8_t* found_out,
    size_t n,
    cudaStream_t stream)
    {
        if(!table.valid())throw std::runtime_error("batch_lookup_dev: invalid table layout");
        if(n==0)return;

        dim3 block(256);
        dim3 grid((n+block.x-1)/block.x);

        lookup_kernel_soa<<<grid,block,0,stream>>>(table,keys_dev,hashes_dev,vals_out,found_out,n);
        check(cudaGetLastError(),"Lookup kernel launch failed");
        //caller handles sync
    }
    void batch_lookup_host(SoALayout table,
                       const uint8_t* h_keys,
                       const uint64_t* h_hashes,
                       uint8_t* h_vals_out,
                       uint8_t* h_found_out,
                       size_t n,
                       cudaStream_t stream)
    {
        //alloc temps
        uint8_t* d_keys=nullptr;
        uint64_t* d_hashes=nullptr;
        uint8_t* d_vals=nullptr;
        uint8_t* d_found=nullptr;
        
        check(cudaMallocAsync(&d_keys,n*KEY_SIZE,stream),"malloc keys");
        check(cudaMallocAsync(&d_vals,n*VALUE_SIZE,stream),"malloc vals");
        check(cudaMallocAsync(&d_found,n*sizeof(uint8_t),stream),"malloc found");
        if (h_hashes) check(cudaMallocAsync(&d_hashes, n * sizeof(uint64_t), stream), "malloc hashes");

        //copy in
        check(cudaMemcpyAsync(d_keys, h_keys, n * KEY_SIZE, cudaMemcpyHostToDevice, stream), "memcpy keys");
        if (h_hashes) check(cudaMemcpyAsync(d_hashes, h_hashes, n * sizeof(uint64_t), cudaMemcpyHostToDevice, stream), "memcpy hashes");
        
        //run 
        batch_lookup_dev(table,d_keys,d_hashes,d_vals,d_found,n,stream);
        //copy out
        check(cudaMemcpyAsync(h_vals_out, d_vals, n * VALUE_SIZE, cudaMemcpyDeviceToHost, stream), "memcpy vals out");
        check(cudaMemcpyAsync(h_found_out, d_found, n * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream), "memcpy found out");    
        
        //cleanup
        cudaFreeAsync(d_keys,stream);
        cudaFreeAsync(d_vals,stream);
        cudaFreeAsync(d_found,stream);
        if(d_hashes) cudaFreeAsync(d_hashes,stream);
    
        check(cudaStreamSynchronize(stream),"lookup sync failed");
    }
}