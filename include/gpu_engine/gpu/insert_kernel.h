#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include "../storage/layout.h" //soa layout , key size value size and helpers

namespace gpu{
    //insert status code device/host
    enum InsertStatus:uint8_t{
        INSERT_OK=0, //inserted or ow with success
        INSERT_OVERWRITE=1, //ow existing key
        INSERT_FAIL=2  //failed (should fail only on max probes/table full(which shouldnt per design happen))
    };
    //batched device insert API
    // - table: SoA layout describing device-resident table
    // - keys_dev: device pointer to keys packed (n*KEY_SIZE)
    // - vals_dev: device pointer to values packed (n*VALUE_SIZE)
    // - hashes_dev: optional device pointer to precomputed hashes (n) or nullptr
    // - status_dev: optional device pointer to status array (n bytes) or nullptr
    // - stream: cuda stream to launch kernel on
    // - NOTE : API assumes keys_dev/vals_dev are device pointers and already on GPU
    // the function wont allocate temporary device memory
    void batch_insert_device(SoALayout table,
                             const uint8_t* keys_dev,
                             const uint8_t* vals_dev,
                             const uint64_t* hashes_dev,
                             size_t n,
                             uint8_t* status_dev,
                             cudaStream_t stream=0);
    //convenience host wrapper that accepts host pointers and staging buffers.
    //implementations should avoid calling this in hot path (use preallocated device staging).
    void batch_insert_host(SoALayout table,
                            const uint8_t* host_key_packed,
                            const uint8_t* host_vals_packed,
                            const uint64_t* host_hashes,//opt can be nullptr
                            size_t n,
                            uint8_t* host_status, //opt can be nullptr same as line 35
                            cudaStream_t steam=0 //default);
}//namespace gpu