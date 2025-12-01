#pragma once
#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>
#include "storage/layout.h"

//forward declarations
namespace gpu{
        void batch_insert_dev(SoALayout, const uint8_t*, const uint8_t*, const uint64_t*, size_t, uint8_t*, cudaStream_t);
        void batch_lookup_dev(SoALayout, const uint8_t*, const uint64_t*, uint8_t*, uint8_t*, size_t, cudaStream_t);
}

class HashRouter{
    public:
        HashRouter(SoaLayout table,size_t max_batch_size);
        ~HashRouter();

        //API 
        //keys: pointer to packed keys ( count* key_size)
        //values: pointer to packed values (count*values_size)
        //count: number of items
        //returns number of successful inserts
        size_t put_batch(const uint8_t* keys,const uint8_t* values,size_t count);

        //results preallocated buffer (count* value_size)
        //found preallocated buffer count*sizeof(bool)
        void get_batch(const uint8_t* keys,size_t count,uint8_t* results,bool* found);

    private:
        SoALayout table_;
        size_t max_batch_;
        cudaStream_t stream_;
        //Pinned host memory;
        uint8_t* h_stage_keys;
        uint8_t* h_stage_values;
        uint8_t* h_stage_status;

        //device memory
        uint8_t* d_stage_keys;
        uint8_t* d_stage_vals_;
        uint8_t* d_stage_status_;
};
