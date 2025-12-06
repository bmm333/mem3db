#pragma once
#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>
#include "storage/layout.h"

//forward declarations
namespace gpu{
        void batch_insert_dev(SoALayout, const uint8_t*, const uint8_t*, const uint64_t*, size_t, uint8_t*, cudaStream_t);
        void batch_lookup_dev(SoALayout, const uint8_t*, const uint64_t*, uint8_t*, uint8_t*, size_t, cudaStream_t);
        void vector_search_dev(SoaLayout,const float*,float*,size_t,cudaStream_t);
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

        //Vector Search API
        //query_vector: 32 floats (128 bytes)
        //top_k: How many results to return
        //out_keys: Buffer for result keys(top_k * key_size)
        //out_scores: Buffer for result scores (top_k * sizeof(float))
        void vector_search(const float* query_vector,int top_k,uint8_t* out_keys,float* out_scores);

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

        //search buffer (Pre-alloc for speed)
        float* d_search_query_; //1 vector on gpu
        float* d_search_scores_; //array of scores (size=capacity)
        float* h_search_scores_; //Array of scores on CPU (for sorting)
};
