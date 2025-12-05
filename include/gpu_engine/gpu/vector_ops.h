#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "storage/layout.h"

namespace gpu{
    //Vector search result
    struct SearchResult{
        float score;      //Similarity score
        uint64_t id;     //Vector ID
    }

    
    / Calculate Dot Product of a query vector against ALL items in the table
    // table: The SoA table containing the vectors (in table.values)
    // query_vector: Device pointer to the single query vector (128 bytes / 32 floats)
    // scores_out: Device pointer to output array of floats (size = capacity)
    // count: The current capacity or number of items to scan
    void vector_search_dev(SoALayout table, 
                        const float* query_vector, 
                        float* scores_out, 
                        size_t count, 
                        cudaStream_t stream = 0);

    }
}