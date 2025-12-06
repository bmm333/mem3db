#include "include/gpu_engine/execution/hash_router.h"
#include "include/gpu_engine/gpu/device_context.h"
#include "include/gpu_engine/gpu/insert_kernel.h"
#include "include/gpu_engine/gpu/lookup_kernel.h"
#include "include/gpu_engine/gpu/vector_ops.h"
#include <cstring>
#include <stdexcept>
#include <algorithm> //partial sort
#include <iostream>

static void check(cudaError_t err,const char* msg)
{
    if(e!=cudaSuccess) throw std::runtime_error(std::string(msg)+": "+cudaGetErrorString(err));
}
//helper struct for sorting
struct ScoreIndex{
    float score;
    size_t index;
};

HashRouter::HashRouter(SoALayout table,size_t max_batch_size):table_(table),max_batch_(max_batch_size)
{
    stream_ = DeviceContext::instance().stream();

    size_t keys_sz   = max_batch_ * KEY_SIZE;
    size_t vals_sz   = max_batch_ * VALUE_SIZE;
    size_t status_sz = max_batch_ * sizeof(uint8_t);
    check(cudaMallocHost(&h_stage_keys_, keys_sz), "Host alloc keys");
    check(cudaMallocHost(&h_stage_vals_, vals_sz), "Host alloc vals");
    check(cudaMallocHost(&h_stage_status_, status_sz), "Host alloc status");
    check(cudaMalloc(&d_stage_keys_, keys_sz), "Dev alloc keys");
    check(cudaMalloc(&d_stage_vals_, vals_sz), "Dev alloc vals");
    check(cudaMalloc(&d_stage_status_, status_sz), "Dev alloc status");
    size_t total_capacity = table.capacity; 
    check(cudaMalloc(&d_search_query_, 128), "Dev alloc search query"); // 32 floats
    check(cudaMalloc(&d_search_scores_, total_capacity * sizeof(float)), "Dev alloc search scores");
    check(cudaMallocHost(&h_search_scores_, total_capacity * sizeof(float)), "Host alloc search scores");
}

HashRouter::~HashRouter()
{
    cudaFreeHost(h_stage_keys_);
    cudaFreeHost(h_stage_vals_);
    cudaFreeHost(h_stage_status_);
    cudaFree(d_stage_keys_);
    cudaFree(d_stage_vals_);
    cudaFree(d_stage_status_);
    cudaFree(d_search_query_)
    cudaFree(d_search_scores_);
    cudaFreeHost(h_search_scores_);
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
void HashRouter::search_vector(const float* query_vector,int top_k,uint8_t* out_keys,float* out_scores)
{
    // 1. copy query to gpu 
    check(cudaMemcpyAsync(d_search_query_,query_vector,32*sizeof(float),cudaMemcpyHostToDevice,stream_),"DMA Query");
    //2 Launch Kernel (Brute force Scan)
    //Scan entire capacity Fast on gpu
    gpu::vector_search_dev(table_,d_search_query_,d_search_scores_,table_.capacity,stream_);

    //3 Copy back to cpu (pinned memory)
    //Note: Copying 4MB(1m floats) is fast ~ 1ms
    check(cudaMemcpyAsync(h_search_scores_,d_search_scores_,table_.capacity*sizeof(float),cudaMemcpyDeviceToHost,stream_),"DMA Scores back");
    check(cudaStreamSynchronize(stream_),"Sync search");

    //4 Cpu Top-k (partial sort)
    //iterate the scores array. If score > -infinity we considere it
    std::vector<ScoreIndex> candidates;
    candidates.reserve(table_.capacity/2); //heuristic reserve
    for (size_t i = 0; i < table_.capacity; ++i) {
        float s = h_search_scores_[i];
        if (s > -900000.0f) { 
            candidates.push_back({s, i});
        }
    }
    //sort to find top k
    size_t k=std::min((size_t)top_k,candidates.size());
    std::partial_sort(candidates.begin(),candidates.begin()+k,candidates.end(),[](const ScoreIndex& a,const ScoreIndex& b){return a.score>b.score;}); //descending (higher sim is better)
    
    //5 Gather keys
    //we have indices, need the keys
    //since we dont have cpu mirror of keys , we fetch them directly from the gpu
    //optimization: on real system we'd batch this. here we do k small transfers
    for(size_t i=0;i<k;++i)
    {
        size_t idx=candidates[i].index;
        out_scores[i]=candidates[i].score;
        //fetch key from gpu table
        const uint8_t* d_key_ptr=table_.keys+(idx*KEY_SIZE);
        //blocking copy for simplicity (its small)
        cudaMemcpy(h_key_ptr,d_key_ptr,KEY_SIZE,cudaMemcpyDeviceToHost);
    }
}
