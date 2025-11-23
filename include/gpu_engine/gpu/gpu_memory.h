#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <atomic>
#include <algorithm>
#include "device_context.h"

//small helper macro for cuda error checking
#define CUDA_CHECK(expr)                                       \
    do {                                                       \
        cudaError_t err = (expr);                              \
        if (err != cudaSuccess) {                              \
            throw std::runtime_error(                          \
                std::string("CUDA Error: ") +                  \
                cudaGetErrorString(err) +                      \
                " at " + __FILE__ + ":" + std::to_string(__LINE__) \
            );                                                 \
        }                                                      \
    } while (0)
    
//Gpu vram arena : one contiguous block, bump-pointer allocator 
class GPUMemoryArena{
    public:
        //Signleton
        static GPUMemoryArena& instance()
        {
            static GPUMemoryArena arena;
            return arena;
        }
        //Reserve vram upfront
        void initialize(size_t byte)
        {
            std::lock_guard<std::mutex>lock(mutex_);
            if(initialized_)return;//idempotent
            CUDA_CHECK(cudaSetDevice(DeviceContext::instance().device_id()));
            total_bytes_ = byte;
            CUDA_CHECK(cudaMalloc(&base_ptr_, total_bytes_));
            offset_.store(0);
            initialized_ = true;
        }
        //bump allocation aligned
        void* allocate(size_t bytes,size_t alignment=128)
        {
            if(!initialized_)
            {
                throw std::runtime_error("GPUMemoryArena not initialized");
            }
            size_t aligned=align_up(bytes,alignment);
            size_t old=offset_.fetch_add(aligned);
            if(old+aligned>total_bytes_)
            {
                offset_.fetch_sub(aligned);
                throw std::runtime_error("VRAM AREA OOM");
            }
            return static_cast<char*>(base_ptr_) + old;
        }
        //reset entire arena (rebuild/flush)
        void reset()
        {
            std::lock_guard<std::mutex>lock(mutex_);
            offset_.store(0);
        }
        //Stats
        size_t used()const{return offset_.load();}
        size_t total()const{return total_bytes_;}
        size_t free_bytes()const{return total_bytes_ - used();}
        void* base()const{return base_ptr_;}
    private:
        GPUMemoryArena():base_ptr_(nullptr),initialized_(false),total_bytes_(0),offset_(0){}
        ~GPUMemoryArena()
        {
            if(base_ptr_)cudaFree(base_ptr_);
        }
        GPUMemoryArena(const GPUMemoryArena&)=delete;
        GPUMemoryArena& operator=(const GPUMemoryArena&)=delete;
        //helpers
        static size_t align_up(size_t x,size_t a)
        {
            return (x+a-1)&~(a-1);
        }
    private:
        void* base_ptr_;
        size_t total_bytes_;
        std::atomic<size_t> offset_;
        bool initialized_;
        mutable std::mutex mutex_;
}