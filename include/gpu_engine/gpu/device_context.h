#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <mutex>

class DeviceContext{
    public:
        static DeviceContext& instance(){
            static DeviceContext ctx;
            return ctx;
        }
        int device_id() const{return device_id_;}
        //returns stream for operations; simple RR for now
        cudaStream_t stream()
        {
            cudaStream_t s=streams_[next_stream_];
            next_stream_=(next_stream_+1)%streams_.size();
            return s;
        }
    private:
        DeviceContext()
        {
            int count;
            cudaGetDeviceCount(&count);
            if(count==0) throw std::runtime_error("No CUDA devices found");

            //Picks Gpu with max free memory idk how much performance this will add but theoretically should be better
            size_t max_free=0;
            for(int i=0;i<count;++i)
            {
                size_t free,total;
                cudaSetDevice(i);
                cudaMemGetInfo(&free,&total);
                if(free>max_free)
                {
                    max_free=free;
                    device_id_=i;
                }
            }
            cudaSetDevice(device_id_);
            //create multiple streams for parallelism
            const int N_streams=4;
            streams_.resize(N_streams);
            for(auto& s:streams_)
            {
                cudaStreamCreate(&s);
            }
            next_stream_=0;
        }
        ~DeviceContext()
        {
            for(auto s:streams_)
            {
                cudaStreamDestroy(s);
            }
        }
        int device_id_;
        std::vector<cudaStream_t> streams_;
        size_t next_stream_;
};