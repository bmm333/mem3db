class DeviceContext {
    public:
        static DeviceContext& instance();
        cudaStream_t stream();
        int device_id();
    private:
        DeviceContext();
        cudaStream_t stream_;
};

//Removing cuda boilerplate code per ne wedits;