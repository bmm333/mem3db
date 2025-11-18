#include "inmemdb/storage/hash_table_backend.h"
#include <iostream>
#include <chrono>

int main()
{
    HashTableConfig config;
    config.initial_capacity = 16384;
    HashTableBackend backend(config);
    //basic testing
    backend.put("name","Ben");
    backend.put("age","22");
    std::cout<<"name":<<backend.get("name")<<std::endl;
    std::cout<<"age":<<backend.get("age")<<std::endl;
    //Benchmarking
    auto start=std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000000; ++i) {
        char key[32];
        snprintf(key, sizeof(key), "key_%d", i);
        backend.put(key, "value");
    }
    auto end=std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout<<"\nCpu baseline Benchmakr:"<<std::endl;
    std::cout << "Inserted 1M entries in " << duration.count() << " ms" << std::endl;
    std::cout << "Throughput: " << (1000000.0 / duration.count() * 1000) << " ops/sec" << std::endl;
    std::cout << "Load factor: " << backend.load_factor() << std::endl;
    std::cout << "Inline count: " << backend.inline_count() << std::endl;

    return 0;
}