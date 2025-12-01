#include "include/gpu_engine/storage/layout.h"
#include "include/gpu_engine/execution/hash_router.h"
#include "include/gpu_engine/storage/layout.h"
#include "include/gpu_eingine/gpu/gpu_memory.h"
#include <vector>
#include <random>
#include <iostream>
#include <chrono>
#include <cstring>

//constats for test
constexpr size_t TEST_CAPACITY=1<<20 //1 million slots (PW Of 2)
constexpr size_t TEST_BATC=100000; //100k itesm per batch
constexpr int NUM_BATCHES=5; //number of batches to test


//helper random data generator
void generate_random_data(std::vector<uint8_t>& keys, std::vector<uint8_t>& vals, size_t count)
{
    keys.resize(count*KEY_SIZE);
    vals.resize(count*VALUES_SIZE);
    //fast pseudo random
    for(size_t i=0;i<count;++i)
    {
        //fill key (first 8 bytes is id rest garbage)
        size_t k_offset=i*KEY_SIZE;
        uint64_t id=i+1; //avoid 0 key
        std::memcpy(&keys[k_offset],&id,sizeof(id));
        //fill value (first 8 bytes is id *10)
        size_t v_offset=i*VALUES_SIZE;
        uint64_t val=id*10;
        std::memcpy(&vals[v_offset],&val,sizeof(val));
    }
}

int main()
{
    try{
        std::cout<<"[INIT] Starting GPU Engine..."<<std::endl;
        //1 initialize vram arena (allocating 2gb for testing)
        GPUMemoryArena::instance().initialize(2ULL * 1024 * 1024 * 1024);
        / 2. Allocate Table
        std::cout << "[Init] Allocating SoA Table (" << TEST_CAPACITY << " slots)..." << std::endl;
        SoALayout table = allocate_soa(TEST_CAPACITY, 256, 256); // cap, max_probe, batch

        // 3. Create Router
        std::cout << "[Init] Creating HashRouter..." << std::endl;
        HashRouter router(table, TEST_BATCH);

        // 4. Run Workload
        auto start_total = std::chrono::high_resolution_clock::now();
        size_t total_inserted = 0;

        for (int b = 0; b < NUM_BATCHES; ++b) {
            std::cout << "\n--- Batch " << b + 1 << " / " << NUM_BATCHES << " ---" << std::endl;

            // A. Generate Data
            std::vector<uint8_t> keys, vals;
            generate_random_data(keys, vals, TEST_BATCH);

            // B. PUT
            auto t0 = std::chrono::high_resolution_clock::now();
            size_t success = router.put_batch(keys.data(), vals.data(), TEST_BATCH);
            auto t1 = std::chrono::high_resolution_clock::now();
            
            double ms_put = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cout << "PUT: " << success << " items in " << ms_put << " ms (" 
                      << (size_t)(TEST_BATCH / (ms_put / 1000.0) / 1e6) << " M ops/sec)" << std::endl;
            
            total_inserted += success;

            // C. GET (Verify)
            std::vector<uint8_t> results(TEST_BATCH * VALUE_SIZE);
            std::vector<bool> found(TEST_BATCH); // vector<bool> is weird but works here, ideally use uint8_t array

            // Fix for API: HashRouter expects bool* or uint8_t* for found mask. 
            // Let's use a uint8_t buffer for safety with the raw pointer API we defined.
            std::vector<uint8_t> found_bytes(TEST_BATCH);

            auto t2 = std::chrono::high_resolution_clock::now();
            // Casting found_bytes.data() to bool* works because bool is 1 byte, but strictly better to use bool array.
            // For this test, let's cast.
            router.get_batch(keys.data(), TEST_BATCH, results.data(), reinterpret_cast<bool*>(found_bytes.data()));
            auto t3 = std::chrono::high_resolution_clock::now();

            double ms_get = std::chrono::duration<double, std::milli>(t3 - t2).count();
            std::cout << "GET: Retrieved in " << ms_get << " ms (" 
                      << (size_t)(TEST_BATCH / (ms_get / 1000.0) / 1e6) << " M ops/sec)" << std::endl;

            // D. Validate Data Integrity
            size_t valid_count = 0;
            for (size_t i = 0; i < TEST_BATCH; ++i) {
                if (found_bytes[i]) {
                    uint64_t actual_val = *reinterpret_cast<uint64_t*>(&results[i * VALUE_SIZE]);
                    uint64_t expected_val = *reinterpret_cast<uint64_t*>(&vals[i * VALUE_SIZE]);
                    if (actual_val == expected_val) {
                        valid_count++;
                    } else {
                        // std::cerr << "Mismatch at " << i << ": " << actual_val << " vs " << expected_val << std::endl;
                    }
                }
            }
            std::cout << "VALIDATION: " << valid_count << " / " << TEST_BATCH << " match correctly." << std::endl;
        }

        auto end_total = std::chrono::high_resolution_clock::now();
        std::cout << "\n[Done] Total time: " << std::chrono::duration<double>(end_total - start_total).count() << "s" << std::endl;

        // Cleanup
        free_soa(table);
    }catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}