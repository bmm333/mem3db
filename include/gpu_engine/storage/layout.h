#pragma once
#include <cstdint>
#include <cstddef>

//Layout constants

//Fixed size key/value (fine tuner later when benchmarking is ready)
static constexpr size_t KEY_SIZE=128; //bytes
static constexpr size_t VALUE_SIZE=128; //bytes

//flags per entry
enum EntryFlag:uint8_t{
    EMPTY=0,
    FILLED=1,
    TOMBSTONE=2,
};
//gpu alignment rules
static constexpr size_t ALIGNMENT=128; //matching key size &^gpu l2 line

//SoA table POD describing mem layout

struct SoALayout{
    //Column buffers in GPU VRAM (device pointers)
    uint8_t* keys; //[capacity* key_size]
    uint8_t* values; //[capacity* value_size]
    uint64_t* hashes; //[capacity]
    uint8_t* flags; //[capacity]
    //MetaData
    size_t capacity; //power of 2
    size_t mask; //capacity-1
    //Probe settings (Used y GPU kernels)
    uint32_t max_probes;
    uint32_t batch_size; //kernel batch size;

    __host__ __device__
    size_t index(size_t hash)const noexcept{
        return hash & mask;
    }
};

//Layout size computation

inline size_t keys_bytes(size_t capacity){return capacity*KEY_SIZE;}
inline size_t values_bytes(size_t capacity){return capacity*VALUE_SIZE;}
inline size_t hashes_bytes(size_t capacity){return capacity*sizeof(uint64_t);}
inline size_t flags_bytes(size_t capacity){return capacity*sizeof(uint8_t);}
//total gpu footprint (logging debugging and admisions)
inline size_t total_bytes(size_t capacity){
    return keys_bytes(capacity)
        +values_bytes(capacity)
        +hashes_bytes(capacity)
        +flags_bytes(capacity);
}