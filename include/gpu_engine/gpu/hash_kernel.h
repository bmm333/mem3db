#pragma once
#include <cstdint>

struct HashJob{
    uint8_t* input;
    uint8_t* output;
    size_t length;
};

void launch_hash_kernel(HashJob* jobs, size_t job_count);