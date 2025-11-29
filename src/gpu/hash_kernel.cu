#include "include/gpu_engine/gpu/hash_kernel.h
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <stdexcept>

// xxHash-like primes
constexpr uint64_t PRIME64_1 = 0x9E3779B185EBCA87ULL;
constexpr uint64_t PRIME64_2 = 0xC2B2AE3D27D4EB4FULL;
constexpr uint64_t PRIME64_3 = 0x165667B19E3779F9ULL;
constexpr uint64_t PRIME64_4 = 0x85EBCA77C2B2AE63ULL;
constexpr uint64_t PRIME64_5 = 0x27D4EB2F165667C5ULL;

#if defined(__CUDA_ARCH__)
  #define HOST_DEVICE __host__ __device__
#else
  #define HOST_DEVICE
#endif

// safe rotate
HOST_DEVICE static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

// safe 8-byte load (works with unaligned pointers)
HOST_DEVICE static inline uint64_t load_u64_dev(const uint8_t* p) {
    uint64_t v = 0;
    // copy bytes to avoid UB on unaligned access
    // unroll for performance; GPU compiler will optimize
    v  = (uint64_t)p[0];
    v |= (uint64_t)p[1] << 8;
    v |= (uint64_t)p[2] << 16;
    v |= (uint64_t)p[3] << 24;
    v |= (uint64_t)p[4] << 32;
    v |= (uint64_t)p[5] << 40;
    v |= (uint64_t)p[6] << 48;
    v |= (uint64_t)p[7] << 56;
    return v;
}

HOST_DEVICE static inline uint32_t load_u32_dev(const uint8_t* p) {
    uint32_t v = 0;
    v  = (uint32_t)p[0];
    v |= (uint32_t)p[1] << 8;
    v |= (uint32_t)p[2] << 16;
    v |= (uint32_t)p[3] << 24;
    return v;
}

HOST_DEVICE static inline uint64_t round64(uint64_t acc, uint64_t input) {
    acc += input * PRIME64_2;
    acc = rotl64(acc, 31);
    acc *= PRIME64_1;
    return acc;
}

HOST_DEVICE static inline uint64_t merge_round64(uint64_t acc, uint64_t val) {
    val = round64(0, val);
    acc ^= val;
    acc = acc * PRIME64_1 + PRIME64_4;
    return acc;
}

HOST_DEVICE static inline uint64_t fmix64_dev(uint64_t h) {
    h ^= h >> 33;
    h *= PRIME64_2;
    h ^= h >> 29;
    h *= PRIME64_3;
    h ^= h >> 32;
    return h;
}

// Device-side xxHash64-ish implementation (safe for GPU)
__device__ uint64_t xxhash64_device(const void* input, size_t len, uint64_t seed = 0) {
    const uint8_t* p = static_cast<const uint8_t*>(input);
    const uint8_t* const bEnd = p + len;
    uint64_t h64;

    if (len >= 32) {
        const uint8_t* const limit = bEnd - 32;
        uint64_t v1 = seed + PRIME64_1 + PRIME64_2;
        uint64_t v2 = seed + PRIME64_2;
        uint64_t v3 = seed + 0;
        uint64_t v4 = seed - PRIME64_1;

        do {
            uint64_t k1 = load_u64_dev(p); p += 8;
            uint64_t k2 = load_u64_dev(p); p += 8;
            uint64_t k3 = load_u64_dev(p); p += 8;
            uint64_t k4 = load_u64_dev(p); p += 8;
            v1 = round64(v1, k1);
            v2 = round64(v2, k2);
            v3 = round64(v3, k3);
            v4 = round64(v4, k4);
        } while (p <= limit);

        h64 = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h64 = merge_round64(h64, v1);
        h64 = merge_round64(h64, v2);
        h64 = merge_round64(h64, v3);
        h64 = merge_round64(h64, v4);
    } else {
        h64 = seed + PRIME64_5;
    }

    h64 += (uint64_t)len;

    // remaining 8-byte chunks
    while (p + 8 <= bEnd) {
        uint64_t k1 = round64(0, load_u64_dev(p));
        h64 ^= k1;
        h64 = rotl64(h64, 27) * PRIME64_1 + PRIME64_4;
        p += 8;
    }

    // remaining 4-byte chunk
    if (p + 4 <= bEnd) {
        uint32_t k1 = load_u32_dev(p);
        h64 ^= (uint64_t)k1 * PRIME64_1;
        h64 = rotl64(h64, 23) * PRIME64_2 + PRIME64_3;
        p += 4;
    }

    // last bytes
    while (p < bEnd) {
        h64 ^= (uint64_t)(*p) * PRIME64_5;
        h64 = rotl64(h64, 11) * PRIME64_1;
        p++;
    }

    h64 = fmix64_dev(h64);
    return h64;
}

// ---------- Kernel ----------
// Simple kernel where each thread handles one job.
// HashJob should contain device pointers to input buffer and output buffer.
__global__ static void hash_kernel_device(HashJob* jobs, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const HashJob job = jobs[idx]; // copy
    // job.input is device pointer to bytes; job.output is pointer to at least 8 bytes
    uint64_t h = xxhash64_device(job.input, job.length, 0);
    // store result (safe write of 8 bytes)
    uint64_t* out64 = reinterpret_cast<uint64_t*>(job.output);
    out64[0] = h;
}

// ---------- Host wrapper ----------
static inline void cuda_check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + " : " + cudaGetErrorString(e));
    }
}

// Launch with optional stream (0 = default stream)
void launch_hash_kernel(HashJob* d_jobs, size_t count, cudaStream_t stream = 0) {
    if (count == 0) return;
    const int block = 128; // tune
    const int grid = (int)((count + block - 1) / block);

    hash_kernel_device<<<grid, block, 0, stream>>>(d_jobs, count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("hash_kernel launch failed: ") + cudaGetErrorString(err));
    }
    // optional sync here for simple API; callers can pass stream and synchronize themselves
    if (stream == 0) {
        cuda_check(cudaDeviceSynchronize(), "hash_kernel sync failed");
    }
}
