#pragma once
#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>
#include "storage/layout.h"

namespace gpu {
// Lookup Status Codes
enum LookupStatus : uint8_t {
    LOOKUP_FOUND     = 1,
    LOOKUP_NOT_FOUND = 0
};

// Device Wrapper (Fast Path - Data already on GPU)
// - keys_dev: Input keys (n * KEY_SIZE)
// - hashes_dev: Optional pre-computed hashes
// - values_out: Output buffer for values (n * VALUE_SIZE)
// - found_out: Output byte array (1 = found, 0 = not found)
void batch_lookup_dev(SoALayout table,
                      const uint8_t* keys_dev,
                      const uint64_t* hashes_dev,
                      uint8_t* values_out,
                      uint8_t* found_out,
                      size_t n,
                      cudaStream_t stream = 0);

// Host Wrapper (Convenience Path - Data on CPU)
void batch_lookup_host(SoALayout table,
                       const uint8_t* h_keys,
                       const uint64_t* h_hashes,
                       uint8_t* h_values_out,
                       uint8_t* h_found_out,
                       size_t n,
                       cudaStream_t stream = 0);

} // namespace gpu