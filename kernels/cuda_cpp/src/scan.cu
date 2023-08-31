#include <stdint.h>

#define MAX_BLOCK_SZ 1024
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#ifdef ZERO_BANK_CONFLICTS
    #define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
    #define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

extern "C" __global__
void device_add_block_sums(
    int32_t* d_out,
    int32_t const* d_in,
    int32_t* d_block_sums,
    size_t numElems
) {
    int32_t d_block_sum_val = d_block_sums[blockIdx.x];

    // Simple implementation's performance is not significantly (if at all)
    // better than previous verbose implementation
    size_t cpy_idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    if (cpy_idx < numElems) {
        d_out[cpy_idx] = d_in[cpy_idx] + d_block_sum_val;
        if (cpy_idx + blockDim.x < numElems) {
            d_out[cpy_idx + blockDim.x] = d_in[cpy_idx + blockDim.x] + d_block_sum_val;
        }
    }
}

// Modified version of Mark Harris' implementation of the Blelloch scan
//  according to https://www.mimuw.edu.pl/~ps209291/kgkp/slides/scan.pdf
extern "C" __global__
void device_scan(
    int32_t* d_out,
    int32_t const* d_in,
    int32_t* d_block_sums,
    size_t len,
    size_t shmem_sz,
    size_t max_elems_per_block
) {
    // Allocated on invocation
    extern __shared__ int32_t s_out[];

    size_t thid = threadIdx.x;
    size_t ai = thid;
    size_t bi = thid + blockDim.x;

    // Zero out the shared memory
    // Helpful especially when input size is not power of two
    s_out[thid] = 0;
    s_out[thid + blockDim.x] = 0;
    // If CONFLICT_FREE_OFFSET is used, shared memory
    // must be a few more than 2 * blockDim.x
    if (thid + max_elems_per_block < shmem_sz) {
        s_out[thid + max_elems_per_block] = 0;
    }

    __syncthreads();

    // Copy d_in to shared memory
    // Note that d_in's elements are scattered into shared memory
    // in light of avoiding bank conflicts
    size_t cpy_idx = max_elems_per_block * blockIdx.x + threadIdx.x;
    if (cpy_idx < len) {
        s_out[ai + CONFLICT_FREE_OFFSET(ai)] = d_in[cpy_idx];
        if (cpy_idx + blockDim.x < len) {
            s_out[bi + CONFLICT_FREE_OFFSET(bi)] = d_in[cpy_idx + blockDim.x];
        }
    }

    // For both upsweep and downsweep:
    // Sequential indices with conflict free padding
    // Amount of padding = target index / num banks
    // This "shifts" the target indices by one every multiple
    // of the num banks
    // offset controls the stride and starting index of
    // target elems at every iteration
    // d just controls which threads are active
    // Sweeps are pivoted on the last element of shared memory

    // Upsweep/Reduce step
    size_t offset = 1;
    for (size_t d = max_elems_per_block >> 1; d > 0; d >>= 1) {
        __syncthreads();

        if (thid < d) {
            size_t ai = offset * ((thid << 1) + 1) - 1;
            size_t bi = offset * ((thid << 1) + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            s_out[bi] += s_out[ai];
        }
        offset <<= 1;
    }

    // Save the total sum on the global block sums array
    // Then clear the last element on the shared memory
    if (thid == 0) {
        d_block_sums[blockIdx.x] =
            s_out[max_elems_per_block - 1 + CONFLICT_FREE_OFFSET(max_elems_per_block - 1)];
        s_out[max_elems_per_block - 1 + CONFLICT_FREE_OFFSET(max_elems_per_block - 1)] = 0;
    }

    // Downsweep step
    for (size_t d = 1; d < max_elems_per_block; d <<= 1) {
        offset >>= 1;
        __syncthreads();

        if (thid < d) {
            size_t ai = offset * ((thid << 1) + 1) - 1;
            size_t bi = offset * ((thid << 1) + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            int32_t tmp = s_out[ai];
            s_out[ai] = s_out[bi];
            s_out[bi] += tmp;
        }
    }
    __syncthreads();

    // Copy contents of shared memory to global memory
    if (cpy_idx < len) {
        d_out[cpy_idx] = s_out[ai + CONFLICT_FREE_OFFSET(ai)];
        if (cpy_idx + blockDim.x < len) {
            d_out[cpy_idx + blockDim.x] = s_out[bi + CONFLICT_FREE_OFFSET(bi)];
        }
    }
}