#include <cstdint>

/// Logarithm of number of memory banks (assuming 32).
constexpr size_t LOG_NB_BANKS = 5;

/// Update each element of the output buffer by adding the partial sum of the previous block.
extern "C" __global__
auto add_block_sums(
    int32_t const* d_in,
    size_t len,
    int32_t* d_out,
    int32_t const* block_sums,
    size_t _len2
) -> void {
    size_t idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    int32_t block_sum = block_sums[blockIdx.x];
    if (idx < len) {
        d_out[idx] = d_in[idx] + block_sum;
        if (idx + blockDim.x < len) {
            d_out[idx + blockDim.x] = d_in[idx + blockDim.x] + block_sum;
        }
    }
}

/// Modified version of Mark Harris' implementation of the Blelloch scan according to 
/// `https://www.mimuw.edu.pl/~ps209291/kgkp/slides/scan.pdf`.
extern "C" __global__
auto scan(
    int32_t const* d_in,
    size_t len,
    int32_t* d_out,
    int32_t* block_sums,
    size_t block_size,
    size_t smem_size
) -> void {
    // Allocated on invocation
    extern __shared__ int32_t sdata[];

    size_t tid = threadIdx.x;
    size_t ai = tid;
    size_t bi = tid + blockDim.x;

    // Zero out the shared memory (especially helpful when input size is not power of two)
    sdata[ai] = 0;
    sdata[bi] = 0;
    if (tid + block_size < smem_size) {
        sdata[tid + block_size] = 0;
    }
    __syncthreads();

    // Copy `d_in` to shared memory
    // NOTE: the input elements are scattered into shared memory in light of avoiding bank conflicts.
    size_t idx = threadIdx.x + block_size * blockIdx.x;
    if (idx < len) {
        sdata[ai + (ai >> LOG_NB_BANKS)] = d_in[idx];
        if (idx + blockDim.x < len) {
            sdata[bi + (bi >> LOG_NB_BANKS)] = d_in[idx + blockDim.x];
        }
    }

    // For both upsweep and downsweep:
    // Sequential indices with conflict free padding.
    // Amount of padding: target index / number of banks
    // This "shifts" the target indices by one every multiple of the number of banks.
    //
    // `offset` controls the stride and starting index of the target element at every iteration.
    // `mask` controls which threads are active.
    // Sweeps are pivoted on the last element of shared memory.

    // Upsweep/reduce step
    size_t offset = 1;
    size_t mask = block_size >> 1;
    while (mask > 0) {
        __syncthreads();
        if (tid < mask) {
            size_t ai = offset * ((tid << 1) + 1) - 1;
            size_t bi = offset * ((tid << 1) + 2) - 1;
            ai += (ai >> LOG_NB_BANKS);
            bi += (bi >> LOG_NB_BANKS);
            sdata[bi] += sdata[ai];
        }
        offset <<= 1;
        mask >>= 1;
    }

    // Save the total sum on the global block sums array and clear the last element in shared memory
    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[block_size - 1 + ((block_size - 1) >> LOG_NB_BANKS)];
        sdata[block_size - 1 + ((block_size - 1) >> LOG_NB_BANKS)] = 0;
    }

    // Downsweep step
    mask = 1;
    offset >>= 1;
    while (mask < block_size) {
        __syncthreads();
        if (tid < mask) {
            size_t ai = offset * ((tid << 1) + 1) - 1;
            size_t bi = offset * ((tid << 1) + 2) - 1;
            ai += (ai >> LOG_NB_BANKS);
            bi += (bi >> LOG_NB_BANKS);

            int32_t tmp = sdata[ai];
            sdata[ai] = sdata[bi];
            sdata[bi] += tmp;
        }

        mask <<= 1;
        offset >>= 1;
    }
    __syncthreads();

    // Copy contents of shared memory to global memory
    if (idx < len) {
        d_out[idx] = sdata[ai + (ai >> LOG_NB_BANKS)];
        if (idx + blockDim.x < len) {
            d_out[idx + blockDim.x] = sdata[bi + (bi >> LOG_NB_BANKS)];
        }
    }
}
