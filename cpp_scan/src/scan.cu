#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "scan.hpp"
#include "timer.h"
#include "utils.h"

#include <math.h>

#ifdef ZERO_BANK_CONFLICTS
    #define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
    #define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

__global__ void
device_add_block_sums(int32_t* d_out, int32_t const* d_in, int32_t* d_block_sums, size_t numElems) {
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
__global__ void device_scan(
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

void scan(int32_t* d_out, int32_t const* d_in, size_t len) {
    // Zero out d_out
    checkCudaErrors(cudaMemset(d_out, 0, len * sizeof(int32_t)));

    // Set up number of threads and blocks

    uint32_t block_sz = MAX_BLOCK_SZ / 2;
    size_t max_elems_per_block = 2 * block_sz; // due to binary tree nature of algorithm

    // If input size is not power of two, the remainder will still need a whole
    // block Thus, number of blocks must be the ceiling of input size / max elems
    // that a block can handle int32_t grid_sz = (int32_t) std::ceil((double) len
    // / (double) max_elems_per_block); UPDATE: Instead of using ceiling and
    // risking miscalculation due to precision, just automatically add 1 to the
    // grid size when the input size cannot be divided cleanly by the block's
    // capacity
    uint32_t grid_sz = (uint32_t)(len / max_elems_per_block);
    // Take advantage of the fact that integer division drops the decimals
    if (len % max_elems_per_block != 0) {
        grid_sz += 1;
    }

    // Conflict free padding requires that shared memory be more than 2 * block_sz
    size_t shmem_sz = max_elems_per_block + ((max_elems_per_block - 1) >> LOG_NUM_BANKS);

    // Allocate memory for array of total sums produced by each block
    // Array length must be the same as number of blocks
    int32_t* d_block_sums;
    checkCudaErrors(cudaMalloc(&d_block_sums, sizeof(int32_t) * grid_sz));
    checkCudaErrors(cudaMemset(d_block_sums, 0, sizeof(int32_t) * grid_sz));

    // Sum scan data allocated to each block
    device_scan<<<grid_sz, block_sz, sizeof(int32_t) * shmem_sz>>>(
        d_out,
        d_in,
        d_block_sums,
        len,
        shmem_sz,
        max_elems_per_block
    );

    // Sum scan total sums produced by each block
    // Use basic implementation if number of total sums is <= 2 * block_sz
    // (This requires only one block to do the scan)
    if (grid_sz <= max_elems_per_block) {
        int32_t* d_dummy_blocks_sums;
        checkCudaErrors(cudaMalloc(&d_dummy_blocks_sums, sizeof(int32_t)));
        checkCudaErrors(cudaMemset(d_dummy_blocks_sums, 0, sizeof(int32_t)));
        device_scan<<<1, block_sz, sizeof(int32_t) * shmem_sz>>>(
            d_block_sums,
            d_block_sums,
            d_dummy_blocks_sums,
            grid_sz,
            shmem_sz,
            max_elems_per_block
        );
        checkCudaErrors(cudaFree(d_dummy_blocks_sums));
    }
    // Else, recurse on this same function as you'll need the full-blown scan
    // for the block sums
    else {
        int32_t* d_in_block_sums;
        checkCudaErrors(cudaMalloc(&d_in_block_sums, sizeof(int32_t) * grid_sz));
        checkCudaErrors(cudaMemcpy(
            d_in_block_sums,
            d_block_sums,
            sizeof(int32_t) * grid_sz,
            cudaMemcpyDeviceToDevice
        ));
        scan(d_block_sums, d_in_block_sums, grid_sz);
        checkCudaErrors(cudaFree(d_in_block_sums));
    }

    // Add each block's total sum to its scan output
    // in order to get the final, global scanned array
    device_add_block_sums<<<grid_sz, block_sz>>>(d_out, d_out, d_block_sums, len);

    checkCudaErrors(cudaFree(d_block_sums));
}
