#include "scan.h"
#include "timer.h"
#include "utils.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <iostream>

constexpr size_t BLOCK_SIZE = 1024;
constexpr size_t LOG_NB_BANKS = 5;

/// Update each element of the output buffer by adding the partial sum of the previous block.
__global__ 
auto add_block_sums(
    int32_t* d_out,
    int32_t const* d_in,
    size_t len,
    int32_t const* block_sums,
    size_t _useless
) -> void {
    int32_t block_sum = block_sums[blockIdx.x];
    size_t idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        d_out[idx] = d_in[idx] + block_sum;
        if (idx + blockDim.x < len) {
            d_out[idx + blockDim.x] = d_in[idx + blockDim.x] + block_sum;
        }
    }
}

/// Modified version of Mark Harris' implementation of the Blelloch scan according to 
/// `https://www.mimuw.edu.pl/~ps209291/kgkp/slides/scan.pdf`.
__global__ auto scan(
    int32_t* d_out,
    int32_t const* d_in,
    size_t len,
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
    // Note that the input elements are scattered into shared memory in light of avoiding bank
    // conflicts.
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

auto scan_inner(
    int32_t* d_out,
    int32_t const* d_in,
    size_t len
) -> void {
    // Zero out `d_out`
    checkCudaErrors(cudaMemset(d_out, 0, len * sizeof(int32_t)));

    // Set up number of threads and blocks
    size_t block_size = BLOCK_SIZE / 2;
    size_t max_elems_per_block = block_size * 2;
    size_t grid_size = len / max_elems_per_block;
    if (len % max_elems_per_block != 0) {
        grid_size += 1;
    }
    size_t smem_size = max_elems_per_block + ((max_elems_per_block - 1) >> LOG_NB_BANKS);

    // Allocate memory for array of total sums produced by each block.
    // Buffer length must be the same as number of blocks.
    int32_t* block_sums;
    checkCudaErrors(cudaMalloc(&block_sums, grid_size * sizeof(int32_t)));
    checkCudaErrors(cudaMemset(block_sums, 0, grid_size * sizeof(int32_t)));

    // Sum scan data allocated to each block
    scan<<<grid_size, block_size, smem_size * sizeof(int32_t)>>>(
        d_out,
        d_in,
        len,
        block_sums,
        block_size,
        smem_size
    );

    // Sum scan total sums produced by each block.
    // Use basic implementation if the number of total sums is <= `block_size` (this only requires
    // one block to do the scan).
    // Otherwise, recurse on this same function as we'll need the full scan for the block sums.
    if (grid_size <= max_elems_per_block) {
        int32_t* dummy;
        checkCudaErrors(cudaMalloc(&dummy, sizeof(int32_t)));
        checkCudaErrors(cudaMemset(dummy, 0, sizeof(int32_t)));

        scan<<<1, block_size, smem_size * sizeof(int32_t)>>>(
            block_sums,
            block_sums,
            grid_size,
            dummy,
            max_elems_per_block,
            smem_size
        );
        checkCudaErrors(cudaFree(dummy));
    } else {
        int32_t* in_block_sums = nullptr;
        checkCudaErrors(cudaMalloc(&in_block_sums, grid_size * sizeof(int32_t)));
        checkCudaErrors(cudaMemcpy(
            in_block_sums,
            block_sums,
            grid_size * sizeof(int32_t),
            cudaMemcpyDeviceToDevice
        ));

        scan_inner(block_sums, in_block_sums, grid_size);
        checkCudaErrors(cudaFree(in_block_sums));
    }

    // // NOTE: Uncomment to examine block sums.
    // int32_t* h_block_sums = new int32_t[grid_size];
    // checkCudaErrors(cudaMemcpy(
    //     h_block_sums,
    //     block_sums,
    //     grid_size * sizeof(int32_t),
    //     cudaMemcpyDeviceToHost
    // ));
    // std::cout << "Block sums: ";
    // for (size_t i = 0; i < grid_size; ++i) {
    //     std::cout << h_block_sums[i] << ", ";
    // }
    // std::cout << std::endl;
    // std::cout << "Block sums length: " << grid_size << std::endl;
    // delete[] h_block_sums;

    // Add each block's total sum to its scan output in order to get the final, global scanned array
    add_block_sums<<<grid_size, block_size>>>(d_out, d_out, len, block_sums, /* useless */ grid_size);

    checkCudaErrors(cudaFree(block_sums));
}
