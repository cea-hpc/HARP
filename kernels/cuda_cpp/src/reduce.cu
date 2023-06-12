#include <cstdint>

/// Block size.
constexpr size_t BS = 1024;
/// Warp size.
constexpr size_t WS = 32;
/// Mask to select all threads in a warp.
constexpr uint32_t FULL_MASK = 0xFFFFFFFF;

/// Sum reduction operator for 32-bit integer vectors.
/// This implementation assumes vectors of length `n` a power of two and a kernel block size of 1024
/// threads/block.
///
/// This kernel performs a reduction for each thread block, meaning that a host-side reduction is
/// necessary to compute the final result.
extern "C" __global__
auto reduce(int32_t* gdata_in, size_t n, int32_t* gdata_out) -> void {
    __shared__ int32_t sdata[BS];

    size_t const tid = threadIdx.x;
    size_t grid_size = (BS * gridDim.x) << 1;

    // Perform first level of reduction while fetching data from global memory and storing it into
    // shared memory.
    int32_t acc = 0;
    size_t idx = tid + (2 * BS) * blockIdx.x;
    while (idx < n) {
        acc += gdata_in[idx];
        if (idx + BS < n) {
            acc += gdata_in[idx + BS];
        }
        idx += grid_size;
    }

    sdata[tid] = acc;
    __syncthreads();

    // Perform the remaining steps of the reduction
    if (BS >= 1024 && tid < 512) {
        acc += sdata[tid + 512];
        sdata[tid] = acc;
    }
    __syncthreads();
    if (BS >= 512 && tid < 256) {
        acc += sdata[tid + 256];
        sdata[tid] = acc;
    }
    __syncthreads();
    if (BS >= 256 && tid < 128) {
        acc += sdata[tid + 128];
        sdata[tid] = acc;
    }
    __syncthreads();
    if (BS >= 128 && tid < 64) {
        acc += sdata[tid + 64];
        sdata[tid] = acc;
    }
    __syncthreads();

    // For the last step, perform a warp-level reduction using the `__shfl_down_sync` intrinsic
    if (tid < WS) {
        if (BS >= WS * 2) {
            acc += sdata[tid + WS];
        }
        for (size_t offset = WS / 2; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(FULL_MASK, acc, offset, WS);
        }
    }

    // Write result back to global memory
    if (tid == 0) {
        gdata_out[blockIdx.x] = acc;
    }
}
