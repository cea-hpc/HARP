#include <cstdint>

constexpr uint64_t BS = 32;
constexpr uint64_t WPT = 8;
constexpr uint64_t RBS = BS / WPT;

extern "C" __global__ auto tiled_dgemm(
    uint64_t m, uint64_t n, uint64_t k,
    double alpha,
    double const* A, uint64_t lda,
    double const* B, uint64_t ldb,
    double beta,
    double* C, uint64_t ldc
) -> void {
    uint64_t const tidx = threadIdx.x;
    uint64_t const tidy = threadIdx.y;
    uint64_t const gidx = BS * blockIdx.x + tidx;
    uint64_t const gidy = BS * blockIdx.y + tidy;

    __shared__ double Ab[BS][BS];
    __shared__ double Bb[BS][BS];

    // Initialize the accumulation registers
    double acc[WPT];
    #pragma unroll
    for (size_t w = 0; w < WPT; ++w) {
        acc[w] = 0.0f;
    }

    size_t const nb_blocks = k / BS;
    for (size_t t = 0; t < nb_blocks; ++t) {
        #pragma unroll
        for (size_t w = 0; w < WPT; ++w) {
            size_t const bidx = BS * t + tidx;
            size_t const bidy = BS * t + tidy;

            Ab[tidy + w * RBS][tidx] = A[(bidy + w * RBS) * lda + gidx];
            Bb[tidy + w * RBS][tidx] = B[(gidy + w * RBS) * ldb + bidx];
        }
        __syncthreads();

        for (size_t l = 0; l < BS; ++l) {
            for (size_t w = 0; w < WPT; ++w) {
                acc[w] += Ab[l][tidx] * Bb[tidy + w * RBS][l];
            }
        }
        __syncthreads();
    }

    for (size_t w = 0; w < WPT; ++w) {
        C[(gidy + w * RBS) * ldc + gidx] *= beta + alpha * acc[w];
    }
}
