#include <cstdint>

constexpr uint64_t BLOCK_SIZE = 32;

extern "C" __global__
auto tiled_dgemm(
    uint64_t m, uint64_t n, uint64_t k,
    double alpha,
    double const* A, uint64_t lda,
    double const* B, uint64_t ldb,
    double beta,
    double* C, uint64_t ldc
) -> void {
    uint64_t block_row = blockIdx.y;
    uint64_t block_col = blockIdx.x;
    uint64_t row = threadIdx.y;
    uint64_t col = threadIdx.x;

    double* Ct = &C[block_row * ldc * BLOCK_SIZE + block_col * BLOCK_SIZE];

    double acc = 0.0;
    for (size_t t = 0; t < (k / BLOCK_SIZE); ++t) {
        __shared__ double At[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double Bt[BLOCK_SIZE][BLOCK_SIZE];

        At[row][col] = A[block_row * lda * BLOCK_SIZE + t * BLOCK_SIZE];
        Bt[row][col] = B[t * ldb * BLOCK_SIZE + block_col * BLOCK_SIZE];
        __syncthreads();

        for (size_t l = 0; l < BLOCK_SIZE; ++l) {
            acc += At[row][l] * Bt[l][col];
        }
        __syncthreads();
    }

    Ct[row * ldc + col] *= beta + alpha * acc;
}
