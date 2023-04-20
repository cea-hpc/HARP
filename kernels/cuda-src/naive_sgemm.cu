#include <cstdint>

extern "C" __global__
auto naive_sgemm(
    uint64_t m, uint64_t n, uint64_t k,
    float alpha,
    float const* A, uint64_t lda,
    float const* B, uint64_t ldb,
    float beta,
    float* C, uint64_t ldc
) -> void {
    uint64_t i = threadIdx.y;
    uint64_t j = threadIdx.x;

    float acc = 0.0;
    for (size_t l = 0; l < k; ++l) {
        acc += A[i * lda + l] * B[l * ldb + j];
    }
    C[i* ldc + j] *= beta + alpha * acc;
}
