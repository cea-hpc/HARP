kernel void naive_sgemm(
    ulong m, ulong n, ulong k,
    float alpha,
    global float const* A, ulong lda,
    global float const* B, ulong ldb,
    float beta,
    global float* C, ulong ldc
) {
    size_t const i = get_global_id(0);
    size_t const j = get_global_id(1);

    float acc = 0.0;
    for (size_t l = 0; l < k; ++l) {
        acc += A[i * lda + l] * B[l * ldb + j];
    }
    C[i * ldc + j] *= beta + alpha * acc;
}
