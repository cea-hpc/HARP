kernel void naive_dgemm(
    ulong m, ulong n, ulong k,
    double alpha,
    global double const* A, ulong lda,
    global double const* B, ulong ldb,
    double beta,
    global double* C, ulong ldc
) {
    size_t const i = get_global_id(0);
    size_t const j = get_global_id(1);

    double acc = 0.0;
    for (size_t l = 0; l < k; ++l) {
        acc += A[i * lda + l] * B[l * ldb + j];
    }
    C[i * ldc + j] *= beta + alpha * acc;
}
