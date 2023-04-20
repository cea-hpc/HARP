__constant size_t BLOCK_SIZE = 32;

__kernel void tiled_sgemm(
    ulong m, ulong n, ulong k,
    float alpha,
    __global float const* A, ulong lda,
    __global float const* B, ulong ldb,
    float beta,
    __global float* C, ulong ldc
) {
    size_t const row = get_local_id(0);
    size_t const col = get_local_id(1);
    size_t const global_row = BLOCK_SIZE * get_group_id(0) + row;
    size_t const global_col = BLOCK_SIZE * get_group_id(1) + col;

    __local float A_tile[BLOCK_SIZE][BLOCK_SIZE];
    __local float B_tile[BLOCK_SIZE][BLOCK_SIZE];

    float acc = 0.0;
    size_t const ntiles = k / BLOCK_SIZE;
    for (size_t t = 0; t < ntiles; ++t) {
        size_t const tiled_row = BLOCK_SIZE * t + row;
        size_t const tiled_col = BLOCK_SIZE * t + col;

        A_tile[col][row] = A[tiled_col * lda + global_row];
        B_tile[col][row] = B[global_col * ldb + tiled_row];
        barrier(CLK_LOCAL_MEM_FENCE);

        for (size_t l = 0; l < BLOCK_SIZE; ++l) {
            acc += A_tile[l][row] * B_tile[col][l];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final result in C
    C[global_col * ldc + global_row] *= beta + alpha * acc;
}
