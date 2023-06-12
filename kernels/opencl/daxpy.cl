__kernel void daxpy(
    double alpha,
    __global double const* x,
    __global double* y
) {
    uint const idx = get_global_id(0);
    y[idx] += alpha * x[idx];    
}
