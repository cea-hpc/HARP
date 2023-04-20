__kernel void saxpy(
    float alpha,
    __global float const* x,
    __global float* y
) {
    uint const idx = get_global_id(0);
    y[idx] += alpha * x[idx];    
}
