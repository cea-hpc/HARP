#include <cstdint>

extern "C" __global__
auto daxpy(uint64_t n, double alpha, double const* x, double* y) -> void {
    uint64_t idx = threadIdx.x;
    if (idx < n) {
        y[idx] += alpha * x[idx];
    }
}
