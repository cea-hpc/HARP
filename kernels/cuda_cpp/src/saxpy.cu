#include <cstdint>

extern "C" __global__
auto saxpy(uint64_t n, float alpha, float const* x, float* y) -> void {
    uint64_t idx = threadIdx.x;
    if (idx < n) {
        y[idx] += alpha * x[idx];
    }
}
