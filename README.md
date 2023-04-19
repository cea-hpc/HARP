# HARP - Hardware-Accelerated Rust Profiling

## About
HARP is a simple profiler for evaluating the performance of hardware-accelerated Rust code. It aims at gauging the capabilities of Rust as a first-class language for GPGPU computing, especially in the field of High Performance Computing (HPC).  

Currently, HARP can profile the following GPU-accelerated kernels (targeting OpenCL C and NVIDIA CUDA C++ implementations):
- AXPY (general vector-vector addition)
- GEMM (general dense matrix-matrix multiplication)
Profiling can be done on both single-precision and double-precision floating-point formats (see [IEEE 754](https://en.wikipedia.org/wiki/IEEE_754)).

## Quickstart
### Pre-requisites
- [Rust](https://rustup.rs/) 1.68.0+
- [OpenCL 2.0+](https://www.khronos.org/opencl/)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 11.2+ (12.0 recommended) and appropriate drivers
  - `libnvvm` library
  - [LLVM 7.x](https://github.com/llvm/llvm-project/releases/tag/llvmorg-7.1.0) (7.0 to 7.4)
- [Python](https://www.python.org/downloads/) 3.7+ (only needed for plot generation)

### Build
First, clone this repository locally:
```sh
git clone https://github.com/cea-hpc/HARP
cd HARP
```

As any Rust-based project, HARP is built with `cargo`:
```sh
cargo build --release
```

### Run
See HARP's documentation for the full list of supported flags or use the `--help` option.  

To execute HARP and profile a DGEMM on multiple matrix sizes, execute the following example command:
```sh
cargo run --release -- dgemm --sizes 512 1024 1532 2048 3072 4096
# Or with shortand aliases
cargo r -r -- dgemm -s 512 1024 1532 2048 3072 4096
```

### Documentation
The crate's documentation is available using `cargo`:
```sh
cargo doc --open
```

## Contributing
Contributions are welcome and accepted as pull requests on [GitHub](https://github.com/cea-hpc/HARP).

You may also ask questions or file bug reports on the issue tracker.

## License
Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/cea-hpc/harp/blob/master/LICENSE-APACHE) or [https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0));
- MIT License ([LICENSE-MIT](https://github.com/cea-hpc/harp/blob/master/LICENSE-MIT) or [http://opensource.org/licenses/MIT](http://opensource.org/licenses/MIT))
at your option.  

The [SPDX](https://spdx.dev/) license identifier for this project is MIT OR Apache-2.0.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
