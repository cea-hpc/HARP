//! HARP - Hardware-Accelerated Rust Profiling
//!
//! # About
//! HARP is a simple profiler for evaluating the performance of hardware-accelerated Rust code. It
//! aims at gauging the capabilities of Rust as a first-class language for GPGPU computing,
//! especially in the field of High Performance Computing (HPC).  
//!
//! Currently, HARP can profile the following GPU-accelerated kernels (targeting OpenCL C and NVIDIA
//! CUDA C++ implementations):
//! - AXPY (general vector-vector addition)
//! - GEMM (general dense matrix-matrix multiplication)
//!
//! Profiling can be done on both single-precision and double-precision floating-point formats (see
//! [IEEE 754][1]).
//!
//! # Quickstart
//! ## Pre-requisites
//! Make sure an [OpenCL 2.0+][2] library and the [NVIDIA CUDA Toolkit][3] are installed on your
//! system before beginning.
//!
//! ## Build
//! As any Rust-based project, HARP is built and run with `cargo`:
//! ```sh
//! cargo build --release
//! ```
//!
//! ## Help
//! To see the help usage:
//! ```sh
//! cargo run -- help
//!
//! Performance profiling of hardware-accelerated Rust.
//!
//! Usage: harp [OPTIONS] <COMMAND>
//!
//! Commands:
//!   saxpy   Single-precision general vector addition (SAXPY): `alpha * x + y`
//!   daxpy   Double-precision general vector addition (DAXPY): alpha * x + y
//!   sgemm   Double-precision general matrix multiplication (SGEMM): alpha * A * B + beta * C
//!   dgemm   Double-precision general matrix multiplication (DGEMM): alpha * A * B + beta * C
//!   help    Print this message or the help of the given subcommand(s)
//!
//! Options:
//!   -m, --meta-repetitions <META_REPS>
//!           Number of meta-repetitions for the benchmark
//!           
//!           [default: 31]
//!
//!   -t, --tight-loop-repetitions <TIGHT_REPS>
//!           Number of repetitions of the tight loop
//!           
//!           [default: 1]
//!
//!   -o, --output-file <OUTPUT_FILE>
//!           Output file, defaults to `stdout` if unspecified
//!
//!   -s, --seed <SEED>
//!           Seed for the random number generator (RNG)
//!
//!   -h, --help
//!           Print help (see a summary with '-h')
//! ```
//!
//! ## Example run
//! To execute HARP and profile a DGEMM on multiple matrix sizes, execute the following command:
//! ```sh
//! cargo run --release -- dgemm --sizes 512 1024 1532 2048 3072 4096
//! ```
//!
//! ## Documentation
//! The crate's documentation is available using `cargo`:
//! ```sh
//! cargo doc --open
//! ```
//!
//! [1]: https://en.wikipedia.org/wiki/IEEE_754
//! [2]: https://www.khronos.org/opencl/
//! [3]: https://developer.nvidia.com/cuda-downloads

pub mod cli;
pub mod consts;
pub mod drivers;
pub mod kernels;
pub mod perf_report;
pub mod utils;

use crate::cli::{CliArgs, KernelCmd};
use crate::utils::*;

use clap::Parser;

fn main() {
    let args = CliArgs::parse();

    match args.kernel {
        KernelCmd::Saxpy { lengths: _ } => drivers::axpy::<f32>(args),
        KernelCmd::Daxpy { lengths: _ } => drivers::axpy::<f64>(args),
        KernelCmd::Sgemm { sizes: _ } => drivers::gemm::<f32>(args),
        KernelCmd::Dgemm { sizes: _ } => drivers::gemm::<f64>(args),
        KernelCmd::Ireduce { lengths: _ } => drivers::reduce(args),
        KernelCmd::Iscan { lengths: _ } => drivers::scan(args),
    }
}
