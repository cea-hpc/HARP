//! Command-Line Interface related code.
//!
//! This module handles the parsing of CLI arguments using the [`clap`][1] crate.
//! It defines the availables runtime options and subcommands.
//!
//! [1]: https://crates.io/crates/clap

use crate::consts;

use clap::{Parser, Subcommand};

use std::path::PathBuf;

/// Performance profiling of hardware-accelerated Rust.
///
/// Simple benchmarking tool that compares the performance of the hardware-accelerated Rust code
/// targeting NVIDIA GPUs (OpenCL and CUDA) against various CPU implementations of BLAS-like
/// operations, both in serial and parallel.
#[derive(Clone, Debug, Parser)]
pub struct CliArgs {
    /// Number of meta-repetitions for the benchmark.
    #[arg(
        short,
        long,
        value_name = "META_REPS",
        default_value_t = consts::META_REPETITIONS,
        value_parser = clap::value_parser!(u8).range(2..u8::MAX.into()),
    )]
    pub meta_repetitions: u8,

    /// Number of repetitions of the tight loop.
    #[arg(
        short,
        long,
        value_name = "TIGHT_REPS",
        default_value_t = consts::TIGHT_LOOP_REPETITIONS,
        value_parser = clap::value_parser!(u16).range(1..u16::MAX.into()),
    )]
    pub tight_loop_repetitions: u16,

    /// Kernel command to run.
    #[command(subcommand)]
    pub kernel: KernelCmd,

    /// Output file, defaults to `stdout` if unspecified.
    #[arg(short, long)]
    pub output_file: Option<PathBuf>,

    /// Seed for the random number generator (RNG).
    #[arg(short, long, value_name = "SEED")]
    pub seed: Option<u64>,
}

/// List of available kernels to profile.
#[derive(Debug, Clone, PartialEq, Subcommand)]
pub enum KernelCmd {
    /// Single-precision general vector addition (SAXPY): `alpha * x + y`
    Saxpy {
        /// Lengths of the vectors.
        #[arg(
            short,
            long,
            required = true,
            num_args = 1..,
        )]
        lengths: Vec<usize>,
    },
    /// Double-precision general vector addition (DAXPY): alpha * x + y
    Daxpy {
        /// Lengths of the vectors.
        #[arg(
            short,
            long,
            required = true,
            num_args = 1..,
        )]
        lengths: Vec<usize>,
    },
    /// Double-precision general matrix multiplication (SGEMM): alpha * A * B + beta * C
    Sgemm {
        /// Size of the matrices.
        #[arg(
            short,
            long,
            required = true,
            num_args = 1..,
        )]
        sizes: Vec<usize>,
    },
    /// Double-precision general matrix multiplication (DGEMM): alpha * A * B + beta * C
    Dgemm {
        /// Size of the matrices.
        #[arg(
            short,
            long,
            required = true,
            num_args = 1..,
        )]
        sizes: Vec<usize>,
    },
}
