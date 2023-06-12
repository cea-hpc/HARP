//! Performance report related structures and functions.
//!
//! This module defines all the necessary data structures and functions needed to generate
//! performance reports out of the recorded execution times of the benchmarked kernels.

use statistical::{mean, standard_deviation};

use std::{fmt, io::Write, mem::size_of};

/// Enum defining the target of kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TargetKind {
    Host,
    Device,
}

impl fmt::Display for TargetKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Host => write!(f, "host"),
            Self::Device => write!(f, "device"),
        }
    }
}

/// List of implemented kernels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KernelKind {
    Saxpy,
    Daxpy,
    Sgemm,
    Dgemm,
    Ireduce,
    Iscan,
}

impl fmt::Display for KernelKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Saxpy => write!(f, "saxpy"),
            Self::Daxpy => write!(f, "daxpy"),
            Self::Sgemm => write!(f, "sgemm"),
            Self::Dgemm => write!(f, "dgemm"),
            Self::Ireduce => write!(f, "reduce"),
            Self::Iscan => write!(f, "scan"),
        }
    }
}

/// Marker trait for kernel variants (i.e. implementations).
pub trait KernelVariant {}

/// Host-specific possible kernel implementations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HostKernelVariant {
    SeqNaive,
    SeqIter,
    ParIter,
}

impl KernelVariant for HostKernelVariant {}

impl fmt::Display for HostKernelVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SeqNaive => write!(f, "Sequential naive"),
            Self::SeqIter => write!(f, "Sequential w/ iterators"),
            Self::ParIter => write!(f, "Parallel w/ iterators"),
        }
    }
}

/// Device-specific possible kernel implementations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DeviceKernelVariant {
    ClNaive,
    ClTiled,
    CudaNaive,
    CudaTiled,
    RustCuda,
}

impl KernelVariant for DeviceKernelVariant {}

impl fmt::Display for DeviceKernelVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ClNaive => write!(f, "OpenCL naive"),
            Self::ClTiled => write!(f, "OpenCL tiled"),
            Self::CudaNaive => write!(f, "CUDA naive"),
            Self::CudaTiled => write!(f, "CUDA tiled"),
            Self::RustCuda => write!(f, "Rust-CUDA (tiled)"),
        }
    }
}

/// Performance information and statistics of a benchmark.
pub struct PerfReport<V> {
    /// Target platform: either `Host` or `Device`.
    target: TargetKind,
    /// Benchmarked kernel.
    kernel: KernelKind,
    /// Implementation variant of the kernel.
    variant: V,
    /// Number of elements per dimension.
    nb_elems_per_dim: usize,
    /// Size in bytes.
    nb_bytes: usize,
    /// Number of floating-point operations.
    nb_flops: usize,
    /// Minimum recorded runtime in milliseconds.
    min_time: f64,
    /// Median recorded runtime in milliseconds.
    median_time: f64,
    /// Maximum recorded runtime in milliseconds.
    max_time: f64,
    /// Average runtime in milliseconds.
    avg_time: f64,
    /// Runtime standard deviation.
    stddev_time: f64,
    /// Arithmetic intensity in FLOPs/byte.
    arithmetic_intensity: f64,
    /// Memory bandwidth in GiB/s.
    memory_bandwidth: f64,
    /// Computational performance in GFLOP/s.
    computational_performance: f64,
}

impl<V> PerfReport<V> {
    pub fn print_csv_header(output: &mut dyn Write) {
        writeln!(
            output,
            "target,kernel,variant,elems_per_dim,Bytes,FLOPs,min_runtime,median_runtime,max_runtime,avg_runtime,stddev,FLOPs/Byte,GiB/s,GFLOP/s"
        ).expect("Failed to write report's CSV header");
    }
}

impl<V> PerfReport<V>
where
    V: KernelVariant,
{
    /// Creates a new `PerfReport` given a target, a kernel, its variant, the number of elements
    /// per dimension and the recorded execution times.
    pub fn new(
        target: TargetKind,
        kernel: KernelKind,
        variant: V,
        nb_elems_per_dim: usize,
        durations: &mut [f64],
    ) -> Self {
        // Sort durations to avoid having to do two passes to get both min and max elements
        durations.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min_time = *durations.first().expect("Failed to get minimum duration") * 1e3;
        let median_time = *durations
            .get(durations.len() / 2)
            .expect("Failed to get median duration")
            * 1e3;
        let max_time = *durations.last().expect("Failed to get maximum duration") * 1e3;
        let avg_time = mean(durations);
        let stddev_time = standard_deviation(durations, Some(avg_time));

        let (nb_bytes, nb_flops) = match kernel {
            KernelKind::Saxpy => (
                2 * size_of::<f32>() * nb_elems_per_dim,
                2 * nb_elems_per_dim,
            ),
            KernelKind::Daxpy => (
                2 * size_of::<f64>() * nb_elems_per_dim,
                2 * nb_elems_per_dim,
            ),
            KernelKind::Sgemm => (
                3 * size_of::<f32>() * nb_elems_per_dim * nb_elems_per_dim,
                nb_elems_per_dim * nb_elems_per_dim * (2 * nb_elems_per_dim + 3),
            ),
            KernelKind::Dgemm => (
                3 * size_of::<f64>() * nb_elems_per_dim * nb_elems_per_dim,
                nb_elems_per_dim * nb_elems_per_dim * (2 * nb_elems_per_dim + 3),
            ),
            KernelKind::Ireduce => (size_of::<i32>() * nb_elems_per_dim, nb_elems_per_dim),
            KernelKind::Iscan => (size_of::<i32>() * nb_elems_per_dim, nb_elems_per_dim),
        };

        let memory_bandwidth = nb_bytes as f64 / 1024_f64.powi(3) / avg_time;
        let arithmetic_intensity = nb_flops as f64 / nb_bytes as f64;
        let computational_performance = nb_flops as f64 / (1024_f64.powi(3) * avg_time);

        let avg_time = avg_time * 1e3;

        Self {
            target,
            kernel,
            variant,
            nb_elems_per_dim,
            nb_bytes,
            nb_flops,
            min_time,
            median_time,
            max_time,
            avg_time,
            stddev_time,
            memory_bandwidth,
            arithmetic_intensity,
            computational_performance,
        }
    }
}

impl<V: fmt::Display> fmt::Display for PerfReport<V>
where
    V: KernelVariant,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{},{},{},{},{},{},{:18.15},{:18.15},{:18.15},{:18.15},{},{},{},{}",
            self.target,
            self.kernel,
            self.variant,
            self.nb_elems_per_dim,
            self.nb_bytes,
            self.nb_flops,
            self.min_time,
            self.median_time,
            self.max_time,
            self.avg_time,
            self.stddev_time,
            self.arithmetic_intensity,
            self.memory_bandwidth,
            self.computational_performance,
        )
    }
}
