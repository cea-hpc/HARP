//! Host kernel drivers.
//!
//! This module implements the driver functions responsible for profiling the chosen kernels on the
//! host (CPU).

use crate::{kernels::host, perf_report::*, utils::*};

use std::time::Instant;

/// Host driver for the generic AXPY kernel.
///
/// This function selects the correct host kernel given `variant` and profiles it.
pub fn axpy<T: HarpFloat>(
    alpha: T,
    x: &[T],
    y: &mut [T],
    meta_reps: u8,
    tight_reps: u16,
    variant: HostKernelVariant,
) -> PerfReport<HostKernelVariant> {
    // Match on given kernel variant
    let kernel = match variant {
        HostKernelVariant::SeqNaive => host::axpy::<T>,
        HostKernelVariant::SeqIter => host::iter_axpy::<T>,
        HostKernelVariant::ParIter => host::par_iter_axpy::<T>,
    };

    // Measure execution time of kernel
    let mut durations = Vec::with_capacity(meta_reps.into());
    for _ in 0..durations.capacity() {
        let dur = Instant::now();
        for _ in 0..tight_reps {
            kernel(alpha, x, y);
        }
        durations.push((dur.elapsed() / tight_reps as u32).as_secs_f64());
    }

    let kind = match is_of_type::<f32>(&alpha) {
        true => KernelKind::Saxpy,
        false => KernelKind::Daxpy,
    };
    PerfReport::new(TargetKind::Host, kind, variant, y.len(), &mut durations)
}

/// Host driver for the generic GEMM kernel.
///
/// This function selects the correct host kernel given `variant` and profiles it.
// NOTE: We use upper-case characters to designate matrices.
#[allow(non_snake_case, clippy::too_many_arguments)]
pub fn gemm<T: HarpFloat>(
    size: usize,
    alpha: T,
    beta: T,
    A: &[T],
    B: &[T],
    C: &mut [T],
    meta_reps: u8,
    tight_reps: u16,
    variant: HostKernelVariant,
) -> PerfReport<HostKernelVariant> {
    // Match on given host kernel variant
    let kernel = match variant {
        HostKernelVariant::SeqNaive => host::gemm,
        HostKernelVariant::SeqIter => host::iter_gemm,
        HostKernelVariant::ParIter => host::par_iter_gemm,
    };

    // Measure execution time of host kernel
    let mut durations = Vec::with_capacity(meta_reps.into());
    for _ in 0..durations.capacity() {
        let dur = Instant::now();
        for _ in 0..tight_reps {
            kernel(alpha, A, size, B, size, beta, C, size);
        }
        durations.push((dur.elapsed() / tight_reps as u32).as_secs_f64());
    }

    let kind = match is_of_type::<f32>(&alpha) {
        true => KernelKind::Sgemm,
        false => KernelKind::Dgemm,
    };
    PerfReport::new(TargetKind::Host, kind, variant, size, &mut durations)
}

/// Host driver for the 32-bit integer sum reduction kernel.
///
/// This function selects the correct host kernel given `variant` and profiles it.
pub fn reduce(
    x: &[i32],
    meta_reps: u8,
    tight_reps: u16,
    variant: HostKernelVariant,
) -> PerfReport<HostKernelVariant> {
    // Match on given kernel variant
    let kernel = match variant {
        HostKernelVariant::SeqIter => host::reduce,
        HostKernelVariant::ParIter => host::par_reduce,
        _ => unreachable!(),
    };

    // Measure execution time of kernel
    let mut durations = Vec::with_capacity(meta_reps.into());
    for _ in 0..durations.capacity() {
        let dur = Instant::now();
        for _ in 0..tight_reps {
            let _ = std::hint::black_box(kernel(x));
        }
        durations.push((dur.elapsed() / tight_reps as u32).as_secs_f64());
    }

    PerfReport::new(
        TargetKind::Host,
        KernelKind::Ireduce,
        variant,
        x.len(),
        &mut durations,
    )
}

/// Host driver for the 32-bit integer sum exclusive scan kernel.
pub fn scan(x: &[i32], meta_reps: u8, tight_reps: u16) -> PerfReport<HostKernelVariant> {
    let kernel = host::scan;

    // Measure execution time of kernel
    let mut durations = Vec::with_capacity(meta_reps.into());
    for _ in 0..durations.capacity() {
        let dur = Instant::now();
        for _ in 0..tight_reps {
            let _ = kernel(x);
        }
        durations.push((dur.elapsed() / tight_reps as u32).as_secs_f64());
    }

    PerfReport::new(
        TargetKind::Host,
        KernelKind::Ireduce,
        HostKernelVariant::SeqIter,
        x.len(),
        &mut durations,
    )
}
