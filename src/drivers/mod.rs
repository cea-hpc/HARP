//! Kernel drivers.
//!
//! This module provides the API for appropriately initializing and profiling the kernels, as well
//! as post-processing the results. It acts as a layer abstracting over the kernel's target: either
//! the host (CPU) or the device (GPU, FPGA, etc...).
//!
//! # High-level approach for kernel profiling
//! ## 1. Data initialization
//! This step is generally trivial for host kernels as data can be easily shared between
//! implementations. However, when targeting a device and depending on the chosen implementation
//! (i.e. OpenCL or NVIDIA CUDA), different steps may be required in order to
//! correctly copy/initialize the data on the device.
//!
//! ## 2. Performance evaluation
//! This step consists in measuring the execution time of the chosen kernel. In order to get
//! an accurate evaluation of a kernel's performance, we repeat this process in a "meta
//! repetitions loop". This allows us to get enough measurements and assess the precision of the
//! results (see [`crate::consts`] for the default amount of meta repetitions and [`crate::cli`]
//! for overriding the default from the command-line).
//!
//! In addition, when measuring the performance of kernels that operate on very few data (e.g.
//! an AXPY kernel with vectors small enough to fit in the L1 cache), we may want to increase the
//! execution time by repeatedly calling the kernel in a tight loop and averaging the elapsed
//! time over the number of iterations of this loop (see [`crate::cli`] for overriding the number
//! of repetitions of the tight loop from the command-line).
//!
//! ## 3. Post-processing
//! This step consists in extracting performance metrics from the recorded execution times of the
//! kernel and information about the manipulated data, such as computational performance (in
//! GFLOP/s), memory bandwidth (in GiB/s) or arithmetic intensity (in FLOPs/Byte).
//!
//! The resulting metrics are then outputted to `stdout`, or a file if specified (see
//! [`crate::cli`] for specifying an output file from the command-line).

mod device;
mod host;

use crate::{cli::*, kernels::device::*, perf_report::*, utils::*};

use std::{
    fs::OpenOptions,
    io::{stdout, Write},
};

/// Driver function responsible for initializing the data for the AXPY kernel and forwarding it to
/// the dedicated target drivers. It gathers the results for each benchmarked kernel variant and
/// generates an output summary.
pub fn axpy<T: HarpFloat + ocl::OclPrm + cust::memory::DeviceCopy>(args: CliArgs) {
    let lengths = match args.kernel {
        KernelCmd::Saxpy { lengths } | KernelCmd::Daxpy { lengths } => lengths,
        _ => unreachable!(),
    };

    // Initialize vector storing the performance reports for each benchmark
    // TODO: Consider using a single vector to hold all performance reports.
    let mut host_perf_reports = Vec::new();
    let mut device_perf_reports = Vec::new();

    // Initialize the same `alpha` for all vector lengths
    let alpha = T::rand_scalar(args.seed);

    // Very messy "trick" to runtime-check the type of `T` (either `f32`, or `f64`)
    let (cl_kern, cuda_kern) = match is_of_type::<f32>(&alpha) {
        true => (&CL_SAXPY, &CUDA_SAXPY),
        false => (&CL_DAXPY, &CUDA_DAXPY),
    };

    for len in lengths {
        eprint!("Vector length: {len}\r");
        // Initialize new `x` and `y` vectors for each length
        let x = T::rand_vector(len, args.seed);
        let y = T::rand_vector(len, args.seed);

        // NOTE: The data (scalar and vectors) is immutably shared for all kernel variants. This is
        // to make sure the values cannot impact the performance of the profiled kernel
        // implementation.

        // TODO: Consider using a crate such as `strum` to iterate over the available kernel
        // variants in order to make the code more compact and simplify refactoring when adding new
        // kernel implementations.
        // OR make it so that the user can specify a set of variants to benchmark which would be
        // iterated through here.
        host_perf_reports.push(host::axpy::<T>(
            alpha,
            &x,
            &mut y.clone(),
            args.meta_repetitions,
            args.tight_loop_repetitions,
            HostKernelVariant::SeqNaive,
        ));

        host_perf_reports.push(host::axpy::<T>(
            alpha,
            &x,
            &mut y.clone(),
            args.meta_repetitions,
            args.tight_loop_repetitions,
            HostKernelVariant::SeqIter,
        ));

        host_perf_reports.push(host::axpy::<T>(
            alpha,
            &x,
            &mut y.clone(),
            args.meta_repetitions,
            args.tight_loop_repetitions,
            HostKernelVariant::ParIter,
        ));

        device_perf_reports.push(
            device::ocl_axpy(
                cl_kern,
                alpha,
                &x,
                &y.clone(),
                args.meta_repetitions,
                args.tight_loop_repetitions,
            )
            .unwrap(),
        );

        device_perf_reports.push(
            device::cuda_axpy(
                cuda_kern,
                alpha,
                &x,
                &y.clone(),
                args.meta_repetitions,
                args.tight_loop_repetitions,
            )
            .unwrap(),
        );
    }

    let mut output: Box<dyn Write> = match args.output_file {
        Some(ref name) => Box::new(
            OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(name)
                .unwrap(),
        ),
        None => Box::new(stdout()),
    };

    PerfReport::<()>::print_csv_header(&mut output);
    for report in host_perf_reports {
        writeln!(output, "{report}").expect("Failed to write report");
    }
    for report in device_perf_reports {
        writeln!(output, "{report}").expect("Failed to write report");
    }
}

/// Driver function responsible for initializing the data for the GEMM kernel and forwarding it to
/// the dedicated target drivers. It gathers the results for each benchmarked kernel variant and
/// generates an output summary.
// NOTE: We use upper-case characters to designate matrices.
#[allow(non_snake_case)]
pub fn gemm<T: HarpFloat + ocl::OclPrm + cust::memory::DeviceCopy>(args: CliArgs) {
    let sizes = match args.kernel {
        KernelCmd::Sgemm { sizes } | KernelCmd::Dgemm { sizes } => sizes,
        _ => unreachable!(),
    };

    // Initialize vector storing the performance reports for each benchmark
    // TODO: Consider using a single vector to hold all performance reports.
    let mut host_perf_reports = Vec::new();
    let mut device_perf_reports = Vec::new();

    // Initialize the same `alpha` and `beta` for all matrix sizes
    let alpha = T::rand_scalar(args.seed);
    let beta = T::rand_scalar(None);

    // Very messy "trick" to runtime-check the type of `T` (either `f32`, or `f64`)
    let (naive_cl_kern, naive_cuda_kern, tiled_cl_kern, tiled_cuda_kern) =
        match is_of_type::<f32>(&alpha) {
            true => (
                &CL_NAIVE_SGEMM,
                &CUDA_NAIVE_SGEMM,
                &CL_TILED_SGEMM,
                &CUDA_TILED_SGEMM,
            ),
            false => (
                &CL_NAIVE_DGEMM,
                &CUDA_NAIVE_DGEMM,
                &CL_TILED_DGEMM,
                &CUDA_TILED_DGEMM,
            ),
        };

    for size in sizes {
        eprint!("Matrix size: {size}\r");
        // Initialize new `A`, `B` and `C` matrices for each size
        let A = T::rand_vector(size * size, args.seed);
        let B = T::rand_vector(size * size, args.seed);
        let C = T::rand_vector(size * size, args.seed);

        // NOTE: The data (scalar and vectors) is immutably shared for all kernel variants. This is
        // to make sure the values cannot impact the performance of the profiled kernel
        // implementation.

        // TODO: Consider using a crate such as `strum` to iterate over the available kernel
        // variants in order to make the code more compact and simplify refactoring when adding new
        // kernel implementations.
        // OR make it so that the user can specify a set of variants to benchmark which would be
        // iterated through here.

        // NOTE: For now, we do not run sequential GEMMs above 256x256 to avoid very long runtimes.
        if size <= 256 {
            host_perf_reports.push(host::gemm::<T>(
                size,
                alpha,
                beta,
                &A,
                &B,
                &mut C.clone(),
                args.meta_repetitions,
                args.tight_loop_repetitions,
                HostKernelVariant::SeqNaive,
            ));

            host_perf_reports.push(host::gemm::<T>(
                size,
                alpha,
                beta,
                &A,
                &B,
                &mut C.clone(),
                args.meta_repetitions,
                args.tight_loop_repetitions,
                HostKernelVariant::SeqIter,
            ));
        }

        // NOTE: For now, we do not run host parallel and naive OpenCL GEMMs above 512x512 to avoid
        // very long runtimes.
        if size <= 512 {
            host_perf_reports.push(host::gemm::<T>(
                size,
                alpha,
                beta,
                &A,
                &B,
                &mut C.clone(),
                args.meta_repetitions,
                args.tight_loop_repetitions,
                HostKernelVariant::ParIter,
            ));

            device_perf_reports.push(
                device::ocl_gemm(
                    naive_cl_kern,
                    size,
                    alpha,
                    beta,
                    &A,
                    &B,
                    &C.clone(),
                    args.meta_repetitions,
                    args.tight_loop_repetitions,
                    DeviceKernelVariant::ClNaive,
                )
                .unwrap(),
            );
        }

        device_perf_reports.push(
            device::ocl_gemm(
                tiled_cl_kern,
                size,
                alpha,
                beta,
                &A,
                &B,
                &C.clone(),
                args.meta_repetitions,
                args.tight_loop_repetitions,
                DeviceKernelVariant::ClTiled,
            )
            .unwrap(),
        );

        device_perf_reports.push(
            device::cuda_gemm(
                naive_cuda_kern,
                size,
                alpha,
                beta,
                &A,
                &B,
                &C.clone(),
                args.meta_repetitions,
                args.tight_loop_repetitions,
                DeviceKernelVariant::CudaNaive,
            )
            .unwrap(),
        );

        device_perf_reports.push(
            device::cuda_gemm(
                tiled_cuda_kern,
                size,
                alpha,
                beta,
                &A,
                &B,
                &C.clone(),
                args.meta_repetitions,
                args.tight_loop_repetitions,
                DeviceKernelVariant::CudaTiled,
            )
            .unwrap(),
        );
    }

    let mut output: Box<dyn Write> = match args.output_file {
        Some(ref name) => Box::new(
            OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(name)
                .unwrap(),
        ),
        None => Box::new(stdout()),
    };

    PerfReport::<()>::print_csv_header(&mut output);
    for report in host_perf_reports {
        writeln!(output, "{report}").expect("Failed to write report");
    }
    for report in device_perf_reports {
        writeln!(output, "{report}").expect("Failed to write report");
    }
}
