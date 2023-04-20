//! Device kernel drivers.
//!
//! This module implements the driver functions responsible for profiling the chosen kernels on
//! available devices (GPUs, FPGAs, etc...).
//!
//! # Crates used for calling kernels on a device
//! - [`ocl`][1] for an idiomatic Rust implementation of OpenCL bindings;
//! - [`cust`][2] for an ecosystem of libraries and tools enabling the launch of CUDA kernels from
//!   Rust.
//!
//! [1]: https://crates.io/crates/ocl
//! [2]: https://crates.io/crates/cust

use crate::{
    consts::{BLOCK_SIZE_1D, BLOCK_SIZE_2D},
    kernels::device::DeviceKernel,
    perf_report::*,
    utils::{is_of_type, HarpFloat},
};

use cust::{
    function::{BlockSize, GridSize},
    memory::DeviceCopy,
    prelude::*,
};
use ocl::{OclPrm, ProQue};

use std::{error::Error, time::Instant};

/// OpenCL device driver for the generic AXPY kernel.
pub fn ocl_axpy<T: HarpFloat + OclPrm>(
    kernel_info: &DeviceKernel,
    alpha: T,
    h_x: &[T],
    h_y: &[T],
    meta_reps: u8,
    tight_reps: u16,
) -> ocl::Result<PerfReport<DeviceKernelVariant>> {
    assert_eq!(h_x.len(), h_y.len());
    let len = h_y.len();

    // Create OpenCL program-queue object
    let pro_que = ProQue::builder()
        .src(kernel_info.source())
        .dims(h_y.len())
        .build()?;

    // Create device vectors
    let d_x = pro_que.buffer_builder().copy_host_slice(h_x).build()?;
    let d_y = pro_que.buffer_builder().copy_host_slice(h_y).build()?;

    // Declare OpenCL kernel object
    let mut kernel = pro_que
        .kernel_builder(kernel_info.name())
        .arg(alpha)
        .arg(&d_x)
        .arg(&d_y)
        .build()?;

    if len >= BLOCK_SIZE_1D && len % BLOCK_SIZE_1D == 0 {
        kernel.set_default_local_work_size(ocl::SpatialDims::One(BLOCK_SIZE_1D));
    }

    // Measure execution time of device kernel
    let mut durations = Vec::with_capacity(meta_reps.into());
    for _ in 0..durations.capacity() {
        let dur = Instant::now();
        for _ in 0..tight_reps {
            unsafe {
                kernel.enq()?;
            }
        }
        pro_que.queue().finish()?;
        durations.push((dur.elapsed() / tight_reps.into()).as_secs_f64());
    }

    let kind = match is_of_type::<f32>(&alpha) {
        true => KernelKind::Saxpy,
        false => KernelKind::Daxpy,
    };

    Ok(PerfReport::new(
        TargetKind::Device,
        kind,
        DeviceKernelVariant::ClNaive,
        h_y.len(),
        &mut durations,
    ))
}

/// NVIDIA CUDA device driver for the generic AXPY kernel.
pub fn cuda_axpy<T: HarpFloat + DeviceCopy>(
    kernel_info: &DeviceKernel,
    alpha: T,
    h_x: &[T],
    h_y: &[T],
    meta_reps: u8,
    tight_reps: u16,
) -> Result<PerfReport<DeviceKernelVariant>, Box<dyn Error>> {
    assert_eq!(h_x.len(), h_y.len());
    let len = h_y.len();

    // Initialize CUDA context
    let _ctx = cust::quick_init()?;

    // Create CUDA module from compiled PTX
    let module = Module::from_ptx(kernel_info.source(), &[])?;

    // Create CUDA stream
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Create device-side vectors
    let d_x = DeviceBuffer::from_slice(h_x)?;
    let d_y = DeviceBuffer::from_slice(h_y)?;

    // Get kernel from module
    let kernel = module.get_function(kernel_info.name())?;

    let block_size = BlockSize::x(BLOCK_SIZE_1D as u32);
    let grid_size = GridSize::x(len as u32 / block_size.x);

    // Measure execution time of kernel
    let mut durations = Vec::with_capacity(meta_reps.into());
    for _ in 0..durations.capacity() {
        let dur = Instant::now();
        for _ in 0..tight_reps {
            unsafe {
                launch!(
                    kernel<<<grid_size, block_size, 0, stream>>>(
                        h_y.len(),
                        alpha,
                        d_x.as_device_ptr(),
                        d_y.as_device_ptr(),
                    )
                )?;
            }
        }
        stream.synchronize()?;
        durations.push((dur.elapsed() / tight_reps as u32).as_secs_f64());
    }

    let kind = match is_of_type::<f32>(&alpha) {
        true => KernelKind::Saxpy,
        false => KernelKind::Daxpy,
    };

    Ok(PerfReport::new(
        TargetKind::Device,
        kind,
        DeviceKernelVariant::CudaNaive,
        h_y.len(),
        &mut durations,
    ))
}

/// OpenCL device driver for the generic GEMM kernel.
// NOTE: We use upper-case characters to designate matrices.
#[allow(non_snake_case, clippy::too_many_arguments)]
pub fn ocl_gemm<T: HarpFloat + OclPrm>(
    kernel_info: &DeviceKernel,
    size: usize,
    alpha: T,
    beta: T,
    h_A: &[T],
    h_B: &[T],
    h_C: &[T],
    meta_reps: u8,
    tight_reps: u16,
    variant: DeviceKernelVariant,
) -> ocl::Result<PerfReport<DeviceKernelVariant>> {
    // Create OpenCL program-queue object
    let pro_que = ProQue::builder()
        .src(kernel_info.source())
        .dims(size * size)
        .build()?;

    // Create device vectors
    let d_A = pro_que.buffer_builder().copy_host_slice(h_A).build()?;
    let d_B = pro_que.buffer_builder().copy_host_slice(h_B).build()?;
    let d_C = pro_que.buffer_builder().copy_host_slice(h_C).build()?;

    // Declare OpenCL kernel object
    let mut kernel = pro_que
        .kernel_builder(kernel_info.name())
        .global_work_size([size, size])
        .arg(size as u64)
        .arg(size as u64)
        .arg(size as u64)
        .arg(alpha)
        .arg(&d_A)
        .arg(size as u64)
        .arg(&d_B)
        .arg(size as u64)
        .arg(beta)
        .arg(&d_C)
        .arg(size as u64)
        .build()?;

    if size >= BLOCK_SIZE_2D && size % BLOCK_SIZE_2D == 0 {
        kernel.set_default_local_work_size(ocl::SpatialDims::Two(BLOCK_SIZE_2D, BLOCK_SIZE_2D));
    }

    // Measure execution time of device kernel
    let mut durations = Vec::with_capacity(meta_reps.into());
    for _ in 0..durations.capacity() {
        let dur = Instant::now();
        for _ in 0..tight_reps {
            unsafe {
                kernel.enq()?;
            }
        }
        pro_que.queue().finish()?;
        durations.push((dur.elapsed() / tight_reps.into()).as_secs_f64());
    }

    let kind = match is_of_type::<f32>(&alpha) {
        true => KernelKind::Sgemm,
        false => KernelKind::Dgemm,
    };
    Ok(PerfReport::new(
        TargetKind::Device,
        kind,
        variant,
        size,
        &mut durations,
    ))
}

/// NVIDIA CUDA device driver for the generic GEMM kernel.
// NOTE: We use upper-case characters to designate matrices.
#[allow(non_snake_case, clippy::too_many_arguments)]
pub fn cuda_gemm<T: HarpFloat + DeviceCopy>(
    kernel_info: &DeviceKernel,
    size: usize,
    alpha: T,
    beta: T,
    h_A: &[T],
    h_B: &[T],
    h_C: &[T],
    meta_reps: u8,
    tight_reps: u16,
    variant: DeviceKernelVariant,
) -> Result<PerfReport<DeviceKernelVariant>, Box<dyn Error>> {
    let tight_reps = if size > 512 { 1 } else { tight_reps };

    // Initialize CUDA context
    let _ctx = cust::quick_init()?;

    // Create CUDA module from compiled PTX
    let module = Module::from_ptx(kernel_info.source(), &[])?;

    // Create CUDA stream
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Create device-side vectors
    let d_A = DeviceBuffer::from_slice(h_A)?;
    let d_B = DeviceBuffer::from_slice(h_B)?;
    let d_C = DeviceBuffer::from_slice(h_C)?;

    // Get kernel from module
    let kernel = module.get_function(kernel_info.name())?;

    let block_size = BlockSize::xy(BLOCK_SIZE_2D as u32, BLOCK_SIZE_2D as u32);
    let grid_size = GridSize::xy(size as u32 / block_size.x, size as u32 / block_size.y);

    // Measure execution time of kernel
    let mut durations = Vec::with_capacity(meta_reps.into());
    for _ in 0..durations.capacity() {
        let dur = Instant::now();
        for _ in 0..tight_reps {
            unsafe {
                launch!(
                    kernel<<<grid_size, block_size, 0, stream>>>(
                        size, size, size,
                        alpha,
                        d_A.as_device_ptr(),
                        size,
                        d_B.as_device_ptr(),
                        size,
                        beta,
                        d_C.as_device_ptr(),
                        size,
                    )
                )?;
            }
        }
        stream.synchronize()?;
        durations.push((dur.elapsed() / tight_reps as u32).as_secs_f64());
    }

    let kind = match is_of_type::<f32>(&alpha) {
        true => KernelKind::Sgemm,
        false => KernelKind::Dgemm,
    };
    Ok(PerfReport::new(
        TargetKind::Device,
        kind,
        variant,
        size,
        &mut durations,
    ))
}
