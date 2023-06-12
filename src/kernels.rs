//! Kernel implementations.
//!
//! This modules contains the actual implementation of the host and device kernels.

pub mod device {
    //! Device kernel implementations.

    use cust::{function::Function, prelude::*, stream::Stream};
    use std::mem::size_of;

    /// Represents a device kernel.
    ///
    /// As there is no generic way of writing of function that will execute on an accelerator in
    /// Rust, there isn't a mechanism that allows to express kernels other than with static string
    /// slices (`&'static str`). Thus, we have to write the kernels in the syntax of the target
    /// "framework": either OpenCL C or NVIDIA CUDA C++, and directly import the kernel source code
    /// as raw text.
    pub struct DeviceKernel {
        kernel_name: &'static str,
        kernel_source: &'static str,
    }

    impl DeviceKernel {
        /// Creates a `DeviceKernel` from a kernel name (actual name of the kernel function in the
        /// source code) and a kernel source code (generally a file's contents).
        pub const fn new(kernel_name: &'static str, kernel_source: &'static str) -> Self {
            Self {
                kernel_name,
                kernel_source,
            }
        }

        /// Returns the function name of the given kernel.
        pub fn name(&self) -> &'static str {
            self.kernel_name
        }

        /// Returns the source code for the given kernel.
        pub fn source(&self) -> &'static str {
            self.kernel_source
        }
    }

    /// Name and source code of the OpenCL SAXPY.
    pub static CL_SAXPY: DeviceKernel =
        DeviceKernel::new("saxpy", include_str!("../kernels/opencl/saxpy.cl"));

    /// Name and source code of the OpenCL DAXPY.
    pub static CL_DAXPY: DeviceKernel =
        DeviceKernel::new("daxpy", include_str!("../kernels/opencl/daxpy.cl"));

    /// Name and source code of the naive OpenCL SGEMM.
    pub static CL_NAIVE_SGEMM: DeviceKernel = DeviceKernel::new(
        "naive_sgemm",
        include_str!("../kernels/opencl/naive_sgemm.cl"),
    );

    /// Name and source code of the naive OpenCL DGEMM.
    pub static CL_NAIVE_DGEMM: DeviceKernel = DeviceKernel::new(
        "naive_dgemm",
        include_str!("../kernels/opencl/naive_dgemm.cl"),
    );

    /// Name and source code of the tiled OpenCL SGEMM.
    pub static CL_TILED_SGEMM: DeviceKernel = DeviceKernel::new(
        "tiled_sgemm",
        include_str!("../kernels/opencl/tiled_sgemm.cl"),
    );

    /// Name and source code of the tiled OpenCL DGEMM.
    pub static CL_TILED_DGEMM: DeviceKernel = DeviceKernel::new(
        "tiled_dgemm",
        include_str!("../kernels/opencl/tiled_dgemm.cl"),
    );

    /// Name and source code of the NVIDIA CUDA SAXPY (as pre-compiled PTX).
    pub static CUDA_SAXPY: DeviceKernel =
        DeviceKernel::new("saxpy", include_str!("../kernels/cuda_cpp/saxpy.ptx"));

    /// Name and source code of the NVIDIA CUDA DAXPY (as pre-compiled PTX).
    pub static CUDA_DAXPY: DeviceKernel =
        DeviceKernel::new("daxpy", include_str!("../kernels/cuda_cpp/daxpy.ptx"));

    /// Name and source code of the naive NVIDIA CUDA SGEMM (as pre-compiled PTX).
    pub static CUDA_NAIVE_SGEMM: DeviceKernel = DeviceKernel::new(
        "naive_sgemm",
        include_str!("../kernels/cuda_cpp/naive_sgemm.ptx"),
    );

    /// Name and source code of the naive NVIDIA CUDA DGEMM (as pre-compiled PTX).
    pub static CUDA_NAIVE_DGEMM: DeviceKernel = DeviceKernel::new(
        "naive_dgemm",
        include_str!("../kernels/cuda_cpp/naive_dgemm.ptx"),
    );

    /// Name and source code of the tiled NVIDIA CUDA SGEMM (as pre-compiled PTX).
    pub static CUDA_TILED_SGEMM: DeviceKernel = DeviceKernel::new(
        "tiled_sgemm",
        include_str!("../kernels/cuda_cpp/tiled_sgemm.ptx"),
    );

    /// Name and source code of the tiled NVIDIA CUDA DGEMM (as pre-compiled PTX).
    pub static CUDA_TILED_DGEMM: DeviceKernel = DeviceKernel::new(
        "tiled_dgemm",
        include_str!("../kernels/cuda_cpp/tiled_dgemm.ptx"),
    );

    pub static RUST_CUDA_TILED_DGEMM: DeviceKernel =
        DeviceKernel::new("gemm", include_str!("../kernels/rust_cuda/gemm.ptx"));

    pub static CUDA_INTEGER_SUM_REDUCE: DeviceKernel =
        DeviceKernel::new("reduce", include_str!("../kernels/cuda_cpp/reduce.ptx"));

    pub static CUDA_INTEGER_SUM_EXCLUSIVE_SCAN: DeviceKernel =
        DeviceKernel::new("scan", include_str!("../kernels/cuda_cpp/scan.ptx"));

    pub fn scan(
        scan_kernel: &Function,
        block_sum_kernel: &Function,
        d_in: &DeviceBuffer<i32>,
        d_out: &mut DeviceBuffer<i32>,
        block_size: u32,
        smem_size: u32,
        stream: &Stream,
    ) {
        // Zero out the output data buffer
        d_out
            .set_zero()
            .expect("failed to zero out device output buffer");

        // Compute new grid size
        let len = d_in.len() as u32;
        let mut grid_size = len / block_size;
        if len % block_size != 0 {
            grid_size += 1;
        }

        // Allocate buffer for each block's partial sum
        let mut block_sums = DeviceBuffer::<i32>::zeroed(grid_size as usize)
            .expect("failed to create `block_sums` device buffer");

        // Launch first step of the kernel
        unsafe {
            launch!(
                scan_kernel<<<grid_size, block_size / 2, smem_size * size_of::<i32>() as u32, stream>>>(
                    d_in.as_device_ptr(),
                    d_in.len(),
                    d_out.as_device_ptr(),
                    block_sums.as_device_ptr(),
                    block_size,
                    smem_size
                )
            )
            .expect("failed to launch kernel `scan`");
        }
        stream
            .synchronize()
            .expect("failed to synchronize kernel `scan`");

        // Finish if there is only block left, else recurse until there is
        if grid_size <= block_size {
            let dummy =
                DeviceBuffer::<i32>::zeroed(1).expect("failed to create `dummy` device buffer");

            unsafe {
                launch!(
                    scan_kernel<<<1, block_size / 2, smem_size * size_of::<i32>() as u32, stream>>>(
                        block_sums.as_device_ptr(),
                        block_sums.len(), // this is `grid_size`
                        block_sums.as_device_ptr(),
                        dummy.as_device_ptr(),
                        block_size,
                        smem_size
                    )
                )
                .expect("failed to launch last `scan` step");
            }
            stream
                .synchronize()
                .expect("failed to synchronize kernel `scan` (end of recursion)");
        } else {
            let mut in_block_sums = DeviceBuffer::<i32>::zeroed(block_sums.len())
                .expect("failed to create `in_block_sums` device buffer");

            block_sums
                .copy_to(&mut in_block_sums)
                .expect("failed to copy `block_sums` into `in_block_sums`");

            scan(
                scan_kernel,
                block_sum_kernel,
                &in_block_sums,
                &mut block_sums,
                block_size,
                smem_size,
                stream,
            );
        }

        // Update the rest of the buffer's elements with the partial sum of each block
        unsafe {
            launch!(
            block_sum_kernel<<<grid_size, block_size / 2, 0, stream>>>(
                d_in.as_device_ptr(),
                d_in.len(),
                d_out.as_device_ptr(),
                block_sums.as_device_ptr(),
                block_sums.len()
            ))
            .expect("failed to launch kernel `block_sums`");
        }
        stream
            .synchronize()
            .expect("failed to synchronize kernel `block_sums`");
    }
}

pub mod host {
    //! Host kernel implementations.
    //!
    //! The parallel implementations rely on the [`rayon`][1] crate.
    //!
    //! [1]: https://crates.io/crates/rayon

    use rayon::prelude::*;

    use crate::HarpFloat;

    // Naive implementation of the AXPY kernel (unidiomatic Rust).
    pub fn axpy<T: HarpFloat>(alpha: T, x: &[T], y: &mut [T]) {
        for i in 0..y.len() {
            y[i] += alpha * x[i];
        }
    }

    // Idiomatic Rust implementation of the AXPY kernel (using iterators).
    pub fn iter_axpy<T: HarpFloat>(alpha: T, x: &[T], y: &mut [T]) {
        y.iter_mut()
            .zip(x.iter())
            .for_each(|(yi, xi)| *yi += alpha * *xi);
    }

    // Parallel implementation of the AXPY kernel (using `rayon`'s parallel iterators).
    pub fn par_iter_axpy<T: HarpFloat>(alpha: T, x: &[T], y: &mut [T]) {
        y.par_iter_mut()
            .zip(x.par_iter())
            .for_each(|(yi, xi)| *yi += alpha * *xi);
    }

    // Naive implementation of the GEMM kernel (unidiomatic Rust).
    #[allow(clippy::too_many_arguments, non_snake_case)]
    pub fn gemm<T: HarpFloat>(
        alpha: T,
        A: &[T],
        lda: usize,
        B: &[T],
        ldb: usize,
        beta: T,
        C: &mut [T],
        ldc: usize,
    ) {
        for i in 0..lda {
            for j in 0..ldb {
                let mut acc = T::default();
                for l in 0..ldc {
                    acc += A[i * lda + l] * B[l * ldb + j];
                }
                C[i * ldc + j] *= beta + alpha * acc;
            }
        }
    }

    // Idiomatic Rust implementation of the GEMM kernel (using iterators).
    #[allow(clippy::too_many_arguments, non_snake_case)]
    pub fn iter_gemm<T: HarpFloat>(
        alpha: T,
        A: &[T],
        lda: usize,
        B: &[T],
        ldb: usize,
        beta: T,
        C: &mut [T],
        ldc: usize,
    ) {
        C.chunks_exact_mut(ldc)
            .zip(A.chunks_exact(lda))
            .for_each(|(c_row, a_row)| {
                c_row
                    .iter_mut()
                    .zip(B.chunks_exact(ldb))
                    .for_each(|(c_ij, b_col)| {
                        *c_ij *= beta
                            + alpha
                                * a_row
                                    .iter()
                                    .zip(b_col)
                                    .fold(T::default(), |acc, (a_l, b_l)| acc + *a_l * *b_l);
                    })
            });
    }

    // Parallel implementation of the GEMM kernel (using `rayon`'s parallel iterators).
    #[allow(clippy::too_many_arguments, non_snake_case)]
    pub fn par_iter_gemm<T: HarpFloat>(
        alpha: T,
        A: &[T],
        lda: usize,
        B: &[T],
        ldb: usize,
        beta: T,
        C: &mut [T],
        ldc: usize,
    ) {
        C.par_chunks_exact_mut(ldc)
            .zip(A.par_chunks_exact(lda))
            .for_each(|(c_row, a_row)| {
                c_row
                    .iter_mut()
                    .zip(B.chunks_exact(ldb))
                    .for_each(|(c_ij, b_col)| {
                        *c_ij *= beta
                            + alpha
                                * a_row
                                    .iter()
                                    .zip(b_col)
                                    .fold(T::default(), |acc, (a_l, b_l)| acc + *a_l * *b_l);
                    })
            });
    }

    pub fn reduce(x: &[i32]) -> i32 {
        x.iter().sum()
    }

    pub fn par_reduce(x: &[i32]) -> i32 {
        x.par_iter().cloned().reduce(|| 0, |acc, e| acc + e)
    }

    pub fn scan(x: &[i32]) -> Vec<i32> {
        let mut skip = false;
        x.iter()
            .scan(0, |state, e| {
                if skip {
                    skip = true;
                    return Some(*state);
                }
                *state += e;
                Some(*state)
            })
            .collect()
    }
}
