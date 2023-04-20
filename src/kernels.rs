//! Kernel implementations.
//!
//! This modules contains the actual implementation of the host and device kernels.

pub mod device {
    //! Device kernel implementations.

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
        DeviceKernel::new("saxpy", include_str!("../kernels/saxpy.cl"));

    /// Name and source code of the OpenCL DAXPY.
    pub static CL_DAXPY: DeviceKernel =
        DeviceKernel::new("daxpy", include_str!("../kernels/daxpy.cl"));

    /// Name and source code of the naive OpenCL SGEMM.
    pub static CL_NAIVE_SGEMM: DeviceKernel =
        DeviceKernel::new("naive_sgemm", include_str!("../kernels/naive_sgemm.cl"));

    /// Name and source code of the naive OpenCL DGEMM.
    pub static CL_NAIVE_DGEMM: DeviceKernel =
        DeviceKernel::new("naive_dgemm", include_str!("../kernels/naive_dgemm.cl"));

    /// Name and source code of the tiled OpenCL SGEMM.
    pub static CL_TILED_SGEMM: DeviceKernel =
        DeviceKernel::new("tiled_sgemm", include_str!("../kernels/tiled_sgemm.cl"));

    /// Name and source code of the tiled OpenCL DGEMM.
    pub static CL_TILED_DGEMM: DeviceKernel =
        DeviceKernel::new("tiled_dgemm", include_str!("../kernels/tiled_dgemm.cl"));

    /// Name and source code of the NVIDIA CUDA SAXPY (as pre-compiled PTX).
    pub static CUDA_SAXPY: DeviceKernel =
        DeviceKernel::new("saxpy", include_str!("../kernels/saxpy.ptx"));

    /// Name and source code of the NVIDIA CUDA DAXPY (as pre-compiled PTX).
    pub static CUDA_DAXPY: DeviceKernel =
        DeviceKernel::new("daxpy", include_str!("../kernels/daxpy.ptx"));

    /// Name and source code of the naive NVIDIA CUDA SGEMM (as pre-compiled PTX).
    pub static CUDA_NAIVE_SGEMM: DeviceKernel =
        DeviceKernel::new("naive_sgemm", include_str!("../kernels/naive_sgemm.ptx"));

    /// Name and source code of the naive NVIDIA CUDA DGEMM (as pre-compiled PTX).
    pub static CUDA_NAIVE_DGEMM: DeviceKernel =
        DeviceKernel::new("naive_dgemm", include_str!("../kernels/naive_dgemm.ptx"));

    /// Name and source code of the tiled NVIDIA CUDA SGEMM (as pre-compiled PTX).
    pub static CUDA_TILED_SGEMM: DeviceKernel =
        DeviceKernel::new("tiled_sgemm", include_str!("../kernels/tiled_sgemm.ptx"));

    /// Name and source code of the tiled NVIDIA CUDA DGEMM (as pre-compiled PTX).
    pub static CUDA_TILED_DGEMM: DeviceKernel =
        DeviceKernel::new("tiled_dgemm", include_str!("../kernels/tiled_dgemm.ptx"));
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
}
