//! Crate-level constants.

/// Default number of meta repetitions performed when benchmarking a kernel.
pub const META_REPETITIONS: u8 = 31;

/// Default number of tight loop repetitions performed when benchmarking a kernel.
pub const TIGHT_LOOP_REPETITIONS: u16 = 1;

/// Vector block size.
pub const BLOCK_SIZE_1D: usize = 1024;

/// Matrix block size.
pub const BLOCK_SIZE_2D: usize = 32;

/// Number of items to process per CUDA threads.
// NOTE: specific to CUDA tiled DGEMM.
pub const WORK_PER_THREAD: usize = 8;
