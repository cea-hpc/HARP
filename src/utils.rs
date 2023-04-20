//! Utility functions and traits.

use rand::prelude::*;

use std::{
    any::Any,
    ops::{Add, AddAssign, Mul, MulAssign},
};

/// Utility trait that helps getting runtime information about types implementing it.
pub trait Object {
    fn as_any(&self) -> &dyn Any;
}

/// Runtime type-checking utility.
pub fn is_of_type<T: 'static>(x: &dyn Object) -> bool {
    x.as_any().is::<T>()
}

impl Object for f32 {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Object for f64 {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Utility trait that generalizes floating-point types in HARP and implements common
/// functionnalities needed by the kernel and driver functions.
///
/// Also provides a generic way of generating floating-point scalars and vectors in the wanted type.
pub trait HarpFloat:
    num::Float + Default + Add + AddAssign + Mul + MulAssign + Send + Sync + Object
{
    /// Produces a random scalar of type `T`, in the range [0.0, 100.0).
    fn rand_scalar(seed: Option<u64>) -> Self;
    /// Produces a random vector of type `T` and length `n`, filled with values in the range
    /// [0.0, 100.0).
    fn rand_vector(n: usize, seed: Option<u64>) -> Vec<Self>;
}

impl HarpFloat for f32 {
    fn rand_scalar(seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or(0);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
        let between = rand::distributions::Uniform::new(0.0_f32, 100.0_f32);
        between.sample(&mut rng)
    }

    fn rand_vector(n: usize, seed: Option<u64>) -> Vec<Self> {
        let seed = seed.unwrap_or(0);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
        let between = rand::distributions::Uniform::new(0.0_f32, 100.0_f32);
        (0..n).map(|_| between.sample(&mut rng)).collect()
    }
}

impl HarpFloat for f64 {
    fn rand_scalar(seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or(0);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
        let between = rand::distributions::Uniform::new(0.0_f64, 100.0_f64);
        between.sample(&mut rng)
    }

    fn rand_vector(n: usize, seed: Option<u64>) -> Vec<Self> {
        let seed = seed.unwrap_or(0);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
        let between = rand::distributions::Uniform::new(0.0_f64, 100.0_f64);
        (0..n).map(|_| between.sample(&mut rng)).collect()
    }
}
