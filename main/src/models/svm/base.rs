// File: src/models/svm/base.rs
//
// This file implements the core SVM functionality with a flexible 
// optimization strategy pattern, similar to the logistic regression implementation.

//use std::arch::x86_64::*;
//use std::sync::Arc;
//use rayon::prelude::*;

// Feature flags for different CPU architectures and optimizations
//#[cfg(target_arch = "x86_64")]
//use std::arch::x86_64 as arch;
//#[cfg(target_arch = "aarch64")] 
//use std::arch::aarch64 as arch;

// Trait for different optimization strategies
pub trait OptimizationStrategy {
    fn forward(&self, weights: &[f32], input: &[f32], bias: f32) -> f32;
}

// Basic sequential implementation
pub struct Sequential;
impl OptimizationStrategy for Sequential {
    fn forward(&self, weights: &[f32], input: &[f32], bias: f32) -> f32 {
        let dot_product: f32 = weights.iter()
            .zip(input.iter())
            .map(|(w, x)| w * x)
            .sum();
        dot_product + bias
    }
}

// Main SVM struct that can use different optimization strategies
pub struct SupportVectorMachine<T: OptimizationStrategy> {
    weights: Vec<f32>,
    bias: f32,
    strategy: T,
}

impl<T: OptimizationStrategy> SupportVectorMachine<T> {
    pub fn new(weights: Vec<f32>, bias: f32, strategy: T) -> Self {
        Self {
            weights,
            bias, 
            strategy,
        }
    }

    pub fn predict(&self, input: &[f32]) -> i32 {
        assert_eq!(self.weights.len(), input.len(), "Input dimension mismatch");
        let score = self.strategy.forward(&self.weights, input, self.bias);
        if score > 0.0 { 1 } else { -1 }
    }
}


