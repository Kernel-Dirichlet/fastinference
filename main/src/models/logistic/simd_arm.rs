// File: src/models/logistic/simd_arm.rs
//
// This file implements SIMD-optimized logistic regression using NEON instructions.
// The optimizations focus on vectorized dot products and efficient sigmoid calculations.

#[cfg(target_arch = "aarch64")]
use std::arch::is_aarch64_feature_detected;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
use crate::models::logistic::base::OptimizationStrategy;

// NEON optimized implementation using 128-bit registers
pub struct NEON;

#[cfg(target_arch = "aarch64")]
impl OptimizationStrategy for NEON {
    fn forward(&self, weights: &[f32], input: &[f32], bias: f32) -> f32 {
        if !is_aarch64_feature_detected!("neon") {
            // Fallback to scalar implementation if NEON not available
            return scalar_forward(weights, input, bias);
        }

        unsafe {
            let feature_dim = weights.len();
            let mut sum_vec = vdupq_n_f32(0.0);
            let mut i = 0;

            // Process 4 elements at a time using NEON
            while i + 4 <= feature_dim {
                let x_vec = vld1q_f32(&input[i]);
                let w_vec = vld1q_f32(&weights[i]);
                sum_vec = vfmaq_f32(sum_vec, x_vec, w_vec);
                i += 4;
            }

            // Handle remaining elements sequentially
            let mut sum_scalar = 0.0;
            while i < feature_dim {
                sum_scalar += input[i] * weights[i];
                i += 1;
            }

            // Combine NEON vector sum with scalar sum
            let mut sum_array: [f32; 4] = [0.0; 4];
            vst1q_f32(sum_array.as_mut_ptr(), sum_vec);
            let sum = sum_array.iter().sum::<f32>() + sum_scalar + bias;

            sigmoid(sum)
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl OptimizationStrategy for NEON {
    fn forward(&self, weights: &[f32], input: &[f32], bias: f32) -> f32 {
        scalar_forward(weights, input, bias)
    }
}

fn scalar_forward(weights: &[f32], input: &[f32], bias: f32) -> f32 {
    let sum = weights.iter()
        .zip(input.iter())
        .map(|(w, x)| w * x)
        .sum::<f32>() + bias;
    sigmoid(sum)
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

