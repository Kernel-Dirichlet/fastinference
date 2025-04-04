// File: src/models/logistic/simd.rs
//
// This file implements SIMD-optimized logistic regression using SSE and AVX instructions.
// The optimizations focus on vectorized dot products and efficient sigmoid calculations.
//
// Project structure:
// src/
//   models/
//     logistic/
//       mod.rs        - Module exports
//       base.rs       - Core trait definitions
//       simd.rs       - This file - SIMD optimizations
//       multicore.rs  - Multi-threading optimizations

//
// SIMD Register Types:
// - SSE: 128-bit registers that can process 4 x 32-bit floats in parallel
// - AVX: 256-bit registers that can process 8 x 32-bit floats in parallel
//
// The implementations below focus on single-core SIMD optimizations.
// Multi-threading optimizations are handled separately in multicore.rs.

use crate::models::logistic::base::OptimizationStrategy;
use std::arch::x86_64::*;

// SSE optimized implementation using 128-bit registers
#[cfg(target_arch = "x86_64")]
pub struct SSE;

#[cfg(target_arch = "x86_64")]
impl OptimizationStrategy for SSE {
    fn forward(&self, weights: &[f32], input: &[f32], bias: f32) -> f32 {
        unsafe {
            let feature_dim = weights.len();
            let mut sum_vec = _mm_setzero_ps();
            let mut i = 0;

            // Check 16-byte alignment for optimal SSE performance
            let input_aligned = (input.as_ptr() as usize) % 16 == 0;
            let weights_aligned = (weights.as_ptr() as usize) % 16 == 0;

            // Process 4 elements at a time using SSE
            if input_aligned && weights_aligned {
                while i + 4 <= feature_dim {
                    let x_vec = _mm_load_ps(&input[i]);
                    let w_vec = _mm_load_ps(&weights[i]);
                    sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(x_vec, w_vec));
                    i += 4;
                }
            } else {
                while i + 4 <= feature_dim {
                    let x_vec = _mm_loadu_ps(&input[i]);
                    let w_vec = _mm_loadu_ps(&weights[i]);
                    sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(x_vec, w_vec));
                    i += 4;
                }
            }

            // Handle remaining elements sequentially
            let mut sum_scalar = 0.0;
            while i < feature_dim {
                sum_scalar += input[i] * weights[i];
                i += 1;
            }

            // Combine SSE vector sum with scalar sum
            let mut sum_array: [f32; 4] = [0.0; 4];
            _mm_storeu_ps(sum_array.as_mut_ptr(), sum_vec);
            let sum = sum_array.iter().sum::<f32>() + sum_scalar + bias;

            sigmoid(sum)
        }
    }
}

// AVX optimized implementation using 256-bit registers
#[cfg(target_arch = "x86_64")]
pub struct AVX;

#[cfg(target_arch = "x86_64")]
impl OptimizationStrategy for AVX {
    fn forward(&self, weights: &[f32], input: &[f32], bias: f32) -> f32 {
        unsafe {
            let feature_dim = weights.len();
            let mut sum_vec = _mm256_setzero_ps();
            let mut i = 0;

            // Check 32-byte alignment for optimal AVX performance
            let input_aligned = (input.as_ptr() as usize) % 32 == 0;
            let weights_aligned = (weights.as_ptr() as usize) % 32 == 0;

            // Process 8 elements at a time using AVX
            if input_aligned && weights_aligned {
                while i + 8 <= feature_dim {
                    let x_vec = _mm256_load_ps(&input[i]);
                    let w_vec = _mm256_load_ps(&weights[i]);
                    let mul = _mm256_mul_ps(x_vec, w_vec);
                    sum_vec = _mm256_add_ps(sum_vec, mul);
                    i += 8;
                }
            } else {
                while i + 8 <= feature_dim {
                    let x_vec = _mm256_loadu_ps(&input[i]);
                    let w_vec = _mm256_loadu_ps(&weights[i]);
                    let mul = _mm256_mul_ps(x_vec, w_vec);
                    sum_vec = _mm256_add_ps(sum_vec, mul);
                    i += 8;
                }
            }

            // Handle remaining elements sequentially
            let mut sum_scalar = 0.0;
            while i < feature_dim {
                sum_scalar += input[i] * weights[i];
                i += 1;
            }

            // Combine AVX vector sum with scalar sum
            let mut sum_array: [f32; 8] = [0.0; 8];
            _mm256_storeu_ps(sum_array.as_mut_ptr(), sum_vec);
            let sum = sum_array.iter().sum::<f32>() + sum_scalar + bias;

            sigmoid(sum)
        }
    }
}
// sigmoid will be replaced later with a polynomial approximation
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
