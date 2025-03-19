use std::fs::File;
use std::io::{self, BufReader, Read};
use std::time::Instant;

use crate::models::logistic::base::{LogisticRegression,Sequential};
use crate::models::logistic::simd_x86::{SSE, AVX};

fn calculate_stats(times: &[f64]) -> (f64, f64) {
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let variance = times.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / times.len() as f64;
    let std_dev = variance.sqrt();
    (mean, std_dev)
}

pub fn run_benchmarks(params_path: &str, data_path: &str, num_trials_str: &str) -> io::Result<()> {
    // Read parameters file
    let mut params_file = BufReader::new(File::open(params_path)?);
    let mut params_bytes = Vec::new();
    params_file.read_to_end(&mut params_bytes)?;

    // Convert bytes to f32 array
    let params: Vec<f32> = params_bytes.chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    // Split into weights and bias
    let bias = params[params.len()-1];
    let weights = params[..params.len()-1].to_vec();

    // Read data matrix
    let mut data_file = BufReader::new(File::open(data_path)?);
    let mut data_bytes = Vec::new();
    data_file.read_to_end(&mut data_bytes)?;

    // Convert bytes to f32 matrix
    let data: Vec<f32> = data_bytes.chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let num_features = weights.len();
    let num_trials = num_trials_str.parse::<usize>()
        .expect("Trials must be a positive integer");

    println!("\nBenchmarking logistic regression implementations:");
    println!("FEATURE DIMENSION: {}", num_features);
    println!("NUMBER OF TRIALS: {}", num_trials);

    // Baseline sequential implementation
    println!("\n1. Baseline sequential implementation:");
    let model_seq = LogisticRegression::new(weights.clone(), bias, Sequential);
    let mut times_seq = Vec::with_capacity(num_trials);
    
    for _ in 0..num_trials {
        let start = Instant::now();
        for chunk in data.chunks(num_features) {
            let _ = model_seq.predict(chunk);
        }
        times_seq.push(start.elapsed().as_secs_f64());
    }
    let (mean_seq, std_seq) = calculate_stats(&times_seq);
    println!("Mean Time: {:.2e} ± {:.2e} seconds", mean_seq, std_seq);

    // SSE implementation
    println!("\n2. SSE SIMD implementation:");
    let model_sse = LogisticRegression::new(weights.clone(), bias, SSE);
    let mut times_sse = Vec::with_capacity(num_trials);
    
    for _ in 0..num_trials {
        let start = Instant::now();
        for chunk in data.chunks(num_features) {
            let _ = model_sse.predict(chunk);
        }
        times_sse.push(start.elapsed().as_secs_f64());
    }
    let (mean_sse, std_sse) = calculate_stats(&times_sse);
    println!("Mean Time: {:.2e} ± {:.2e} seconds", mean_sse, std_sse);

    // AVX implementation
    println!("\n3. AVX SIMD implementation:");
    let model_avx = LogisticRegression::new(weights.clone(), bias, AVX);
    let mut times_avx = Vec::with_capacity(num_trials);
    
    for _ in 0..num_trials {
        let start = Instant::now();
        for chunk in data.chunks(num_features) {
            let _ = model_avx.predict(chunk);
        }
        times_avx.push(start.elapsed().as_secs_f64());
    }
    let (mean_avx, std_avx) = calculate_stats(&times_avx);
    println!("Mean Time: {:.2e} ± {:.2e} seconds", mean_avx, std_avx);

    Ok(())
}

