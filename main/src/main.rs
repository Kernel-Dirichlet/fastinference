#[cfg(target_arch = "x86_64")]
use std::fs::File;
#[cfg(target_arch = "aarch64")] 
use std::fs::File;
use std::io::prelude::*;
use std::io::{self, BufReader, Read};
use clap::{Command, Arg};

mod models {
    pub mod logistic {
        pub mod base;
        #[cfg(target_arch = "x86_64")]
        pub mod simd_x86;
        #[cfg(target_arch = "aarch64")]
        pub mod simd_arm;
    }
    pub mod svm {
        pub mod base;
    }
}

mod benchmarks {
    pub mod logistic;
}

use crate::models::logistic::base::{LogisticRegression, Sequential as LogisticSequential};
use crate::models::svm::base::{SupportVectorMachine, Sequential as SVMSequential};
fn main() -> io::Result<()> {
    let matches = Command::new("InfernoInference")
        .arg(Arg::new("parameters")
            .long("parameters")
            .required(true)
            .help("Binary file containing model parameters"))
        .arg(Arg::new("data")
            .long("data")
            .required(true)
            .help("Binary file containing input data matrix"))
        .arg(Arg::new("model")
            .long("model")
            .required(true)
            .help("Model type: 'logistic' or 'svm'"))
        .arg(Arg::new("benchmark")
            .long("benchmark")
            .help("Run benchmarks instead of inference")
            .required(false))
        .arg(Arg::new("trials")
            .long("trials")
            .help("Number of trials for benchmarking")
            .default_value("50"))
        .arg(Arg::new("arch")
            .long("arch")
            .help("Architecture to benchmark: 'x86' or 'arm'")
            .default_value(if cfg!(target_arch = "x86_64") { "x86" } else { "arm" }))
        .get_matches();

    let model_type = matches.get_one::<String>("model").unwrap();
    println!("\nModel Type: {}", model_type);
    let params_path = matches.get_one::<String>("parameters").unwrap();
    let mut params_file = BufReader::new(File::open(params_path)?);
    let mut params_bytes = Vec::new();
    params_file.read_to_end(&mut params_bytes)?;
     // bytes -> f32 array
    let params: Vec<f32> = params_bytes.chunks(4)
     .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
     .collect();
    

    // create output file to write results to when not benchmarking
    let mut out_file = File::create("output")?;

    // Split into weights and bias
    let _bias = params[params.len()-1];
    let weights = params[..params.len()-1].to_vec();
    
    // Read data matrix
    let data_path = matches.get_one::<String>("data").unwrap();
    let mut data_file = BufReader::new(File::open(data_path)?);
    let mut data_bytes = Vec::new();
    data_file.read_to_end(&mut data_bytes)?;

    // bytes -> f32 matrix
    let data: Vec<f32> = data_bytes.chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    // Process based on model type
    let num_features = weights.len();
    let num_samples = data.len() / num_features;
    println!("Number of samples: {}", num_samples);
    println!("Feature dimension: {}", num_features);

    // If benchmark flag is present, run benchmarks
    if matches.contains_id("benchmark") {
        
        match model_type.as_str() {
            "logistic" => {
                let arch = matches.get_one::<String>("arch").unwrap();
                match arch.as_str() {
                    "x86" => {
                        #[cfg(target_arch = "x86_64")]
                        {
                            benchmarks::logistic::run_benchmarks(
                                matches.get_one::<String>("parameters").unwrap(),
                                matches.get_one::<String>("data").unwrap(),
                                matches.get_one::<String>("trials").unwrap()
                            )?;
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        {
                            eprintln!("x86 benchmarks not available on this architecture");
                            return Ok(());
                        }
                    },
                    "arm" => {
                        #[cfg(target_arch = "aarch64")]
                        {
                            let model_neon = LogisticRegression::new(weights.clone(), bias, NEON);
                            let num_trials = matches.get_one::<String>("trials").unwrap()
                                .parse::<usize>().expect("Trials must be a positive integer");
                            
                            println!("\nBenchmarking ARM NEON implementation:");
                            let mut times_neon = Vec::with_capacity(num_trials);
                            
                            for _ in 0..num_trials {
                                let start = Instant::now();
                                for chunk in data.chunks(num_features) {
                                    let _ = model_neon.predict(chunk);
                                }
                                times_neon.push(start.elapsed().as_secs_f64());
                            }
                            
                            let mean = times_neon.iter().sum::<f64>() / times_neon.len() as f64;
                            let variance = times_neon.iter()
                                .map(|x| (x - mean).powi(2))
                                .sum::<f64>() / times_neon.len() as f64;
                            let std_dev = variance.sqrt();
                            
                            println!("Mean Time: {:.2e} ± {:.2e} seconds", mean, std_dev);
                        }
                        #[cfg(not(target_arch = "aarch64"))]
                        {
                            eprintln!("ARM benchmarks not available on this architecture");
                            return Ok(());
                        }
                    },
                    _ => {
                        eprintln!("Invalid architecture. Must be 'x86' or 'arm'");
                        return Ok(());
                    }
                }
                return Ok(());
            },
            "svm" => {
                eprintln!("Benchmarking not yet implemented for SVM");
                return Ok(());
            },
            _ => {
                eprintln!("Invalid model type for benchmarking. Must be 'logistic' or 'svm'");
                return Ok(());
            }
        }
    }

    let params_path = matches.get_one::<String>("parameters").unwrap();
    let mut params_file = BufReader::new(File::open(params_path)?);
    let mut params_bytes = Vec::new();
    params_file.read_to_end(&mut params_bytes)?;

    // bytes -> f32 array
    let params: Vec<f32> = params_bytes.chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    // bias is last value in array
    let bias = params[params.len()-1];
    let weights = params[..params.len()-1].to_vec();

    // Read data matrix
    let data_path = matches.get_one::<String>("data").unwrap();
    let mut data_file = BufReader::new(File::open(data_path)?);
    let mut data_bytes = Vec::new();
    data_file.read_to_end(&mut data_bytes)?;

    // bytes -> f32 matrix
    let data: Vec<f32> = data_bytes.chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    // Process based on model type
    let num_features = weights.len();
    let num_samples = data.len() / num_features;
    println!("Number of samples: {}", num_samples);
    println!("Feature dimension: {}", num_features);

    match model_type.as_str() {
        "logistic" => {
            let model = LogisticRegression::new(weights.clone(), bias, LogisticSequential);
            for chunk in data.chunks(num_features) {
                let prob = model.predict(chunk);
                let prediction: u8 = if prob > 0.5 { 1 } else { 0 };
                out_file.write_all(format!("{}\n",prediction).as_bytes())?;
            }
        },
        "svm" => {
            let model = SupportVectorMachine::new(weights.clone(), bias, SVMSequential);
            
            for chunk in data.chunks(num_features) {
                let prediction = model.predict(chunk);
                out_file.write_all(format!("{}\n",prediction).as_bytes())?;
            }
        },
        _ => {
            eprintln!("Invalid model type. Must be 'logistic' or 'svm'");
            return Ok(());
        }
    }

    Ok(())
}

