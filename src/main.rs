use clap::ValueHint;
use fast_inference::benchmarks;
use fast_inference::models::logistic::base::{
    LogisticRegression, Sequential as LogisticSequential,
};
use fast_inference::models::svm::base::{Sequential as SVMSequential, SupportVectorMachine};

use anyhow::{bail, Result};
use clap::{Parser, ValueEnum};

use std::fmt::{Display, Formatter};
#[cfg(target_arch = "x86_64")]
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::PathBuf;

#[derive(Clone, Debug, ValueEnum)]
pub enum ModelType {
    Logistic,
    Svm,
}

impl Display for ModelType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::Logistic => write!(f, "Logistic"),
            ModelType::Svm => write!(f, "SVM"),
        }
    }
}

/// InfernoInference
#[derive(Parser)]
#[command(author, about, version)]
struct Args {
    /// Binary file containing model parameters
    #[arg(short, long, value_hint = ValueHint::FilePath)]
    pub parameters: PathBuf,

    /// Binary file containing input data matrix
    #[arg(short, long, value_hint = ValueHint::FilePath)]
    pub data: PathBuf,

    /// Model type
    #[arg(short, long)]
    pub model: ModelType,

    /// Run benchmarks instead of inference
    #[arg(short, long, default_value_t = false)]
    pub benchmarks: bool,

    #[arg(short, long, default_value_t = 50)]
    pub trials: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("\nModel Type: {}", args.model);
    let mut params_file = BufReader::new(File::open(&args.parameters)?);
    let mut params_bytes = Vec::new();
    params_file.read_to_end(&mut params_bytes)?;
    // bytes -> f32 array
    let params: Vec<f32> = params_bytes
        .chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    // create output file to write results to when not benchmarking
    let mut out_file = File::create("output")?;

    // Split into weights and bias
    let _bias = params[params.len() - 1];
    let weights = params[..params.len() - 1].to_vec();

    // Read data matrix
    let mut data_file = BufReader::new(File::open(&args.data)?);
    let mut data_bytes = Vec::new();
    data_file.read_to_end(&mut data_bytes)?;

    // bytes -> f32 matrix
    let data: Vec<f32> = data_bytes
        .chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    // Process based on model type
    let num_features = weights.len();
    let num_samples = data.len() / num_features;
    println!("Number of samples: {}", num_samples);
    println!("Feature dimension: {}", num_features);

    // If benchmark flag is present, run benchmarks
    if args.benchmarks {
        #[allow(unreachable_code)]
        match args.model {
            ModelType::Logistic => {
                #[cfg(target_arch = "x86_64")]
                {
                    benchmarks::logistic::run_benchmarks(
                        &args.parameters,
                        &args.data,
                        args.trials,
                    )?;
                    return Ok(());
                }

                #[cfg(target_arch = "aarch64")]
                {
                    let model_neon = LogisticRegression::new(weights.clone(), bias, NEON);
                    let num_trials = matches
                        .get_one::<String>("trials")
                        .unwrap()
                        .parse::<usize>()
                        .expect("Trials must be a positive integer");

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
                    let variance = times_neon.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                        / times_neon.len() as f64;
                    let std_dev = variance.sqrt();

                    println!("Mean Time: {:.2e} ± {:.2e} seconds", mean, std_dev);
                    return Ok(());
                }

                bail!("Unknown/unsupported arch");
            }

            ModelType::Svm => {
                bail!("Benchmarking not yet implemented for SVM");
            }
        }
    }

    let mut params_file = BufReader::new(File::open(args.parameters)?);
    let mut params_bytes = Vec::new();
    params_file.read_to_end(&mut params_bytes)?;

    // bytes -> f32 array
    let params: Vec<f32> = params_bytes
        .chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    // bias is last value in array
    let bias = params[params.len() - 1];
    let weights = params[..params.len() - 1].to_vec();

    // Read data matrix
    let mut data_file = BufReader::new(File::open(&args.data)?);
    let mut data_bytes = Vec::new();
    data_file.read_to_end(&mut data_bytes)?;

    // bytes -> f32 matrix
    let data: Vec<f32> = data_bytes
        .chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    // Process based on model type
    let num_features = weights.len();
    let num_samples = data.len() / num_features;
    println!("Number of samples: {}", num_samples);
    println!("Feature dimension: {}", num_features);

    match args.model {
        ModelType::Logistic => {
            let model = LogisticRegression::new(weights.clone(), bias, LogisticSequential);
            for chunk in data.chunks(num_features) {
                let prob = model.predict(chunk);
                let prediction: u8 = if prob > 0.5 { 1 } else { 0 };
                out_file.write_all(format!("{}\n", prediction).as_bytes())?;
            }
        }
        ModelType::Svm => {
            let model = SupportVectorMachine::new(weights.clone(), bias, SVMSequential);

            for chunk in data.chunks(num_features) {
                let prediction = model.predict(chunk);
                out_file.write_all(format!("{}\n", prediction).as_bytes())?;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Args;

    use clap::CommandFactory;

    #[test]
    fn verify_cli() {
        Args::command().debug_assert();
    }
}