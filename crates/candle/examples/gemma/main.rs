use std::fs::File;

use candle::{hf_retrieve, load_safetensors};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::gemma::{Config, Model};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use tokenizers::Tokenizer;

use crate::text::TextGeneration;

pub mod text;

const MODEL: &str = "google/gemma-2b-it";
const REVISION: &str = "main";

const TOKENIZER: &str = "tokenizer.json";
const CONFIG: &str = "config.json";
const SAFETENSORS: &str = "model.safetensors.index.json";

const SEED: u64 = 299792458;
const TEMPERATURE: f64 = 1_f64;
const TOP_P: f64 = 0_f64;
const SAMPLE_LEN: usize = 10000;
const REPEAT_PENALTY: f32 = 1.;
const REPEAT_LAST_N: usize = 64;

const DEFAULT_PROMPT: &str = "Hello! What's your name?";

fn main() {
    let mut start = std::time::Instant::now();

    let token = std::env::var("HF_TOKEN").unwrap();

    let api = ApiBuilder::new().with_token(Some(token)).build().unwrap();

    let model_id = MODEL.to_string();
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        REVISION.to_string(),
    ));
    let tokenizer_filename = hf_retrieve(&repo, TOKENIZER);
    let config_filename = hf_retrieve(&repo, CONFIG);
    let filenames = load_safetensors(&repo, SAFETENSORS);
    if let Some(filenames) = filenames {
        println!("loaded the safetensors in {:?}", start.elapsed());
        if let Ok(tokenizer) = Tokenizer::from_file(tokenizer_filename) {
            if let Ok(file) = File::open(config_filename) {
                if let Ok(config) = serde_json::from_reader::<File, Config>(file) {
                    println!("loaded the config in {:?}", start.elapsed());

                    start = std::time::Instant::now();
                    let device = Device::cuda_if_available(0).unwrap();
                    let dtype = DType::F32;

                    let vb = unsafe {
                        VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device).unwrap()
                    };
                    let model = Model::new(&config, vb).unwrap();
                    println!("loaded the model in {:?}", start.elapsed());

                    let mut pipeline = TextGeneration::new(
                        model,
                        tokenizer,
                        SEED,
                        Some(TEMPERATURE),
                        Some(TOP_P),
                        REPEAT_PENALTY,
                        REPEAT_LAST_N,
                        &device,
                    );
                    pipeline.run(DEFAULT_PROMPT, SAMPLE_LEN);
                } else {
                    println!("failed to load config");
                }
            } else {
                println!("failed to load config");
            }
        } else {
            println!("failed to load tokenizer");
        }
    }
}
