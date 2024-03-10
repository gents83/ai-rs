// An implementation of LLaMA https://github.com/facebookresearch/llama
//
// This is based on nanoGPT in a similar way to:
// https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
//
// The tokenizer config can be retrieved from:
// https://huggingface.co/hf-internal-testing/llama-tokenizer/raw/main/tokenizer.json

use std::io::Write;

use candle::{hf_retrieve, load_safetensors, token::TokenOutputStream};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use candle_transformers::{generation::LogitsProcessor, models::llama as model};
use hf_hub::{api::sync::Api, Repo, RepoType};
use model::{Llama, LlamaConfig};

const LLAMA_MODEL: &str = "meta-llama/Llama-2-7b-hf";
const LLAMA_MODEL_DYPE: DType = DType::F16;
const LLAMA_MODEL_SAFETENSORS: &str = "model.safetensors.index.json";
const LLAMA_REVISION: &str = "main";
const LLAMA_TOKENIZER_FILENAME: &str = "tokenizer.json";
const LLAMA_CONFIG_FILENAME: &str = "config.json";

const LLAMA_MODEL_SEED: u64 = 299792458;
const LLAMA_MODEL_TEMPERATURE: f64 = 1_f64;
const LLAMA_MODEL_TOP_P: f64 = 0_f64;
const LLAMA_MODEL_SAMPLE_LEN: usize = 10000;
const LLAMA_MODEL_REPEAT_PENALTY: f32 = 1_f32;
const LLAMA_MODEL_REPEAT_LAST_N: usize = 64;

const EOS_TOKEN: &str = "</s>";
const DEFAULT_PROMPT: &str = "Hello! What's your name?";

fn main() {
    let device = Device::Cpu;

    let api = Api::new().unwrap();
    let model_id = LLAMA_MODEL.to_string();
    let revision = LLAMA_REVISION.to_string();
    let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

    let tokenizer_filename = hf_retrieve(&repo, LLAMA_TOKENIZER_FILENAME);
    let config_filename = hf_retrieve(&repo, LLAMA_CONFIG_FILENAME);
    let config: LlamaConfig =
        serde_json::from_slice(&std::fs::read(config_filename).unwrap()).unwrap();
    let config = config.into_config(true);

    if let Some(filenames) = load_safetensors(&repo, LLAMA_MODEL_SAFETENSORS) {
        let mut cache = model::Cache::new(true, LLAMA_MODEL_DYPE, &config, &device).unwrap();

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&filenames, LLAMA_MODEL_DYPE, &device).unwrap()
        };
        let llama = Llama::load(vb, &config).unwrap();

        let tokenizer = Tokenizer::from_file(tokenizer_filename).unwrap();
        let eos_token_id = tokenizer.token_to_id(EOS_TOKEN);
        let prompt = DEFAULT_PROMPT;
        let mut tokens = tokenizer.encode(prompt, true).unwrap().get_ids().to_vec();
        let mut tokenizer = TokenOutputStream::new(tokenizer);

        print!("{prompt}");
        let mut logits_processor = LogitsProcessor::new(
            LLAMA_MODEL_SEED,
            Some(LLAMA_MODEL_TEMPERATURE),
            Some(LLAMA_MODEL_TOP_P),
        );
        let start_gen = std::time::Instant::now();
        let mut index_pos = 0;
        let mut token_generated = 0;
        for index in 0..LLAMA_MODEL_SAMPLE_LEN {
            let (context_size, context_index) = if cache.use_kv_cache && index > 0 {
                (1, index_pos)
            } else {
                (tokens.len(), 0)
            };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &device).unwrap().unsqueeze(0).unwrap();
            let logits = llama.forward(&input, context_index, &mut cache).unwrap();
            let logits = logits.squeeze(0).unwrap();
            let logits = if LLAMA_MODEL_REPEAT_PENALTY == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(LLAMA_MODEL_REPEAT_LAST_N);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    LLAMA_MODEL_REPEAT_PENALTY,
                    &tokens[start_at..],
                )
                .unwrap()
            };
            index_pos += ctxt.len();

            let next_token = logits_processor.sample(&logits).unwrap();
            token_generated += 1;
            tokens.push(next_token);

            if Some(next_token) == eos_token_id {
                break;
            }
            if let Some(t) = tokenizer.next_token(next_token) {
                print!("{t}");
                let _ = std::io::stdout().flush();
            }
        }
        if let Some(rest) = tokenizer.decode_rest() {
            print!("{rest}");
        }
        let dt = start_gen.elapsed();
        println!(
            "\n\n{} tokens generated ({} token/s)\n",
            token_generated,
            token_generated as f64 / dt.as_secs_f64(),
        );
    }
}
