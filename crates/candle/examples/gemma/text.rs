use candle::token::TokenOutputStream;
use candle_core::{DType, Device, Tensor};
use candle_transformers::{generation::LogitsProcessor, models::gemma::Model};
use tokenizers::Tokenizer;

const EOS_TOKEN: &str = "<eos>";

pub struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    pub fn run(&mut self, prompt: &str, sample_len: usize) {
        let closed_prompt = prompt.to_string() + EOS_TOKEN;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(closed_prompt, true)
            .unwrap()
            .get_ids()
            .to_vec();

        println!("prompt:");
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t) {
                print!("{t}")
            }
        }

        let sample_count = tokens.len().min(sample_len);

        println!("\nanswer:");
        let mut generated_tokens = 0usize;
        if let Some(eos_token) = self.tokenizer.get_token(EOS_TOKEN) {
            let start_gen = std::time::Instant::now();
            for index in 0..sample_count {
                let context_size = if index > 0 { 1 } else { tokens.len() };
                let start_pos = tokens.len().saturating_sub(context_size);
                let ctxt = &tokens[start_pos..];
                let input = Tensor::new(ctxt, &self.device)
                    .unwrap()
                    .unsqueeze(0)
                    .unwrap();
                let logits = self.model.forward(&input, start_pos).unwrap();
                let logits = logits
                    .squeeze(0)
                    .unwrap()
                    .squeeze(0)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap();
                let logits = if self.repeat_penalty == 1. {
                    logits
                } else {
                    let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                    candle_transformers::utils::apply_repeat_penalty(
                        &logits,
                        self.repeat_penalty,
                        &tokens[start_at..],
                    )
                    .unwrap()
                };

                let next_token = self.logits_processor.sample(&logits).unwrap();
                tokens.push(next_token);
                generated_tokens += 1;
                if next_token == eos_token {
                    break;
                }
                if let Some(t) = self.tokenizer.next_token(next_token) {
                    print!("{t}");
                }
            }
            let dt = start_gen.elapsed();
            if let Some(rest) = self.tokenizer.decode_rest() {
                print!("{rest}");
            }
            println!(
                "\n{generated_tokens} tokens generated ({:.2} token/s)",
                generated_tokens as f64 / dt.as_secs_f64(),
            );
        } else {
            println!("cannot find the {:?} token", EOS_TOKEN);
        }
    }
}
