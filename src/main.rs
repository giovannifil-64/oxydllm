mod model;
mod sampling;
mod tokenizer;

use sampling::SamplingParams;
use tokenizer::Tokenizer;

fn parse_args() -> (String, String, SamplingParams) {
    let args: Vec<String> = std::env::args().collect();

    let mut model_dir = String::new();
    let mut prompt = String::new();
    let mut params = SamplingParams::default();
    let mut positional = 0;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--temperature" | "-t" => {
                i += 1;
                params.temperature = args[i].parse().expect("Value of \"temperature\" is not a valid number");
            }
            "--top-k" => {
                i += 1;
                params.top_k = args[i].parse().expect("Value of \"top-k\" is not a valid integer");
            }
            "--top-p" => {
                i += 1;
                params.top_p = args[i].parse().expect("Value of \"top-p\" is not a valid number");
            }
            "--min-p" => {
                i += 1;
                params.min_p = args[i].parse().expect("Value of \"min-p\" is not a valid number");
            }
            "--repeat-penalty" => {
                i += 1;
                params.repetition_penalty = args[i].parse().expect("Value of \"repeat-penalty\" is not a valid number");
            }
            _ => {
                match positional {
                    0 => model_dir = args[i].clone(),
                    1 => prompt = args[i].clone(),
                    _ => {}
                }
                positional += 1;
            }
        }
        i += 1;
    }

    if model_dir.is_empty() || prompt.is_empty() {
        eprintln!(
            "Usage: rllm <model-dir> <prompt> [--temperature T] [--top-k K] \
             [--top-p P] [--min-p P] [--repeat-penalty R]"
        );
        std::process::exit(1);
    }

    (model_dir, prompt, params)
}

fn main() -> anyhow::Result<()> {
    let (model_dir, user_message, sampling_params) = parse_args();

    let device = model::select_device()?;

    let tokenizer = Tokenizer::from_dir(&model_dir)?;
    println!("Tokenizer loaded.");

    println!("Loading model from '{}'...", model_dir);
    let model = model::load_model(&model_dir, &device)?;
    println!("Model loaded. vocab_size={}", model.vocab_size());
    println!("Sampling: {:?}", sampling_params);

    // Qwen3 ChatML: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
    let prompt = format!(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
         <|im_start|>user\n{}<|im_end|>\n\
         <|im_start|>assistant\n",
        user_message
    );

    let prompt_tokens = tokenizer.encode(&prompt)?;
    println!("Prompt ({} tokens)", prompt_tokens.len());

    let output_tokens = model::generate(model.as_ref(), prompt_tokens, &sampling_params)?;
    let output_text = tokenizer.decode(&output_tokens)?;
    println!("\n{}", output_text);

    Ok(())
}
