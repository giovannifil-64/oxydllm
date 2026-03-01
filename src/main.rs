mod engine;
mod model;
mod sampling;
mod scheduler;
mod tokenizer;

use sampling::SamplingParams;
use tokenizer::Tokenizer;

struct Args {
    model_dir: String,
    prompt: String,
    sampling_params: SamplingParams,
    use_engine: bool,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();

    let mut model_dir = String::new();
    let mut prompt = String::new();
    let mut params = SamplingParams::default();
    let mut use_engine = false;
    let mut positional = 0;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--engine" => {
                use_engine = true;
            }
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
            "Usage: rllm <model-dir> <prompt> [--engine] [--temperature T] [--top-k K] \
             [--top-p P] [--min-p P] [--repeat-penalty R]"
        );
        std::process::exit(1);
    }

    Args { model_dir, prompt, sampling_params: params, use_engine }
}

fn main() -> anyhow::Result<()> {
    let args = parse_args();

    let device = model::select_device()?;

    let tokenizer = Tokenizer::from_dir(&args.model_dir)?;
    println!("Tokenizer loaded.");

    // Qwen3 ChatML template
    let prompt = format!(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
         <|im_start|>user\n{}<|im_end|>\n\
         <|im_start|>assistant\n",
        args.prompt
    );
    let prompt_tokens = tokenizer.encode(&prompt)?;
    println!("Prompt ({} tokens)", prompt_tokens.len());

    if args.use_engine {
        use std::io::Write;

        println!("Loading model from '{}' (engine mode)...", args.model_dir);
        let batch_model = model::load_batch_model(&args.model_dir, &device)?;
        let max_seq_len = batch_model.max_seq_len();
        println!("Model loaded. vocab_size={}", batch_model.vocab_size());
        println!("Sampling: {:?}", args.sampling_params);

        let config = scheduler::SchedulerConfig {
            max_num_sequences: 4,
            max_tokens_per_step: 4096,
        };
        let mut engine = engine::Engine::new(batch_model, config);
        let max_tokens = max_seq_len.saturating_sub(prompt_tokens.len());
        engine.add_request(prompt_tokens, args.sampling_params, max_tokens);

        print!("\n");
        while engine.has_pending_work() {
            let step = engine.step()?;
            for tok in &step.new_tokens {
                let text = tokenizer.decode(&[tok.token])?;
                print!("{}", text);
                std::io::stdout().flush()?;
            }
        }
        println!();
    } else {
        println!("Loading model from '{}'...", args.model_dir);
        let mut model = model::load_model(&args.model_dir, &device)?;
        println!("Model loaded. vocab_size={}", model.vocab_size());
        println!("Sampling: {:?}", args.sampling_params);

        let output_tokens = model::generate(model.as_mut(), prompt_tokens, &args.sampling_params)?;
        let output_text = tokenizer.decode(&output_tokens)?;
        println!("\n{}", output_text);
    }

    Ok(())
}
