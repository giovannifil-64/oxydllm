mod model;
mod tokenizer;

use tokenizer::Tokenizer;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: rllm <path-to-model-folder> <prompt>");
        std::process::exit(1);
    }
    let model_dir = &args[1];
    let user_message = &args[2];

    let device = model::select_device()?;

    let tokenizer = Tokenizer::from_dir(model_dir)?;
    println!("Tokenizer loaded.");

    println!("Loading model from '{}'...", model_dir);
    let model = model::load_model(model_dir, &device)?;
    println!("Model loaded. vocab_size={}", model.vocab_size());

    // Qwen3 ChatML: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
    let prompt = format!(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
         <|im_start|>user\n{}<|im_end|>\n\
         <|im_start|>assistant\n",
        user_message
    );

    let prompt_tokens = tokenizer.encode(&prompt)?;
    println!("Prompt ({} tokens)", prompt_tokens.len());

    let output_tokens = model::generate(model.as_ref(), prompt_tokens)?;
    let output_text = tokenizer.decode(&output_tokens)?;
    println!("\n{}", output_text);

    Ok(())
}
