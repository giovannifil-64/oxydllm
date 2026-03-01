mod engine;
mod model;
mod model_manager;
mod sampling;
mod scheduler;
mod server;
mod tokenizer;

use std::path::PathBuf;
use std::time::Duration;

use sampling::SamplingParams;
use server::ChatMessage;
use tokenizer::Tokenizer;

struct StartArgs {
    models_dir: PathBuf,
    port: u16,
    keep_alive: Duration,
}

struct RunArgs {
    model_dir: String,
    sampling_params: SamplingParams,
}

fn default_models_dir() -> PathBuf {
    dirs_home().join(".rllm").join("models")
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

fn print_usage() {
    eprintln!(
        "\
Usage: rllm <command> [options]

Commands:
  start                Start the HTTP inference server
  run   <model-name>   Interactive chat in terminal

Server options (start):
  --port <PORT>             Listen port (default: 11313)
  --models-dir <DIR>        Models directory (default: ~/.rllm/models/)
  --keep-alive <SECS>       Keep-alive seconds before eviction (default: 900)

Chat options (run):
  --models-dir <DIR>        Models directory (default: ~/.rllm/models/)
  --temperature <T>         Sampling temperature (default: 0.7)
  --top-k <K>              Top-k filtering (default: 0, disabled)
  --top-p <P>              Nucleus sampling (default: 1.0)
  --min-p <P>              Min-p filtering (default: 0.0)
  --repeat-penalty <R>     Repetition penalty (default: 1.0)"
    );
}

fn parse_start_args(args: &[String]) -> Result<StartArgs, String> {
    let mut models_dir: Option<PathBuf> = None;
    let mut port: u16 = 11313;
    let mut keep_alive_secs: u64 = 900;
    let mut i = 0;

    while i < args.len() {
        match args[i].as_str() {
            "--port" => {
                i += 1;
                port = args
                    .get(i)
                    .ok_or("--port requires a value")?
                    .parse()
                    .map_err(|_| "Invalid port number")?;
            }
            "--models-dir" => {
                i += 1;
                models_dir = Some(PathBuf::from(
                    args.get(i).ok_or("--models-dir requires a value")?,
                ));
            }
            "--keep-alive" => {
                i += 1;
                keep_alive_secs = args
                    .get(i)
                    .ok_or("--keep-alive requires a value")?
                    .parse()
                    .map_err(|_| "Invalid keep-alive value")?;
            }
            other => return Err(format!("Unknown option: {}", other)),
        }
        i += 1;
    }

    Ok(StartArgs {
        models_dir: models_dir.unwrap_or_else(default_models_dir),
        port,
        keep_alive: Duration::from_secs(keep_alive_secs),
    })
}

fn parse_run_args(args: &[String]) -> Result<RunArgs, String> {
    let mut model_name = String::new();
    let mut models_dir: Option<PathBuf> = None;
    let mut params = SamplingParams {
        temperature: 0.7,
        ..SamplingParams::default()
    };
    let mut i = 0;

    while i < args.len() {
        match args[i].as_str() {
            "--models-dir" => {
                i += 1;
                models_dir = Some(PathBuf::from(
                    args.get(i).ok_or("--models-dir requires a value")?,
                ));
            }
            "--temperature" | "-t" => {
                i += 1;
                params.temperature = args
                    .get(i)
                    .ok_or("--temperature requires a value")?
                    .parse()
                    .map_err(|_| "Invalid temperature")?;
            }
            "--top-k" => {
                i += 1;
                params.top_k = args
                    .get(i)
                    .ok_or("--top-k requires a value")?
                    .parse()
                    .map_err(|_| "Invalid top-k")?;
            }
            "--top-p" => {
                i += 1;
                params.top_p = args
                    .get(i)
                    .ok_or("--top-p requires a value")?
                    .parse()
                    .map_err(|_| "Invalid top-p")?;
            }
            "--min-p" => {
                i += 1;
                params.min_p = args
                    .get(i)
                    .ok_or("--min-p requires a value")?
                    .parse()
                    .map_err(|_| "Invalid min-p")?;
            }
            "--repeat-penalty" => {
                i += 1;
                params.repetition_penalty = args
                    .get(i)
                    .ok_or("--repeat-penalty requires a value")?
                    .parse()
                    .map_err(|_| "Invalid repeat-penalty")?;
            }
            _ if !args[i].starts_with('-') && model_name.is_empty() => {
                model_name = args[i].clone();
            }
            other => return Err(format!("Unknown option: {}", other)),
        }
        i += 1;
    }

    if model_name.is_empty() {
        return Err("Missing <model-name>".to_string());
    }

    let model_dir = if model_name.contains('/') || model_name.contains('\\') {
        model_name
    } else {
        let base = models_dir.unwrap_or_else(default_models_dir);
        base.join(&model_name).to_string_lossy().to_string()
    };

    Ok(RunArgs {
        model_dir,
        sampling_params: params,
    })
}

fn run_interactive(args: &RunArgs) -> anyhow::Result<()> {
    use std::io::{BufRead, Write};

    let device = model::select_device()?;

    let tokenizer = Tokenizer::from_dir(&args.model_dir)?;
    println!("Tokenizer loaded.");

    println!("Loading model from '{}'...", args.model_dir);
    let batch_model = model::load_batch_model(&args.model_dir, &device, 1)?;
    let max_seq_len = batch_model.max_seq_len();
    println!(
        "Model loaded. vocab_size={}, max_seq_len={}",
        batch_model.vocab_size(),
        max_seq_len
    );

    let config = scheduler::SchedulerConfig {
        max_num_sequences: 1,
        max_tokens_per_step: 4096,
    };
    let mut engine = engine::Engine::new(batch_model, config);

    let mut messages: Vec<ChatMessage> = vec![ChatMessage {
        role: "system".to_string(),
        content: "You are a helpful assistant.".to_string(),
    }];

    println!("\nType your message (/exit to quit).\n");

    let stdin = std::io::stdin();
    let mut reader = stdin.lock();

    loop {
        print!(">>> ");
        std::io::stdout().flush()?;

        let mut input = String::new();
        if reader.read_line(&mut input)? == 0 {
            println!();
            break;
        }
        let input = input.trim().to_string();
        if input.is_empty() {
            continue;
        }
        if input == "/exit" || input == "/quit" {
            break;
        }

        messages.push(ChatMessage {
            role: "user".to_string(),
            content: input,
        });

        let prompt = server::format_chatml(&messages);
        let prompt_tokens = tokenizer.encode(&prompt)?;
        let max_tokens = max_seq_len.saturating_sub(prompt_tokens.len());

        engine.add_request(prompt_tokens, args.sampling_params.clone(), max_tokens);

        let mut response_text = String::new();
        while engine.has_pending_work() {
            let step = engine.step().map_err(|e| anyhow::anyhow!("{}", e))?;
            for tok in &step.new_tokens {
                let text = tokenizer.decode(&[tok.token])?;
                print!("{}", text);
                std::io::stdout().flush()?;
                response_text.push_str(&text);
            }
        }
        println!("\n");

        messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: response_text,
        });
    }

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    match args[1].as_str() {
        "start" => {
            let start_args = parse_start_args(&args[2..]).unwrap_or_else(|e| {
                eprintln!("Error: {e}");
                print_usage();
                std::process::exit(1);
            });
            server::start_server(start_args.models_dir, start_args.port, start_args.keep_alive)?;
        }
        "run" => {
            let run_args = parse_run_args(&args[2..]).unwrap_or_else(|e| {
                eprintln!("Error: {e}");
                print_usage();
                std::process::exit(1);
            });
            run_interactive(&run_args)?;
        }
        "--help" | "-h" | "help" => {
            print_usage();
        }
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            print_usage();
            std::process::exit(1);
        }
    }

    Ok(())
}
