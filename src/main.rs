mod chat_template;
mod common;
mod engine;
mod models;
mod sampling;
mod scheduler;
mod server;
mod tokenizer;

use std::path::PathBuf;
use std::time::Duration;

use sampling::SamplingParams;
use server::ChatMessage;
use tokenizer::Tokenizer;

struct PullArgs {
    repo_id: String,
    models_dir: PathBuf,
    name: Option<String>,
    token: Option<String>,
    force: bool,
}

struct StartArgs {
    models_dir: PathBuf,
    port: u16,
    keep_alive: Duration,
    memory_budget_bytes: Option<usize>,
    cuda_devices: Vec<usize>,
}

struct RunArgs {
    model_dir: String,
    sampling_params: SamplingParams,
    cuda_device: Option<usize>,
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

fn parse_devices(s: &str) -> Result<Vec<usize>, String> {
    s.split(',')
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .map_err(|_| format!("Invalid device index: '{}'  (expected integer)", part.trim()))
        })
        .collect()
}

fn resolve_devices(explicit: Option<Vec<usize>>) -> Vec<usize> {
    if let Some(d) = explicit {
        return d;
    }
    if let Ok(env) = std::env::var("RLLM_DEVICES") {
        if !env.trim().is_empty() {
            match parse_devices(&env) {
                Ok(d) => return d,
                Err(e) => eprintln!("Warning: RLLM_DEVICES ignored — {}", e),
            }
        }
    }
    vec![]
}

fn print_usage() {
    eprintln!(
        "\
Usage: rllm <command> [options]

Commands:
  pull  <user/model>   Download a model from HuggingFace
  start                Start the HTTP inference server
  run   <model-name>   Interactive chat in terminal

Download options (pull):
  --models-dir <DIR>        Destination directory (default: ~/.rllm/models/)
  --name <NAME>             Folder name override (default: model name)
  --token <TOKEN>           HuggingFace token for gated models
  --force                   Overwrite if model already exists

Server options (start):
  --port <PORT>             Listen port (default: 11313)
  --models-dir <DIR>        Models directory (default: ~/.rllm/models/)
  --keep-alive <SECS>       Keep-alive seconds before eviction (default: 900)
  --memory-budget <MB>      Max total VRAM for loaded models in MB; LRU eviction when exceeded
  --devices <IDS>           Comma-separated CUDA device indices to use (default: auto, env: RLLM_DEVICES)
                            Examples: --devices 0   --devices 0,1,2

Chat options (run):
  --models-dir <DIR>        Models directory (default: ~/.rllm/models/)
  --devices <ID>            CUDA device index to use (default: auto, env: RLLM_DEVICES)
  --temperature <T>         Sampling temperature (default: 0.7)
  --top-k <K>               Top-k filtering (default: 0, disabled)
  --top-p <P>               Nucleus sampling (default: 1.0)
  --min-p <P>               Min-p filtering (default: 0.0)
  --repeat-penalty <R>      Repetition penalty (default: 1.0)"
    );
}

fn parse_pull_args(args: &[String]) -> Result<PullArgs, String> {
    let mut repo_id = String::new();
    let mut models_dir: Option<PathBuf> = None;
    let mut name: Option<String> = None;
    let mut token: Option<String> = None;
    let mut force = false;
    let mut i = 0;

    while i < args.len() {
        match args[i].as_str() {
            "--models-dir" => {
                i += 1;
                models_dir = Some(PathBuf::from(
                    args.get(i).ok_or("--models-dir requires a value")?,
                ));
            }
            "--name" => {
                i += 1;
                name = Some(args.get(i).ok_or("--name requires a value")?.clone());
            }
            "--token" => {
                i += 1;
                token = Some(args.get(i).ok_or("--token requires a value")?.clone());
            }
            "--force" => {
                force = true;
            }
            _ if !args[i].starts_with('-') && repo_id.is_empty() => {
                repo_id = args[i].clone();
            }
            other => return Err(format!("Unknown option: {}", other)),
        }
        i += 1;
    }

    if repo_id.is_empty() {
        return Err("Missing <username/model-name>  (e.g. Qwen/Qwen3-0.6B)".to_string());
    }
    if !repo_id.contains('/') {
        return Err(format!(
            "Invalid repo format '{}': expected 'username/model-name' (e.g. Qwen/Qwen3-0.6B)",
            repo_id
        ));
    }

    Ok(PullArgs {
        repo_id,
        models_dir: models_dir.unwrap_or_else(default_models_dir),
        name,
        token,
        force,
    })
}

fn parse_start_args(args: &[String]) -> Result<StartArgs, String> {
    let mut models_dir: Option<PathBuf> = None;
    let mut port: u16 = 11313;
    let mut keep_alive_secs: u64 = 900;
    let mut memory_budget_mb: Option<usize> = None;
    let mut devices_raw: Option<Vec<usize>> = None;
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
            "--memory-budget" => {
                i += 1;
                let mb: usize = args
                    .get(i)
                    .ok_or("--memory-budget requires a value")?
                    .parse()
                    .map_err(|_| "Invalid memory-budget value (expected MB integer)")?;
                memory_budget_mb = Some(mb);
            }
            "--devices" => {
                i += 1;
                devices_raw = Some(
                    parse_devices(args.get(i).ok_or("--devices requires a value")?)
                        .map_err(|e| e)?,
                );
            }
            other => return Err(format!("Unknown option: {}", other)),
        }
        i += 1;
    }

    Ok(StartArgs {
        models_dir: models_dir.unwrap_or_else(default_models_dir),
        port,
        keep_alive: Duration::from_secs(keep_alive_secs),
        memory_budget_bytes: memory_budget_mb.map(|mb| mb * 1024 * 1024),
        cuda_devices: resolve_devices(devices_raw),
    })
}

fn parse_run_args(args: &[String]) -> Result<RunArgs, String> {
    let mut model_name = String::new();
    let mut models_dir: Option<PathBuf> = None;
    let mut devices_raw: Option<Vec<usize>> = None;
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
            "--devices" => {
                i += 1;
                let raw = args.get(i).ok_or("--devices requires a value")?;
                let d = parse_devices(raw)?;
                devices_raw = Some(d);
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
        cuda_device: resolve_devices(devices_raw).into_iter().next(),
    })
}

fn run_interactive(args: &RunArgs) -> anyhow::Result<()> {
    use std::io::{BufRead, Write};

    let cuda_idx = args.cuda_device.unwrap_or(0);
    let device = models::loader::select_device_at(cuda_idx)?;

    let tokenizer = Tokenizer::from_dir(&args.model_dir)?;
    println!("Tokenizer loaded.");

    println!("Loading model from '{}'...", args.model_dir);
    let batch_model = models::loader::load_batch_model(&args.model_dir, &device, 1)?;
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
    let extra_stop_ids = tokenizer.stop_token_ids();
    let mut engine = engine::Engine::new_with_stop_tokens(batch_model, config, &extra_stop_ids);

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

        let prompt = server::apply_chat_template(&tokenizer, &messages);
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
            server::start_server(start_args.models_dir, start_args.port, start_args.keep_alive, start_args.memory_budget_bytes, start_args.cuda_devices)?;
        }
        "run" => {
            let run_args = parse_run_args(&args[2..]).unwrap_or_else(|e| {
                eprintln!("Error: {e}");
                print_usage();
                std::process::exit(1);
            });
            run_interactive(&run_args)?;
        }
        "pull" => {
            let pull_args = parse_pull_args(&args[2..]).unwrap_or_else(|e| {
                eprintln!("Error: {e}");
                print_usage();
                std::process::exit(1);
            });
            let dest_name = pull_args.name.unwrap_or_else(|| {
                pull_args
                    .repo_id
                    .rsplit('/')
                    .next()
                    .unwrap_or(&pull_args.repo_id)
                    .to_string()
            });
            let token = pull_args
                .token
                .or_else(|| std::env::var("HF_TOKEN").ok())
                .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok());
            if !pull_args.models_dir.exists() {
                std::fs::create_dir_all(&pull_args.models_dir)?;
            }
            models::hub::pull(&models::hub::PullConfig {
                repo_id: pull_args.repo_id,
                dest_name,
                models_dir: pull_args.models_dir,
                token,
                force: pull_args.force,
            })?;
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
