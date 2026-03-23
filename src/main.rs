mod chat_template;
mod common;
mod engine;
mod gpu_lock;
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

struct EstimateArgs {
    model: String,
    models_dir: PathBuf,
    token: Option<String>,
    context_len: usize,
    num_sequences: usize,
}

struct PullArgs {
    repo_id: String,
    models_dir: PathBuf,
    name: Option<String>,
    token: Option<String>,
    force: bool,
    variant: Option<String>,
}

struct StartArgs {
    models_dir: PathBuf,
    port: u16,
    keep_alive: Duration,
    memory_budget_bytes: Option<usize>,
    cuda_devices: Vec<usize>,
    max_context_len: usize,
}

struct RunArgs {
    model_dir: String,
    model_id: String,
    sampling_params: SamplingParams,
    cuda_device: Option<usize>,
    max_context_len: usize,
}

struct RmArgs {
    model_name: String,
    models_dir: PathBuf,
    force: bool,
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
    if let Ok(env) = std::env::var("RLLM_DEVICES")
        && !env.trim().is_empty() {
            match parse_devices(&env) {
                Ok(d) => return d,
                Err(e) => eprintln!("Warning: RLLM_DEVICES ignored — {}", e),
            }
        }
    vec![]
}

fn print_usage() {
    eprintln!(
        "\
Usage: rllm <command> [options]

Commands:
  pull     <user/model>   Download a model from HuggingFace
  rm       <model-name>   Remove a model and its files from disk
  start                   Start the HTTP inference server
  run      <model-name>   Interactive chat in terminal
  estimate <model>        Estimate memory footprint and accuracy

Download options (pull):
  --models-dir <DIR>        Destination directory (default: ~/.rllm/models/)
  --name <NAME>             Folder name override (default: model name)
  --token <TOKEN>           HuggingFace token for gated models
  --variant <FORMAT>        GGUF variant to download (e.g. Q4_K_M); skips interactive prompt
  --force                   Overwrite if model already exists

Server options (start):
  --port <PORT>             Listen port (default: 11313)
  --models-dir <DIR>        Models directory (default: ~/.rllm/models/)
  --keep-alive <SECS>       Keep-alive seconds before eviction (default: 900)
  --memory-budget <MB>      Max total VRAM for loaded models in MB; LRU eviction when exceeded
  --max-context-len <N>     Max tokens per sequence for KV cache (default: 4096)
  --devices <IDS>           Comma-separated CUDA device indices to use (default: auto, env: RLLM_DEVICES)
                            Examples: --devices 0   --devices 0,1,2

Chat options (run):
  --models-dir <DIR>        Models directory (default: ~/.rllm/models/)
  --devices <ID>            CUDA device index to use (default: auto, env: RLLM_DEVICES)
  --max-context-len <N>     Max tokens per sequence for KV cache (default: 4096)
  --temperature <T>         Sampling temperature (default: 0.7)
  --top-k <K>               Top-k filtering (default: 0, disabled)
  --top-p <P>               Nucleus sampling (default: 1.0)
  --min-p <P>               Min-p filtering (default: 0.0)
  --repeat-penalty <R>      Repetition penalty (default: 1.0)

Remove options (rm):
  --models-dir <DIR>        Models directory (default: ~/.rllm/models/)
  --force / -f              Skip confirmation prompt

Estimate options (estimate):
  --models-dir <DIR>        Models directory (default: ~/.rllm/models/)
  --token <TOKEN>           HuggingFace token (for private repos)
  --context-len <N>         Context length for KV cache estimate (default: 4096)
  --num-sequences <N>       Concurrent sequences for KV cache estimate (default: 1)"
    );
}

fn parse_estimate_args(args: &[String]) -> Result<EstimateArgs, String> {
    let mut model = String::new();
    let mut models_dir: Option<PathBuf> = None;
    let mut token: Option<String> = None;
    let mut context_len: usize = 4096;
    let mut num_sequences: usize = 1;
    let mut i = 0;

    while i < args.len() {
        match args[i].as_str() {
            "--models-dir" => {
                i += 1;
                models_dir = Some(PathBuf::from(
                    args.get(i).ok_or("--models-dir requires a value")?,
                ));
            }
            "--token" => {
                i += 1;
                token = Some(args.get(i).ok_or("--token requires a value")?.clone());
            }
            "--context-len" => {
                i += 1;
                context_len = args
                    .get(i)
                    .ok_or("--context-len requires a value")?
                    .parse()
                    .map_err(|_| "Invalid context-len (expected integer)")?;
            }
            "--num-sequences" => {
                i += 1;
                num_sequences = args
                    .get(i)
                    .ok_or("--num-sequences requires a value")?
                    .parse()
                    .map_err(|_| "Invalid num-sequences (expected integer)")?;
            }
            _ if !args[i].starts_with('-') && model.is_empty() => {
                model = args[i].clone();
            }
            other => return Err(format!("Unknown option: {}", other)),
        }
        i += 1;
    }

    if model.is_empty() {
        return Err(
            "Missing <model>: provide a local model name or a HF repo ID (user/model)".to_string(),
        );
    }

    Ok(EstimateArgs {
        model,
        models_dir: models_dir.unwrap_or_else(default_models_dir),
        token,
        context_len,
        num_sequences,
    })
}

fn parse_pull_args(args: &[String]) -> Result<PullArgs, String> {
    let mut repo_id = String::new();
    let mut models_dir: Option<PathBuf> = None;
    let mut name: Option<String> = None;
    let mut token: Option<String> = None;
    let mut force = false;
    let mut variant: Option<String> = None;
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
            "--variant" => {
                i += 1;
                variant = Some(args.get(i).ok_or("--variant requires a value")?.clone());
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
        variant,
    })
}

fn parse_rm_args(args: &[String]) -> Result<RmArgs, String> {
    let mut model_name = String::new();
    let mut models_dir: Option<PathBuf> = None;
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
            "--force" | "-f" => {
                force = true;
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

    Ok(RmArgs {
        model_name,
        models_dir: models_dir.unwrap_or_else(default_models_dir),
        force,
    })
}

fn run_rm(args: &RmArgs) -> anyhow::Result<()> {
    use std::io::{Write, BufRead};

    let model_path = models::loader::resolve_model_path(&args.models_dir, &args.model_name)
        .ok_or_else(|| anyhow::anyhow!(
            "Model '{}' not found in '{}'",
            args.model_name,
            args.models_dir.display()
        ))?;

    println!("Model:     {}", args.model_name);
    println!("Directory: {}", model_path.display());

    if !args.force {
        print!("Remove model '{}' and all its files? [y/N] ", args.model_name);
        std::io::stdout().flush()?;
        let mut answer = String::new();
        std::io::stdin().lock().read_line(&mut answer)?;
        if !matches!(answer.trim().to_lowercase().as_str(), "y" | "yes") {
            println!("Aborted.");
            return Ok(());
        }
    }

    std::fs::remove_dir_all(&model_path)?;
    println!("Removed '{}'.", model_path.display());

    // Clean the registry entry if present.
    let registry_path = models::manager::registry_path(&args.models_dir);
    if registry_path.exists() {
        let mut registry = models::manager::load_registry(&args.models_dir);
        if registry.remove(&args.model_name).is_some() {
            models::manager::save_registry(&args.models_dir, &registry);
            println!("Removed '{}' from registry.", args.model_name);
        }
    }

    Ok(())
}

fn parse_start_args(args: &[String]) -> Result<StartArgs, String> {
    let mut models_dir: Option<PathBuf> = None;
    let mut port: u16 = 11313;
    let mut keep_alive_secs: u64 = 900;
    let mut memory_budget_mb: Option<usize> = None;
    let mut devices_raw: Option<Vec<usize>> = None;
    let mut max_context_len: usize = 4096;
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
                    parse_devices(args.get(i).ok_or("--devices requires a value")?)?,
                );
            }
            "--max-context-len" => {
                i += 1;
                max_context_len = args
                    .get(i)
                    .ok_or("--max-context-len requires a value")?
                    .parse()
                    .map_err(|_| "Invalid max-context-len value (expected integer)")?;
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
        max_context_len,
    })
}

fn parse_run_args(args: &[String]) -> Result<RunArgs, String> {
    let mut model_name = String::new();
    let mut models_dir: Option<PathBuf> = None;
    let mut devices_raw: Option<Vec<usize>> = None;
    let mut max_context_len: usize = 4096;
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
            "--max-context-len" => {
                i += 1;
                max_context_len = args
                    .get(i)
                    .ok_or("--max-context-len requires a value")?
                    .parse()
                    .map_err(|_| "Invalid max-context-len value")?;
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
        model_name.clone()
    } else {
        let base = models_dir.unwrap_or_else(default_models_dir);
        models::loader::resolve_model_path(&base, &model_name)
            .unwrap_or_else(|| base.join(&model_name))
            .to_string_lossy()
            .to_string()
    };

    Ok(RunArgs {
        model_dir,
        model_id: model_name,
        sampling_params: params,
        cuda_device: resolve_devices(devices_raw).into_iter().next(),
        max_context_len,
    })
}

fn run_interactive(args: &RunArgs) -> anyhow::Result<()> {
    use std::io::{BufRead, Write};

    let cuda_idx = args.cuda_device.unwrap_or(0);
    let device = models::loader::select_device_at(cuda_idx)?;

    let tokenizer = Tokenizer::from_dir(&args.model_dir)?;
    println!("Tokenizer loaded.");

    println!("Loading model from '{}'...", args.model_dir);
    let is_cpu = matches!(device, candle_core::Device::Cpu);
    let kv_budget = std::sync::Arc::new(
        crate::common::paged::GlobalKvBudget::new(
            crate::common::paged::detect_system_kv_budget(None, is_cpu),
        )
    );
    let (batch_model, weights_size_bytes) = models::loader::load_batch_model(
        &args.model_dir, &args.model_id, &device, args.max_context_len, 1, &kv_budget,
    )?;
    let max_seq_len = batch_model.max_seq_len();
    let kv_cache_bytes = batch_model.kv_cache_bytes();
    let total_bytes = weights_size_bytes + kv_cache_bytes;
    println!(
        "Model loaded. vocab_size={}, max_seq_len={}, size={:.2} GB (weights) + {:.2} GB (KV cache) = {:.2} GB total",
        batch_model.vocab_size(),
        max_seq_len,
        weights_size_bytes as f64 / 1_073_741_824.0,
        kv_cache_bytes as f64 / 1_073_741_824.0,
        total_bytes as f64 / 1_073_741_824.0,
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
        reasoning_content: None,
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
            reasoning_content: None,
        });

        let mut prompt = server::apply_chat_template(&tokenizer, &messages, false);
        let mut prompt_tokens = tokenizer.encode(&prompt)?;

        while prompt_tokens.len() >= max_seq_len && messages.len() > 2 {
            let prev_len = prompt_tokens.len();
            messages.remove(1);
            if messages.len() > 1 && messages[1].role == "assistant" {
                messages.remove(1);
            }
            prompt = server::apply_chat_template(&tokenizer, &messages, false);
            prompt_tokens = tokenizer.encode(&prompt)?;
            eprintln!(
                "[context] Truncated oldest messages ({} → {} tokens, max {})",
                prev_len,
                prompt_tokens.len(),
                max_seq_len,
            );
        }

        if prompt_tokens.len() >= max_seq_len {
            eprintln!(
                "[warning] Context full ({} tokens >= {} max) — cannot generate. \
                 Start a new conversation with /exit.",
                prompt_tokens.len(),
                max_seq_len,
            );
            messages.pop();
            continue;
        }

        let max_tokens = max_seq_len - prompt_tokens.len();
        engine.add_request(prompt_tokens, args.sampling_params.clone(), max_tokens);

        let mut response_text = String::new();
        let mut output_ids: Vec<u32> = Vec::new();
        let mut decoded_len: usize = 0;
        while engine.has_pending_work() {
            let step = engine.step().map_err(|e| anyhow::anyhow!("{}", e))?;
            for tok in &step.new_tokens {
                output_ids.push(tok.token);
                let full = tokenizer.decode(&output_ids)?;
                let new_text = &full[decoded_len..];
                if !new_text.is_empty() {
                    print!("{}", new_text);
                    std::io::stdout().flush()?;
                    response_text.push_str(new_text);
                    decoded_len = full.len();
                }
            }
        }
        println!("\n");

        messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: response_text,
            reasoning_content: None,
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
            server::start_server(start_args.models_dir, start_args.port, start_args.keep_alive, start_args.memory_budget_bytes, start_args.cuda_devices, start_args.max_context_len)?;
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
                variant: pull_args.variant,
            })?;
        }
        "estimate" => {
            let est_args = parse_estimate_args(&args[2..]).unwrap_or_else(|e| {
                eprintln!("Error: {e}");
                print_usage();
                std::process::exit(1);
            });
            let token = est_args
                .token
                .or_else(|| std::env::var("HF_TOKEN").ok())
                .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok());
            models::estimate::run_estimate(&models::estimate::EstimateArgs {
                model: est_args.model,
                models_dir: est_args.models_dir,
                token,
                context_len: est_args.context_len,
                num_sequences: est_args.num_sequences,
            })?;
        }
        "rm" | "remove" => {
            let rm_args = parse_rm_args(&args[2..]).unwrap_or_else(|e| {
                eprintln!("Error: {e}");
                print_usage();
                std::process::exit(1);
            });
            run_rm(&rm_args)?;
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
