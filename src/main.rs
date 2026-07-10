mod chat_template;
mod common;
mod engine;
mod gpu_lock;
mod models;
mod sampling;
mod scheduler;
mod server;
mod telemetry;
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
    shutdown_timeout: Duration,
    memory_budget_bytes: Option<usize>,
    cuda_devices: Vec<usize>,
    max_context_len: usize,
    kv_quant: common::kv_quant::KvQuantMode,
    qjl_quantization: bool,
    require_gpu: bool,
    max_num_seqs: Option<usize>,
    max_queued_requests: usize,
    api_key: Option<String>,
    request_timeout: Option<Duration>,
    draft_model: Option<String>,
}

struct RunArgs {
    model_dir: String,
    model_id: String,
    sampling_params: SamplingParams,
    cuda_device: Option<usize>,
    max_context_len: usize,
    kv_quant: common::kv_quant::KvQuantMode,
    qjl_quantization: bool,
    require_gpu: bool,
    /// Optional speculative-decoding draft model as (resolved_dir, id).
    draft_model: Option<(String, String)>,
}

struct RmArgs {
    model_name: String,
    models_dir: PathBuf,
    force: bool,
}

struct UpdateArgs {
    nightly: bool,
    pre: bool,
}

struct UninstallArgs {
    purge: bool,
}

fn default_models_dir() -> PathBuf {
    dirs_home().join(".oxydllm").join("models")
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

fn next_arg<'a>(args: &'a [String], i: &mut usize, flag: &str) -> Result<&'a str, String> {
    *i += 1;
    args.get(*i)
        .map(|s| s.as_str())
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_devices(s: &str) -> Result<Vec<usize>, String> {
    s.split(',')
        .map(|part| {
            part.trim().parse::<usize>().map_err(|_| {
                format!(
                    "Invalid device index: '{}'  (expected integer)",
                    part.trim()
                )
            })
        })
        .collect()
}

fn resolve_devices(explicit: Option<Vec<usize>>) -> Vec<usize> {
    if let Some(d) = explicit {
        return d;
    }
    if let Ok(env) = std::env::var("OXYDLLM_DEVICES")
        && !env.trim().is_empty()
    {
        match parse_devices(&env) {
            Ok(d) => return d,
            Err(e) => tracing::warn!(error = %e, "OXYDLLM_DEVICES ignored"),
        }
    }
    vec![]
}

fn print_usage() {
    eprintln!(
        "\
Usage: oxydllm <command> [options]

Commands:
  pull      <user/model>       Download a model from HuggingFace
  rm        <model-name>       Remove a model and its files from disk
  update    [--nightly|--pre]  Update oxydllm to the latest version
  uninstall [--purge]          Remove oxydllm from the system
  list                         List all locally available models
  start                        Start the HTTP inference server
  run       <model-name>       Interactive chat in terminal
  estimate <model>             Estimate memory footprint and accuracy

Download options (pull):
  --models-dir <DIR>           Destination directory (default: ~/.oxydllm/models/)
  --name <NAME>                Folder name override (default: model name)
  --token <TOKEN>              HuggingFace token for gated models
  --variant <FORMAT>           GGUF variant to download (e.g. Q4_K_M); skips interactive prompt
  --force                      Overwrite if model already exists

Server options (start):
  CLI flags take priority over environment variables.

  --port <PORT>              Listen port (default: 11313, env: OXYDLLM_PORT)
  --models-dir <DIR>         Models directory (default: ~/.oxydllm/models/, env: OXYDLLM_MODELS_DIR)
  --keep-alive <SECS>        Keep-alive seconds before eviction (default: 900, env: OXYDLLM_KEEP_ALIVE)
  --shutdown-timeout <SECS>  Seconds to wait for in-flight requests on shutdown (default: 30, env: OXYDLLM_SHUTDOWN_TIMEOUT)
  --memory-budget <MB>       Max total VRAM for loaded models in MB; LRU eviction when exceeded (env: OXYDLLM_MEMORY_BUDGET)
  --max-context-len <N>      Max tokens per sequence for KV cache (default: 4096, env: OXYDLLM_MAX_CONTEXT_LEN)
  --devices <IDS>            Comma-separated CUDA device indices to use (default: auto, env: OXYDLLM_DEVICES)
                             Examples: --devices 0   --devices 0,1,2
  --kv-quant <MODE>          KV cache quantization mode (default: off, env: OXYDLLM_KV_QUANT)
                             - lossless: 4-bit, quality-neutral
                             - balanced: 3-bit, near-identical quality
                             - aggressive: 2-bit, maximum compression
  --qjl-quantization         Enable Stage-2 QJL key residual quantization (default: disabled)
  --allow-cpu                Allow CPU fallback when no GPU is available (default: GPU required, env: OXYDLLM_ALLOW_CPU)
  --max-num-seqs <N>         Max concurrent sequences per model (default: auto from KV budget, env: OXYDLLM_MAX_NUM_SEQS)
  --max-queued-requests <N>  Max requests queued per model before returning 429 (default: 200, env: OXYDLLM_MAX_QUEUED_REQUESTS)
  --api-key <KEY>            Require `Authorization: Bearer <KEY>` (or `X-API-Key`) on /v1/* and /metrics (default: disabled, env: OXYDLLM_API_KEY)
  --request-timeout <SECS>   Wall-clock timeout per chat completion request; 0 disables (default: 300, env: OXYDLLM_REQUEST_TIMEOUT)
  --draft-model <NAME>       Enable greedy speculative decoding with this draft model (env: OXYDLLM_DRAFT_MODEL)
  --otel-endpoint <URL>      Export per-request traces over OTLP/HTTP to this endpoint, e.g. http://localhost:4318
                             (env: OXYDLLM_OTEL_ENDPOINT or OTEL_EXPORTER_OTLP_ENDPOINT; default: disabled)

Chat options (run):
  --models-dir <DIR>         Models directory (default: ~/.oxydllm/models/)
  --devices <ID>             CUDA device index to use (default: auto, env: OXYDLLM_DEVICES)
  --max-context-len <N>      Max tokens per sequence for KV cache (default: 4096)
  --kv-quant <MODE>          KV cache quantization: off, lossless, balanced, aggressive
  --qjl-quantization         Enable Stage-2 QJL key residual quantization (default: disabled)
  --allow-cpu                Allow CPU fallback when no GPU is available (default: GPU required, env: OXYDLLM_ALLOW_CPU)
  --draft-model <NAME>       Enable greedy speculative decoding with this draft model
  --temperature <T>          Sampling temperature (default: 0.7)
  --top-k <K>                Top-k filtering (default: 0, disabled)
  --top-p <P>                Nucleus sampling (default: 1.0)
  --min-p <P>                Min-p filtering (default: 0.0)
  --repeat-penalty <R>       Repetition penalty (default: 1.0)
  --repeat-window <N>        Trailing token window for repetition penalty (default: 0 = full history)

Remove options (rm):
  --models-dir <DIR>         Models directory (default: ~/.oxydllm/models/)
  --force / -f               Skip confirmation prompt

Update options (update):
  --nightly                  Update to the latest nightly build
  --pre / --prerelease       Include pre-release versions (alpha, beta, rc)

Uninstall options (uninstall):
  --purge                    Also remove ~/.oxydllm/ (models and all data)

Estimate options (estimate):
  --models-dir <DIR>         Models directory (default: ~/.oxydllm/models/)
  --token <TOKEN>            HuggingFace token (for private repos)
  --context-len <N>          Context length for KV cache estimate (default: 4096)
  --num-sequences <N>        Concurrent sequences for KV cache estimate (default: 1)"
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
                models_dir = Some(PathBuf::from(next_arg(args, &mut i, "--models-dir")?));
            }
            "--token" => {
                token = Some(next_arg(args, &mut i, "--token")?.to_string());
            }
            "--context-len" => {
                context_len = next_arg(args, &mut i, "--context-len")?
                    .parse()
                    .map_err(|_| "Invalid context-len (expected integer)")?;
            }
            "--num-sequences" => {
                num_sequences = next_arg(args, &mut i, "--num-sequences")?
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
                models_dir = Some(PathBuf::from(next_arg(args, &mut i, "--models-dir")?));
            }
            "--name" => {
                name = Some(next_arg(args, &mut i, "--name")?.to_string());
            }
            "--token" => {
                token = Some(next_arg(args, &mut i, "--token")?.to_string());
            }
            "--variant" => {
                variant = Some(next_arg(args, &mut i, "--variant")?.to_string());
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
                models_dir = Some(PathBuf::from(next_arg(args, &mut i, "--models-dir")?));
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

fn parse_update_args(args: &[String]) -> Result<UpdateArgs, String> {
    let mut nightly = false;
    let mut pre = false;
    for arg in args {
        match arg.as_str() {
            "--nightly" => nightly = true,
            "--pre" | "--prerelease" => pre = true,
            other => return Err(format!("Unknown option: {other}")),
        }
    }
    Ok(UpdateArgs { nightly, pre })
}

fn parse_uninstall_args(args: &[String]) -> Result<UninstallArgs, String> {
    let mut purge = false;
    for arg in args {
        match arg.as_str() {
            "--purge" => purge = true,
            other => return Err(format!("Unknown option: {other}")),
        }
    }
    Ok(UninstallArgs { purge })
}

fn run_rm(args: &RmArgs) -> anyhow::Result<()> {
    use std::io::{BufRead, Write};

    let model_path = models::loader::resolve_model_path(&args.models_dir, &args.model_name)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Model '{}' not found in '{}'",
                args.model_name,
                args.models_dir.display()
            )
        })?;

    println!("Model:     {}", args.model_name);
    println!("Directory: {}", model_path.display());

    if !args.force {
        print!(
            "Remove model '{}' and all its files? [y/N] ",
            args.model_name
        );
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

// `update`/`uninstall` are only meaningful for binaries installed via install.sh,
// which sets OXYDLLM_DIST_BUILD at build time (surfaced by build.rs as a compile
// env). option_env! avoids a custom cfg, so no lint to allow.
fn check_dist_build(cmd: &str) {
    if option_env!("OXYDLLM_DIST_BUILD").is_none() {
        eprintln!("'{cmd}' is only available for binaries installed via install.sh.");
        eprintln!("To rebuild from source: cargo build --release");
        std::process::exit(1);
    }
}

fn run_update(args: &UpdateArgs) -> anyhow::Result<()> {
    check_dist_build("update");
    if args.nightly {
        run_update_nightly()
    } else {
        run_update_stable(args.pre)
    }
}

fn github_client() -> anyhow::Result<reqwest::blocking::Client> {
    let mut headers = reqwest::header::HeaderMap::new();
    if let Ok(token) = std::env::var("GITHUB_TOKEN") {
        let value = reqwest::header::HeaderValue::from_str(&format!("Bearer {token}"))
            .map_err(|e| anyhow::anyhow!("Invalid GITHUB_TOKEN value: {e}"))?;
        headers.insert(reqwest::header::AUTHORIZATION, value);
    }
    reqwest::blocking::Client::builder()
        .user_agent("oxydllm-updater")
        .default_headers(headers)
        .build()
        .map_err(Into::into)
}

fn run_update_stable(include_pre: bool) -> anyhow::Result<()> {
    let client = github_client()?;

    let remote_tag: String = if include_pre {
        let releases: serde_json::Value = client
            .get("https://api.github.com/repos/giovannifil-64/oxydllm/releases")
            .send()?
            .error_for_status()?
            .json()?;
        releases
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("GitHub API returned unexpected format"))?
            .iter()
            .find(|r| r["tag_name"].as_str() != Some("nightly"))
            .ok_or_else(|| anyhow::anyhow!("No releases found"))?["tag_name"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("GitHub API response missing tag_name"))?
            .to_string()
    } else {
        let response = client
            .get("https://api.github.com/repos/giovannifil-64/oxydllm/releases/latest")
            .send()?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            println!("No stable release available yet.");
            println!(
                "Use --pre to include pre-release versions, or --nightly for the latest build."
            );
            return Ok(());
        }
        response.error_for_status()?.json::<serde_json::Value>()?["tag_name"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("GitHub API response missing tag_name"))?
            .to_string()
    };

    let strip_v = |s: &str| s.trim_start_matches('v').to_string();
    let local = strip_v(env!("CARGO_PKG_VERSION"));
    let remote = strip_v(&remote_tag);

    if remote == local {
        println!("Already up-to-date ({remote_tag}).");
        return Ok(());
    }

    println!(
        "Update available: {} → {remote_tag}",
        env!("CARGO_PKG_VERSION")
    );
    run_install_sh("stable", &remote_tag)
}

fn run_update_nightly() -> anyhow::Result<()> {
    let client = github_client()?;

    let response = client
        .get("https://api.github.com/repos/giovannifil-64/oxydllm/releases/tags/nightly")
        .send()?;

    if response.status() == reqwest::StatusCode::NOT_FOUND {
        println!("Nightly release temporarily unavailable — the build may be in progress.");
        println!("Try again in a few minutes.");
        return Ok(());
    }

    let release: serde_json::Value = response.error_for_status()?.json()?;

    let published_at = release["published_at"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("GitHub API response missing published_at"))?;

    let release_ts = chrono::DateTime::parse_from_rfc3339(published_at)
        .map_err(|e| anyhow::anyhow!("Failed to parse release timestamp: {e}"))?
        .timestamp() as u64;

    // option_env!, not env!: build.rs sets this, but the Docker dummy-build layer
    // doesn't always reapply the build-script env to the final compile. Optional
    // metadata (0 when absent), so don't make it a hard compile-time requirement.
    let build_ts: u64 = option_env!("OXYDLLM_BUILD_TS")
        .unwrap_or("0")
        .parse()
        .unwrap_or(0);

    if release_ts <= build_ts {
        println!("Already up-to-date (nightly).");
        return Ok(());
    }

    println!("Nightly update available.");
    run_install_sh("nightly", "nightly")
}

fn run_install_sh(channel: &str, version: &str) -> anyhow::Result<()> {
    println!("Updating to {version} ({channel})...");
    let status = std::process::Command::new("sh")
        .arg("-c")
        .arg("curl -fsSL https://raw.githubusercontent.com/giovannifil-64/oxydllm/main/install.sh | sh")
        .env("OXYDLLM_CHANNEL", channel)
        // Pin the exact resolved tag so install.sh installs it directly instead of
        // re-querying /releases/latest (which 404s when only pre-releases exist).
        .env("OXYDLLM_VERSION", version)
        .status()?;
    if !status.success() {
        anyhow::bail!("Update failed (install.sh exited with non-zero status).");
    }
    Ok(())
}

fn run_uninstall(args: &UninstallArgs) -> anyhow::Result<()> {
    check_dist_build("uninstall");
    run_uninstall_impl(args)
}

fn run_uninstall_impl(args: &UninstallArgs) -> anyhow::Result<()> {
    use std::io::{BufRead, Write};

    let home = dirs_home();
    let binary = std::env::current_exe()?;
    let data_dir = home.join(".oxydllm");

    let service_path: Option<std::path::PathBuf> = match std::env::consts::OS {
        "macos" => Some(home.join("Library/LaunchAgents/com.oxydllm.oxydllmd.plist")),
        "linux" => Some(std::path::PathBuf::from(
            "/etc/systemd/system/oxydllm.service",
        )),
        _ => None,
    };

    println!("This will remove:");
    println!("  Binary   : {}", binary.display());
    if let Some(ref sp) = service_path
        && sp.exists()
    {
        println!("  Service  : {}", sp.display());
    }
    if args.purge {
        println!(
            "  Data dir : {}  (including all downloaded models — cannot be undone)",
            data_dir.display()
        );
    } else {
        println!();
        println!("Your models in {} will NOT be removed.", data_dir.display());
        println!("Use --purge to also delete models and all data.");
    }
    println!();
    print!("Proceed? [y/N] ");
    std::io::stdout().flush()?;
    let mut answer = String::new();
    std::io::stdin().lock().read_line(&mut answer)?;
    if !matches!(answer.trim().to_lowercase().as_str(), "y" | "yes") {
        println!("Aborted.");
        return Ok(());
    }

    match std::env::consts::OS {
        "macos" => {
            if let Some(ref sp) = service_path
                && sp.exists()
            {
                let uid_out = std::process::Command::new("id").arg("-u").output()?;
                let uid = String::from_utf8(uid_out.stdout)?.trim().to_string();
                let _ = std::process::Command::new("launchctl")
                    .args(["bootout", &format!("gui/{uid}"), sp.to_str().unwrap_or("")])
                    .status();
                std::fs::remove_file(sp)?;
                println!("Removed launchd agent.");
            }
        }
        "linux" => {
            if let Some(ref sp) = service_path
                && sp.exists()
            {
                let _ = uninstall_sudo(&["systemctl", "stop", "oxydllm"]);
                let _ = uninstall_sudo(&["systemctl", "disable", "oxydllm"]);
                uninstall_sudo(&["rm", "-f", sp.to_str().unwrap_or("")])?;
                if std::path::Path::new("/etc/default/oxydllm").exists() {
                    uninstall_sudo(&["rm", "-f", "/etc/default/oxydllm"])?;
                }
                uninstall_sudo(&["systemctl", "daemon-reload"])?;
                println!("Removed systemd service.");
            }
        }
        _ => {}
    }

    remove_with_sudo_fallback(&binary)?;
    println!("Removed binary.");

    if args.purge && data_dir.exists() {
        std::fs::remove_dir_all(&data_dir)?;
        println!("Removed data directory.");
    }

    println!("\noxydllm uninstalled.");
    Ok(())
}

fn uninstall_sudo(args: &[&str]) -> anyhow::Result<()> {
    let status = std::process::Command::new("sudo").args(args).status()?;
    if !status.success() {
        anyhow::bail!("Command failed: sudo {}", args.join(" "));
    }
    Ok(())
}

fn remove_with_sudo_fallback(path: &std::path::Path) -> anyhow::Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
            let status = std::process::Command::new("sudo")
                .args(["rm", "-f", path.to_str().unwrap_or("")])
                .status()?;
            if !status.success() {
                anyhow::bail!(
                    "Failed to remove '{}': permission denied (sudo also failed)",
                    path.display()
                );
            }
            Ok(())
        }
        Err(e) => Err(e.into()),
    }
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok()
}

fn env_u16(name: &str) -> Option<u16> {
    std::env::var(name).ok()?.parse().ok()
}

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok()?.parse().ok()
}

fn parse_start_args(args: &[String]) -> Result<StartArgs, String> {
    let mut models_dir: Option<PathBuf> =
        std::env::var("OXYDLLM_MODELS_DIR").ok().map(PathBuf::from);
    let mut port: u16 = env_u16("OXYDLLM_PORT").unwrap_or(11313);
    let mut keep_alive_secs: u64 = env_u64("OXYDLLM_KEEP_ALIVE").unwrap_or(900);
    let mut shutdown_timeout_secs: u64 = env_u64("OXYDLLM_SHUTDOWN_TIMEOUT").unwrap_or(30);
    let mut memory_budget_mb: Option<usize> = env_usize("OXYDLLM_MEMORY_BUDGET");
    let mut devices_raw: Option<Vec<usize>> = None;
    let mut max_context_len: usize = env_usize("OXYDLLM_MAX_CONTEXT_LEN").unwrap_or(4096);
    let mut kv_quant = std::env::var("OXYDLLM_KV_QUANT")
        .ok()
        .map(|v| common::kv_quant::KvQuantMode::parse(&v))
        .transpose()?
        .unwrap_or(common::kv_quant::KvQuantMode::Off);
    let mut qjl_quantization = false;
    let env_allow_cpu = std::env::var("OXYDLLM_ALLOW_CPU")
        .ok()
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false);
    let mut require_gpu = !env_allow_cpu;
    let mut max_num_seqs: Option<usize> = env_usize("OXYDLLM_MAX_NUM_SEQS");
    let mut max_queued_requests: usize = env_usize("OXYDLLM_MAX_QUEUED_REQUESTS").unwrap_or(200);
    let mut draft_model: Option<String> = std::env::var("OXYDLLM_DRAFT_MODEL").ok();
    let mut api_key: Option<String> = std::env::var("OXYDLLM_API_KEY")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    // Default: 300 seconds; 0 disables the timeout.
    let mut request_timeout_secs: u64 = env_u64("OXYDLLM_REQUEST_TIMEOUT").unwrap_or(300);
    let mut i = 0;

    while i < args.len() {
        match args[i].as_str() {
            "--port" => {
                port = next_arg(args, &mut i, "--port")?
                    .parse()
                    .map_err(|_| "Invalid port number")?;
            }
            "--models-dir" => {
                models_dir = Some(PathBuf::from(next_arg(args, &mut i, "--models-dir")?));
            }
            "--keep-alive" => {
                keep_alive_secs = next_arg(args, &mut i, "--keep-alive")?
                    .parse()
                    .map_err(|_| "Invalid keep-alive value")?;
            }
            "--memory-budget" => {
                let mb: usize = next_arg(args, &mut i, "--memory-budget")?
                    .parse()
                    .map_err(|_| "Invalid memory-budget value (expected MB integer)")?;
                memory_budget_mb = Some(mb);
            }
            "--devices" => {
                devices_raw = Some(parse_devices(next_arg(args, &mut i, "--devices")?)?);
            }
            "--max-context-len" => {
                max_context_len = next_arg(args, &mut i, "--max-context-len")?
                    .parse()
                    .map_err(|_| "Invalid max-context-len value (expected integer)")?;
            }
            "--kv-quant" => {
                kv_quant =
                    common::kv_quant::KvQuantMode::parse(next_arg(args, &mut i, "--kv-quant")?)?;
            }
            "--shutdown-timeout" => {
                shutdown_timeout_secs = next_arg(args, &mut i, "--shutdown-timeout")?
                    .parse()
                    .map_err(|_| "Invalid shutdown-timeout value (expected integer seconds)")?;
            }
            "--qjl-quantization" => {
                qjl_quantization = true;
            }
            "--allow-cpu" => {
                require_gpu = false;
            }
            "--max-num-seqs" => {
                let n: usize = next_arg(args, &mut i, "--max-num-seqs")?
                    .parse()
                    .map_err(|_| "Invalid max-num-seqs value (expected positive integer)")?;
                if n == 0 {
                    return Err("--max-num-seqs must be at least 1".to_string());
                }
                max_num_seqs = Some(n);
            }
            "--max-queued-requests" => {
                max_queued_requests = next_arg(args, &mut i, "--max-queued-requests")?
                    .parse()
                    .map_err(|_| "Invalid max-queued-requests value (expected positive integer)")?;
                if max_queued_requests == 0 {
                    return Err("--max-queued-requests must be at least 1".to_string());
                }
            }
            "--api-key" => {
                let key = next_arg(args, &mut i, "--api-key")?.trim();
                if key.is_empty() {
                    return Err("--api-key must not be empty".to_string());
                }
                api_key = Some(key.to_string());
            }
            "--request-timeout" => {
                request_timeout_secs = next_arg(args, &mut i, "--request-timeout")?
                    .parse()
                    .map_err(
                        |_| "Invalid request-timeout value (expected non-negative integer seconds)",
                    )?;
            }
            "--draft-model" => {
                draft_model = Some(next_arg(args, &mut i, "--draft-model")?.to_string());
            }
            "--otel-endpoint" => {
                // Consumed at subscriber-init time (before arg parsing); accept it
                // here so it is not rejected as an unknown option.
                let _ = next_arg(args, &mut i, "--otel-endpoint")?;
            }
            other => return Err(format!("Unknown option: {}", other)),
        }
        i += 1;
    }

    let request_timeout = if request_timeout_secs == 0 {
        None
    } else {
        Some(Duration::from_secs(request_timeout_secs))
    };

    Ok(StartArgs {
        models_dir: models_dir.unwrap_or_else(default_models_dir),
        port,
        keep_alive: Duration::from_secs(keep_alive_secs),
        shutdown_timeout: Duration::from_secs(shutdown_timeout_secs),
        memory_budget_bytes: memory_budget_mb.map(|mb| mb * 1024 * 1024),
        cuda_devices: resolve_devices(devices_raw),
        max_context_len,
        kv_quant,
        qjl_quantization,
        require_gpu,
        max_num_seqs,
        max_queued_requests,
        api_key,
        request_timeout,
        draft_model,
    })
}

fn parse_run_args(args: &[String]) -> Result<RunArgs, String> {
    let mut model_name = String::new();
    let mut draft_name: Option<String> = None;
    let mut models_dir: Option<PathBuf> = None;
    let mut devices_raw: Option<Vec<usize>> = None;
    let mut max_context_len: usize = 4096;
    let mut kv_quant = common::kv_quant::KvQuantMode::Off;
    let mut qjl_quantization = false;
    let env_allow_cpu = std::env::var("OXYDLLM_ALLOW_CPU")
        .ok()
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false);
    let mut require_gpu = !env_allow_cpu;
    let mut params = SamplingParams {
        temperature: 0.7,
        ..SamplingParams::default()
    };
    let mut i = 0;

    while i < args.len() {
        match args[i].as_str() {
            "--models-dir" => {
                models_dir = Some(PathBuf::from(next_arg(args, &mut i, "--models-dir")?));
            }
            "--devices" => {
                let raw = next_arg(args, &mut i, "--devices")?;
                let d = parse_devices(raw)?;
                devices_raw = Some(d);
            }
            "--max-context-len" => {
                max_context_len = next_arg(args, &mut i, "--max-context-len")?
                    .parse()
                    .map_err(|_| "Invalid max-context-len value")?;
            }
            "--temperature" | "-t" => {
                params.temperature = next_arg(args, &mut i, "--temperature")?
                    .parse()
                    .map_err(|_| "Invalid temperature")?;
            }
            "--top-k" => {
                params.top_k = next_arg(args, &mut i, "--top-k")?
                    .parse()
                    .map_err(|_| "Invalid top-k")?;
            }
            "--top-p" => {
                params.top_p = next_arg(args, &mut i, "--top-p")?
                    .parse()
                    .map_err(|_| "Invalid top-p")?;
            }
            "--min-p" => {
                params.min_p = next_arg(args, &mut i, "--min-p")?
                    .parse()
                    .map_err(|_| "Invalid min-p")?;
            }
            "--repeat-penalty" => {
                params.repetition_penalty = next_arg(args, &mut i, "--repeat-penalty")?
                    .parse()
                    .map_err(|_| "Invalid repeat-penalty")?;
            }
            "--repeat-window" => {
                params.repetition_window = next_arg(args, &mut i, "--repeat-window")?
                    .parse()
                    .map_err(|_| "Invalid repeat-window")?;
            }
            "--kv-quant" => {
                kv_quant =
                    common::kv_quant::KvQuantMode::parse(next_arg(args, &mut i, "--kv-quant")?)?;
            }
            "--qjl-quantization" => {
                qjl_quantization = true;
            }
            "--draft-model" => {
                draft_name = Some(next_arg(args, &mut i, "--draft-model")?.to_string());
            }
            "--allow-cpu" => {
                require_gpu = false;
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

    let base = models_dir.unwrap_or_else(default_models_dir);
    let resolve = |name: &str| -> String {
        if std::path::Path::new(name).is_absolute() {
            name.to_string()
        } else {
            models::loader::resolve_model_path(&base, name)
                .unwrap_or_else(|| base.join(name))
                .to_string_lossy()
                .to_string()
        }
    };
    let model_dir = resolve(&model_name);
    let draft_model = draft_name.map(|n| (resolve(&n), n));

    Ok(RunArgs {
        model_dir,
        model_id: model_name,
        sampling_params: params,
        cuda_device: resolve_devices(devices_raw).into_iter().next(),
        max_context_len,
        kv_quant,
        qjl_quantization,
        require_gpu,
        draft_model,
    })
}

fn run_list(models_dir: &std::path::Path) {
    let mut models = models::loader::discover_models(models_dir);
    models.sort_by(|a, b| a.id.to_lowercase().cmp(&b.id.to_lowercase()));

    if models.is_empty() {
        println!("No models found in {}", models_dir.display());
        return;
    }

    let name_w = models
        .iter()
        .map(|m| m.id.chars().count())
        .chain(std::iter::once("NAME".len()))
        .max()
        .unwrap_or(4);
    let arch_w = models
        .iter()
        .map(|m| m.architecture.chars().count())
        .chain(std::iter::once("ARCHITECTURE".len()))
        .max()
        .unwrap_or(12);

    println!();
    println!(
        "  {:<name_w$} {:<arch_w$} {:>9}",
        "NAME", "ARCHITECTURE", "SIZE"
    );
    println!("  {}", "─".repeat(name_w + 1 + arch_w + 1 + 9));

    for m in &models {
        let size_str = if m.size_bytes > 0 {
            fmt_size(m.size_bytes)
        } else {
            "-".to_string()
        };
        println!(
            "  {:<name_w$} {:<arch_w$} {:>9}",
            m.id, m.architecture, size_str,
        );
    }
    println!();
    println!("  {} model(s) in {}", models.len(), models_dir.display());
    println!();
}

fn fmt_size(bytes: usize) -> String {
    const GB: usize = 1024 * 1024 * 1024;
    const MB: usize = 1024 * 1024;
    const KB: usize = 1024;
    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.0} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

fn clamp_to_char_boundary(s: &str, idx: usize) -> usize {
    let mut i = idx.min(s.len());
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn find_first_stop_marker(text: &str, stop_markers: &[String]) -> Option<usize> {
    stop_markers
        .iter()
        .filter_map(|marker| {
            if marker.is_empty() {
                None
            } else {
                text.find(marker)
            }
        })
        .min()
}

fn run_interactive(args: &RunArgs) -> anyhow::Result<()> {
    use std::io::{BufRead, Write};

    let cuda_idx = args.cuda_device.unwrap_or(0);
    let device = models::loader::select_device_at(cuda_idx, args.require_gpu)?;

    let tokenizer = Tokenizer::from_dir(&args.model_dir)?;
    println!("Tokenizer loaded.");

    println!("Loading model from '{}'...", args.model_dir);
    let is_cpu = matches!(device, candle_core::Device::Cpu);
    let kv_budget = std::sync::Arc::new(crate::common::paged::GlobalKvBudget::new(
        crate::common::paged::detect_system_kv_budget(None, is_cpu),
    ));
    let (batch_model, weights_size_bytes) = models::loader::load_batch_model(
        &args.model_dir,
        &args.model_id,
        &device,
        models::loader::LoadBatchOptions {
            max_context_len: args.max_context_len,
            max_num_sequences: 1,
            kv_budget: &kv_budget,
            kv_quant: args.kv_quant,
            qjl_quantization: args.qjl_quantization,
        },
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

    let draft_model = if let Some((draft_dir, draft_id)) = &args.draft_model {
        println!("Loading draft model from '{draft_dir}'...");
        let (draft, _) = models::loader::load_batch_model(
            draft_dir,
            draft_id,
            &device,
            models::loader::LoadBatchOptions {
                max_context_len: args.max_context_len,
                max_num_sequences: 1,
                kv_budget: &kv_budget,
                kv_quant: args.kv_quant,
                qjl_quantization: args.qjl_quantization,
            },
        )?;
        if draft.vocab_size() != batch_model.vocab_size() {
            anyhow::bail!(
                "draft vocab_size {} != target vocab_size {} — draft and target must share a tokenizer",
                draft.vocab_size(),
                batch_model.vocab_size()
            );
        }
        println!("Draft model loaded — speculative decoding enabled.");
        Some(draft)
    } else {
        None
    };

    let config = scheduler::SchedulerConfig {
        max_num_sequences: 1,
        max_tokens_per_step: 4096,
    };
    let extra_stop_ids = tokenizer.stop_token_ids();
    let extra_stop_sequences = tokenizer.stop_token_sequences();
    let mut engine = engine::Engine::new_with_stop_controls(
        batch_model,
        config,
        &extra_stop_ids,
        &extra_stop_sequences,
    );
    if let Some(draft) = draft_model {
        engine = engine.with_draft_model(draft);
    }

    let mut messages: Vec<ChatMessage> = vec![ChatMessage {
        role: "system".to_string(),
        content: Some("You are a helpful assistant.".to_string()),
        reasoning_content: None,
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }];
    let stop_markers = tokenizer.stop_text_markers();
    let hold_back_bytes = stop_markers
        .iter()
        .map(|m| m.len())
        .max()
        .unwrap_or(0)
        .saturating_sub(1);

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
            content: Some(input),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        });

        let mut prompt = server::apply_chat_template(&tokenizer, &messages, false, None, None)
            .unwrap_or_else(|_| crate::chat_template::format_plain_chat(&messages));
        let mut prompt_tokens = tokenizer.encode(&prompt)?;

        while prompt_tokens.len() >= max_seq_len && messages.len() > 2 {
            let prev_len = prompt_tokens.len();
            messages.remove(1);
            if messages.len() > 1 && messages[1].role == "assistant" {
                messages.remove(1);
            }
            prompt = server::apply_chat_template(&tokenizer, &messages, false, None, None)
                .unwrap_or_else(|_| crate::chat_template::format_plain_chat(&messages));
            prompt_tokens = tokenizer.encode(&prompt)?;
            tracing::warn!(
                previous_tokens = prev_len,
                current_tokens = prompt_tokens.len(),
                max_context_len = max_seq_len,
                "truncated oldest messages to fit context window"
            );
        }

        if prompt_tokens.len() >= max_seq_len {
            tracing::warn!(
                prompt_tokens = prompt_tokens.len(),
                max_context_len = max_seq_len,
                "context full, cannot generate; start a new conversation with /exit"
            );
            messages.pop();
            continue;
        }

        let max_tokens = max_seq_len - prompt_tokens.len();
        engine.add_request(prompt_tokens, args.sampling_params.clone(), max_tokens);

        let mut response_text = String::new();
        let mut output_ids: Vec<u32> = Vec::new();
        let mut decoded_len: usize = 0;
        let mut stream_buffer = String::new();
        let mut hit_text_stop = false;
        while engine.has_pending_work() && !hit_text_stop {
            let step = engine.step().map_err(|e| anyhow::anyhow!("{}", e))?;
            for tok in &step.new_tokens {
                output_ids.push(tok.token);
                let full = tokenizer.decode(&output_ids)?;
                let start = clamp_to_char_boundary(&full, decoded_len);
                let new_text = &full[start..];
                let trimmed = new_text.trim_end_matches('\u{FFFD}');
                if trimmed.is_empty() {
                    decoded_len = start;
                    continue;
                }
                decoded_len = start + trimmed.len();
                let emit = trimmed.to_string();

                stream_buffer.push_str(&emit);

                if let Some(stop_idx) = find_first_stop_marker(&stream_buffer, &stop_markers) {
                    let safe_idx = clamp_to_char_boundary(&stream_buffer, stop_idx);
                    if safe_idx > 0 {
                        let safe = &stream_buffer[..safe_idx];
                        print!("{}", safe);
                        std::io::stdout().flush()?;
                        response_text.push_str(safe);
                    }
                    hit_text_stop = true;
                    break;
                }

                if hold_back_bytes == 0 {
                    if !stream_buffer.is_empty() {
                        print!("{}", stream_buffer);
                        std::io::stdout().flush()?;
                        response_text.push_str(&stream_buffer);
                        stream_buffer.clear();
                    }
                } else if stream_buffer.len() > hold_back_bytes {
                    let flush_idx = clamp_to_char_boundary(
                        &stream_buffer,
                        stream_buffer.len() - hold_back_bytes,
                    );
                    if flush_idx > 0 {
                        let safe = &stream_buffer[..flush_idx];
                        print!("{}", safe);
                        std::io::stdout().flush()?;
                        response_text.push_str(safe);
                        stream_buffer = stream_buffer[flush_idx..].to_string();
                    }
                }
            }
        }

        if hit_text_stop {
            let _ = engine.abort_all();
        }

        if !hit_text_stop && !output_ids.is_empty() {
            let full = tokenizer.decode(&output_ids)?;
            if decoded_len < full.len() {
                let start = clamp_to_char_boundary(&full, decoded_len);
                let rest = &full[start..];
                stream_buffer.push_str(rest);
            }
        }

        if !hit_text_stop && !stream_buffer.is_empty() {
            print!("{}", stream_buffer);
            response_text.push_str(&stream_buffer);
        }
        println!("\n");

        messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: Some(response_text),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        });
    }

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // OTLP trace export is only meaningful for the long-running server; resolve
    // the endpoint before installing the subscriber so the OpenTelemetry layer
    // can be attached at init time (it cannot be added afterwards).
    let otel_endpoint = if args.get(1).map(String::as_str) == Some("start") {
        telemetry::resolve_endpoint(&args[2..])
    } else {
        None
    };
    let mut otel_provider = telemetry::init(otel_endpoint.as_deref());

    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    match args[1].as_str() {
        "list" => {
            let models_dir = args[2..]
                .windows(2)
                .find(|w| w[0] == "--models-dir")
                .map(|w| std::path::PathBuf::from(&w[1]))
                .unwrap_or_else(default_models_dir);
            run_list(&models_dir);
        }
        "start" => {
            let start_args = parse_start_args(&args[2..]).unwrap_or_else(|e| {
                eprintln!("Error: {e}");
                print_usage();
                std::process::exit(1);
            });
            let result = server::start_server(server::StartServerArgs {
                models_dir: start_args.models_dir,
                port: start_args.port,
                keep_alive: start_args.keep_alive,
                shutdown_timeout: start_args.shutdown_timeout,
                memory_budget_bytes: start_args.memory_budget_bytes,
                cuda_devices: start_args.cuda_devices,
                max_context_len: start_args.max_context_len,
                kv_quant: start_args.kv_quant,
                qjl_quantization: start_args.qjl_quantization,
                require_gpu: start_args.require_gpu,
                max_num_seqs: start_args.max_num_seqs,
                max_queued_requests: start_args.max_queued_requests,
                api_key: start_args.api_key,
                request_timeout: start_args.request_timeout,
                draft_model: start_args.draft_model,
            });
            // Flush buffered spans before exit, on both clean shutdown and error.
            if let Some(provider) = otel_provider.take() {
                telemetry::shutdown(provider);
            }
            result?
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
            let dest_name = pull_args.name.unwrap_or_else(|| pull_args.repo_id.clone());
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
        "update" => {
            let update_args = parse_update_args(&args[2..]).unwrap_or_else(|e| {
                eprintln!("Error: {e}");
                print_usage();
                std::process::exit(1);
            });
            run_update(&update_args)?;
        }
        "uninstall" => {
            let uninstall_args = parse_uninstall_args(&args[2..]).unwrap_or_else(|e| {
                eprintln!("Error: {e}");
                print_usage();
                std::process::exit(1);
            });
            run_uninstall(&uninstall_args)?;
        }
        "--help" | "-h" | "help" => {
            print_usage();
        }
        "--version" | "-v" => {
            println!("oxydllm {}", env!("CARGO_PKG_VERSION"));
        }
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            print_usage();
            std::process::exit(1);
        }
    }

    Ok(())
}
