use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const HF_ENDPOINT: &str = "https://huggingface.co";

pub struct PullConfig {
    pub repo_id: String,
    pub dest_name: String,
    pub models_dir: PathBuf,
    pub token: Option<String>,
    pub force: bool,
}

pub fn pull(config: &PullConfig) -> anyhow::Result<()> {
    let dest = config.models_dir.join(&config.dest_name);

    if dest.exists() {
        if config.force {
            println!("Removing existing model at {}...", dest.display());
            std::fs::remove_dir_all(&dest)?;
        } else {
            anyhow::bail!(
                "A model named '{}' already exists at {}.\n\
                 Use --force to overwrite, or --name <name> to save under a different name.",
                config.dest_name,
                dest.display()
            );
        }
    }

    println!("Repository : {}", config.repo_id);
    println!("Destination: {}", dest.display());
    if config.token.is_some() {
        println!("Auth       : token provided");
    }
    println!();

    let client = reqwest::blocking::Client::builder()
        .timeout(None)
        .user_agent(concat!("rllm/", env!("CARGO_PKG_VERSION")))
        .build()?;

    print!("Fetching file list...");
    std::io::stdout().flush().ok();
    let all_files = list_repo_files(&client, &config.repo_id, config.token.as_deref())?;
    let to_download = filter_model_files(&all_files);
    println!(" {} file(s)\n", to_download.len());

    if to_download.is_empty() {
        anyhow::bail!(
            "No compatible model files found in '{}'.\n\
             The repository may not contain safetensors weights.",
            config.repo_id
        );
    }

    std::fs::create_dir_all(&dest)?;

    let mut download_ok = true;
    for filename in &to_download {
        if let Err(e) =
            download_file(&client, &config.repo_id, filename, &dest, config.token.as_deref())
        {
            eprintln!("\nError downloading '{}': {}", filename, e);
            download_ok = false;
            break;
        }
    }

    if !download_ok {
        eprintln!("Cleaning up partial download...");
        let _ = std::fs::remove_dir_all(&dest);
        anyhow::bail!("Download incomplete.");
    }

    println!();
    println!("Model '{}' saved to {}", config.dest_name, dest.display());
    println!();
    println!("To use it:");
    println!(
        "  rllm run  {} --models-dir {}",
        config.dest_name,
        config.models_dir.display()
    );
    println!(
        "  rllm start --models-dir {}   (then model: \"{}\" in API)",
        config.models_dir.display(),
        config.dest_name
    );
    Ok(())
}

fn list_repo_files(
    client: &reqwest::blocking::Client,
    repo_id: &str,
    token: Option<&str>,
) -> anyhow::Result<Vec<String>> {
    let url = format!("{}/api/models/{}", HF_ENDPOINT, repo_id);
    let mut builder = client.get(&url);
    if let Some(tok) = token {
        builder = builder.bearer_auth(tok);
    }
    let resp = builder.send()?;
    let status = resp.status().as_u16();
    check_status(status, repo_id)?;

    let json: serde_json::Value = resp.json()?;
    let siblings = json["siblings"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("Unexpected HuggingFace API response (missing 'siblings')"))?;

    Ok(siblings
        .iter()
        .filter_map(|s| s["rfilename"].as_str().map(str::to_string))
        .collect())
}

fn filter_model_files(files: &[String]) -> Vec<String> {
    let mut result: Vec<String> = files
        .iter()
        .filter(|f| {
            if f.contains('/') || f.starts_with('.') {
                return false;
            }
            let l = f.to_lowercase();
            l.ends_with(".json")
                || l.ends_with(".safetensors")
                || l.ends_with(".model")
                || l.ends_with(".tiktoken")
        })
        .cloned()
        .collect();

    result.sort_by_key(|f| if f.ends_with(".json") { 0u8 } else { 1u8 });
    result
}

fn download_file(
    client: &reqwest::blocking::Client,
    repo_id: &str,
    filename: &str,
    dest: &Path,
    token: Option<&str>,
) -> anyhow::Result<()> {
    let url = format!("{}/{}/resolve/main/{}", HF_ENDPOINT, repo_id, filename);
    let mut builder = client.get(&url);
    if let Some(tok) = token {
        builder = builder.bearer_auth(tok);
    }
    let resp = builder.send()?;
    let status = resp.status().as_u16();
    check_status(status, repo_id)?;

    let total_bytes = resp.content_length();
    let dest_path = dest.join(filename);
    let mut out = std::fs::File::create(&dest_path)?;

    let label = truncate_label(filename, 40);
    let mut downloaded: u64 = 0;
    let mut last_tick = Instant::now();
    let mut body = resp;
    let mut buf = vec![0u8; 64 * 1024];

    loop {
        let n = body.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        downloaded += n as u64;

        if last_tick.elapsed().as_millis() >= 100 {
            print_progress(&label, downloaded, total_bytes, false);
            last_tick = Instant::now();
        }
    }

    print_progress(&label, downloaded, total_bytes, true);
    println!();
    Ok(())
}

fn print_progress(label: &str, downloaded: u64, total: Option<u64>, done: bool) {
    const BAR_W: usize = 20;
    let mark = if done { "✓" } else { " " };
    if let Some(tot) = total {
        let pct = ((downloaded * 100) / tot.max(1)) as usize;
        let filled = (pct.min(100) * BAR_W) / 100;
        let bar = format!("{}{}", "█".repeat(filled), "░".repeat(BAR_W - filled));
        print!(
            "\r  {:<40} {} {:>3}%  {:>8} / {:<8}  {}",
            label,
            bar,
            pct,
            fmt_size(downloaded),
            fmt_size(tot),
            mark,
        );
    } else {
        print!("\r  {:<40}  {:>8}  {}", label, fmt_size(downloaded), mark);
    }
    std::io::stdout().flush().ok();
}

fn truncate_label(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("...{}", &s[s.len() - (max - 3)..])
    }
}

fn fmt_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    if bytes >= GB {
        format!("{:.2}GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1}MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.0}KB", bytes as f64 / KB as f64)
    } else {
        format!("{}B", bytes)
    }
}

fn check_status(status: u16, repo_id: &str) -> anyhow::Result<()> {
    match status {
        200..=299 => Ok(()),
        401 => anyhow::bail!(
            "Authentication required for '{repo_id}'.\n\
             Create a token at https://huggingface.co/settings/tokens, then:\n\
               rllm pull {repo_id} --token <TOKEN>\n\
             or set the HF_TOKEN environment variable."
        ),
        403 => anyhow::bail!(
            "Access denied to '{repo_id}' — this model requires accepting a license.\n\
             1. Visit https://huggingface.co/{repo_id} and accept the terms\n\
             2. Create a token at https://huggingface.co/settings/tokens\n\
             3. Run:  rllm pull {repo_id} --token <TOKEN>"
        ),
        404 => anyhow::bail!("Model '{repo_id}' not found on HuggingFace."),
        code => anyhow::bail!("HuggingFace returned HTTP {}.", code),
    }
}
