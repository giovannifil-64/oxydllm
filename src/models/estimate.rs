// ─────────────────────────────────────────────────────────────────────────────
// estimate.rs — Memory and accuracy estimator for local and remote models
// ─────────────────────────────────────────────────────────────────────────────
//
// Usage:
//   rllm estimate <model-name>                    # local model in models_dir
//   rllm estimate <user/repo>                     # remote HF repo (no download)
//   rllm estimate <model> --context-len 8192      # custom context length
//   rllm estimate <model> --num-sequences 4       # concurrent sequences
//
// For local GGUF models, only the file header is parsed — the full quantized
// weight data is NOT loaded into memory, making this command instant.
// ─────────────────────────────────────────────────────────────────────────────

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::Result;
use candle_core::quantized::gguf_file;

const HF_ENDPOINT: &str = "https://huggingface.co";


pub struct EstimateArgs {
    pub model: String,
    pub models_dir: PathBuf,
    pub token: Option<String>,
    pub context_len: usize,
    pub num_sequences: usize,
}

pub fn run_estimate(args: &EstimateArgs) -> Result<()> {
    let local_path = resolve_local_path(&args.model, &args.models_dir);

    if let Some(path) = local_path {
        estimate_local(&path, args.context_len, args.num_sequences)
    } else if args.model.contains('/') {
        estimate_remote(
            &args.model,
            args.token.as_deref(),
            args.context_len,
            args.num_sequences,
        )
    } else {
        anyhow::bail!(
            "Model '{}' not found in {}.\n\
             For remote estimation use a HF repo ID (e.g. Qwen/Qwen3-1.7B-GGUF).",
            args.model,
            args.models_dir.display()
        )
    }
}


fn resolve_local_path(model: &str, models_dir: &Path) -> Option<PathBuf> {
    if model.starts_with('/') || model.starts_with('.') {
        let p = PathBuf::from(model);
        if p.exists() {
            return Some(p);
        }
    }
    let p = models_dir.join(model);
    if p.exists() {
        return Some(p);
    }
    None
}

fn estimate_local(dir: &Path, ctx_len: usize, num_seqs: usize) -> Result<()> {
    if let Some(gguf_path) = find_gguf_file(dir) {
        return estimate_local_gguf(&gguf_path, ctx_len, num_seqs);
    }
    let has_st = std::fs::read_dir(dir)?.any(|e| {
        e.ok()
            .map(|e| {
                e.file_name()
                    .to_string_lossy()
                    .to_lowercase()
                    .ends_with(".safetensors")
            })
            .unwrap_or(false)
    });
    if has_st {
        return estimate_local_safetensors(dir, ctx_len, num_seqs);
    }
    anyhow::bail!("No GGUF or safetensors files found in {}", dir.display())
}

fn find_gguf_file(dir: &Path) -> Option<PathBuf> {
    std::fs::read_dir(dir).ok()?.find_map(|e| {
        let e = e.ok()?;
        let name = e.file_name().to_string_lossy().to_lowercase();
        if name.ends_with(".gguf") {
            Some(e.path())
        } else {
            None
        }
    })
}


fn estimate_local_gguf(gguf_path: &Path, ctx_len: usize, num_seqs: usize) -> Result<()> {
    let model_name = gguf_path
        .parent()
        .and_then(|p| p.file_name())
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "Unknown".to_string());

    let mut file = std::fs::File::open(gguf_path)
        .map_err(|e| anyhow::anyhow!("Cannot open {}: {}", gguf_path.display(), e))?;
    let content = gguf_file::Content::read(&mut file)
        .map_err(|e| anyhow::anyhow!("Cannot parse GGUF header: {}", e))?;

    let arch = meta_string(&content, "general.architecture").unwrap_or_else(|| "unknown".to_string());
    let prefix = &arch;

    let weights_bytes = std::fs::metadata(gguf_path)?.len() as usize;

    let filename = gguf_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy();
    let quant_str = extract_quant_from_filename(&filename)
        .or_else(|| detect_quant_from_content(&content))
        .unwrap_or_else(|| "unknown".to_string());

    let geometry = read_geometry_from_content(&content, prefix).ok();

    println!();
    println!("  Model    {}", model_name);
    println!("  File     {}", gguf_path.display());
    println!("  Arch     {}", arch);
    println!("  Format   {}", quant_str);
    println!();
    print_weights_kv_total(weights_bytes, geometry.as_ref(), ctx_len, num_seqs);
    print_accuracy_line(&quant_str);

    Ok(())
}


fn estimate_local_safetensors(dir: &Path, ctx_len: usize, num_seqs: usize) -> Result<()> {
    let model_name = dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "Unknown".to_string());

    let mut weights_bytes: usize = 0;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_lowercase();
        if name.ends_with(".safetensors") {
            weights_bytes += entry.metadata()?.len() as usize;
        }
    }

    let config_path = dir.join("config.json");
    let geometry = if config_path.exists() {
        parse_geometry_from_config_file(&config_path).ok()
    } else {
        None
    };

    let dtype_str = read_torch_dtype(&config_path)
        .unwrap_or_else(|| "F32 / BF16".to_string());

    println!();
    println!("  Model    {}", model_name);
    println!("  Dir      {}", dir.display());
    println!("  Format   {} safetensors", dtype_str);
    println!();
    print_weights_kv_total(weights_bytes, geometry.as_ref(), ctx_len, num_seqs);
    println!("  Accuracy 100%  (full-precision weights)");
    println!();

    Ok(())
}


fn estimate_remote(repo_id: &str, token: Option<&str>, ctx_len: usize, num_seqs: usize) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .user_agent(concat!("rllm/", env!("CARGO_PKG_VERSION")))
        .build()?;

    print!("  Fetching file list for {}...", repo_id);
    std::io::stdout().flush().ok();

    let url = format!("{}/api/models/{}?blobs=true", HF_ENDPOINT, repo_id);
    let mut req = client.get(&url);
    if let Some(tok) = token {
        req = req.bearer_auth(tok);
    }
    let resp = req.send()?;
    let status = resp.status().as_u16();
    if !(200..=299).contains(&status) {
        anyhow::bail!("HuggingFace returned HTTP {} for '{}'", status, repo_id);
    }
    let json: serde_json::Value = resp.json()?;
    println!(" done");

    let siblings = json["siblings"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("Unexpected HF API response (missing 'siblings')"))?;

    let mut gguf_files: Vec<(String, u64)> = siblings
        .iter()
        .filter_map(|s| {
            let name = s["rfilename"].as_str()?;
            if !name.to_lowercase().ends_with(".gguf") {
                return None;
            }
            let size = s["lfs"]["size"]
                .as_u64()
                .or_else(|| s["size"].as_u64())
                .unwrap_or(0);
            Some((name.to_string(), size))
        })
        .collect();
    gguf_files.sort_by_key(|(_, size)| *size);

    let safetensors_bytes: u64 = siblings
        .iter()
        .filter_map(|s| {
            let name = s["rfilename"].as_str()?;
            if !name.to_lowercase().ends_with(".safetensors") {
                return None;
            }
            s["lfs"]["size"].as_u64().or_else(|| s["size"].as_u64())
        })
        .sum();

    let geometry = fetch_remote_config(&client, repo_id, token).or_else(|| {
        let base_repo = repo_id
            .trim_end_matches("-GGUF")
            .trim_end_matches("-gguf");
        if base_repo != repo_id {
            fetch_remote_config(&client, base_repo, token)
        } else {
            None
        }
    });

    println!();
    println!("  Model  {}  (remote)", repo_id);

    if !gguf_files.is_empty() {
        println!();
        let kv_header = if geometry.is_some() {
            format!("  {:>9}", "KV cache")
        } else {
            format!("  {:>9}", "KV cache")
        };
        println!(
            "  {:<26}  {:>9}{kv_header}  {:>10}  Accuracy",
            "Format", "Weights", "Total"
        );
        println!("  {}", "─".repeat(72));

        let recommended = best_recommendation(&gguf_files);

        for (filename, size) in &gguf_files {
            let quant = extract_quant_from_filename(filename)
                .unwrap_or_else(|| "?".to_string());
            let kv_bytes = geometry.as_ref().map(|g| kv_cache_bytes(g, ctx_len, num_seqs));
            let total = kv_bytes.map(|kv| *size as usize + kv);
            let kv_str = kv_bytes.map(fmt_bytes).unwrap_or_else(|| "?".to_string());
            let total_str = total.map(fmt_bytes).unwrap_or_else(|| "?".to_string());
            let acc = quant_accuracy_str(&quant);
            let star = if Some(filename.as_str()) == recommended { " ★" } else { "" };

            println!(
                "  {:<26}  {:>9}  {:>9}  {:>10}  {}{}",
                quant,
                fmt_bytes(*size as usize),
                kv_str,
                total_str,
                acc,
                star,
            );
        }

        println!();
        if geometry.is_some() {
            println!(
                "  ★ recommended  |  KV cache: {} ctx × {} seq (BF16)",
                ctx_len, num_seqs
            );
        } else {
            println!("  ★ recommended  |  KV cache: unavailable (config.json not in repo)");
        }
    }

    if safetensors_bytes > 0 {
        println!();
        let kv_bytes = geometry.as_ref().map(|g| kv_cache_bytes(g, ctx_len, num_seqs));
        let kv_str = kv_bytes.map(fmt_bytes).unwrap_or_else(|| "?".to_string());
        let total = kv_bytes.map(|b| b + safetensors_bytes as usize);
        let total_str = total.map(fmt_bytes).unwrap_or_else(|| "?".to_string());
        println!("  {}", "─".repeat(72));
        println!(
            "  {:<26}  {:>9}  {:>9}  {:>10}  100%",
            "safetensors (F32/BF16)",
            fmt_bytes(safetensors_bytes as usize),
            kv_str,
            total_str,
        );
    }

    if gguf_files.is_empty() && safetensors_bytes == 0 {
        println!();
        println!("  No GGUF or safetensors files found in this repository.");
    }

    println!();
    Ok(())
}

fn fetch_remote_config(
    client: &reqwest::blocking::Client,
    repo_id: &str,
    token: Option<&str>,
) -> Option<ModelGeometry> {
    let url = format!("{}/{}/resolve/main/config.json", HF_ENDPOINT, repo_id);
    let mut req = client.get(&url);
    if let Some(tok) = token {
        req = req.bearer_auth(tok);
    }
    let resp = req.send().ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let json: serde_json::Value = resp.json().ok()?;
    parse_geometry_from_json(&json)
}

struct ModelGeometry {
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

fn read_geometry_from_content(content: &gguf_file::Content, prefix: &str) -> Result<ModelGeometry> {
    let num_layers = meta_u32(content, &format!("{prefix}.block_count"))
        .ok_or_else(|| anyhow::anyhow!("missing block_count"))? as usize;
    let num_attn_heads = meta_u32(content, &format!("{prefix}.attention.head_count"))
        .ok_or_else(|| anyhow::anyhow!("missing head_count"))? as usize;
    let num_kv_heads = meta_u32(content, &format!("{prefix}.attention.head_count_kv"))
        .ok_or_else(|| anyhow::anyhow!("missing head_count_kv"))? as usize;

    let head_dim = meta_u32(content, &format!("{prefix}.attention.key_length"))
        .map(|v| v as usize)
        .unwrap_or_else(|| {
            content
                .tensor_infos
                .get("blk.0.attn_q.weight")
                .map(|info| info.shape.dims()[0] / num_attn_heads)
                .unwrap_or(64)
        });

    Ok(ModelGeometry { num_layers, num_kv_heads, head_dim })
}

fn parse_geometry_from_config_file(path: &Path) -> Result<ModelGeometry> {
    let content = std::fs::read_to_string(path)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;
    parse_geometry_from_json(&json)
        .ok_or_else(|| anyhow::anyhow!("Could not parse model geometry from config.json"))
}

fn parse_geometry_from_json(json: &serde_json::Value) -> Option<ModelGeometry> {
    let num_layers = json["num_hidden_layers"].as_u64()? as usize;
    let num_kv_heads = json["num_key_value_heads"]
        .as_u64()
        .or_else(|| json["num_attention_heads"].as_u64())? as usize;
    let head_dim = if let Some(hd) = json["head_dim"].as_u64() {
        hd as usize
    } else {
        let hidden = json["hidden_size"].as_u64()? as usize;
        let heads = json["num_attention_heads"].as_u64()? as usize;
        hidden / heads
    };
    Some(ModelGeometry { num_layers, num_kv_heads, head_dim })
}

fn kv_cache_bytes(g: &ModelGeometry, ctx_len: usize, num_seqs: usize) -> usize {
    2 * g.num_layers * g.num_kv_heads * g.head_dim * 2 * ctx_len * num_seqs
}


fn print_weights_kv_total(
    weights_bytes: usize,
    geometry: Option<&ModelGeometry>,
    ctx_len: usize,
    num_seqs: usize,
) {
    println!("  Weights  {}", fmt_bytes(weights_bytes));
    if let Some(g) = geometry {
        let kv = kv_cache_bytes(g, ctx_len, num_seqs);
        let total = weights_bytes + kv;
        println!(
            "  KV cache {}  ({} ctx × {} seq, BF16)",
            fmt_bytes(kv),
            ctx_len,
            num_seqs,
        );
        println!("           {}", "─".repeat(12));
        println!("  Total    {}", fmt_bytes(total));
    } else {
        println!("  KV cache ?  (architecture info unavailable)");
    }
    println!();
}

fn print_accuracy_line(quant: &str) {
    if let Some((pct, desc)) = quant_accuracy(quant) {
        println!("  Accuracy {}  ({})", pct, desc);
    }
    println!();
}

pub fn extract_quant_from_filename(filename: &str) -> Option<String> {
    const KNOWN: &[&str] = &[
        "IQ1_S", "IQ1_M",
        "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
        "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M", "IQ3_K_S", "IQ3_K_M",
        "IQ4_NL", "IQ4_XS",
        "Q2_K_S", "Q2_K",
        "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q3_K",
        "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "Q4_K",
        "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M", "Q5_K",
        "Q6_K_L", "Q6_K",
        "Q8_0", "Q8_K_M", "Q8_K_S", "Q8_K",
        "BF16", "F16", "F32",
    ];

    let upper = filename.to_uppercase();
    for &q in KNOWN {
        if let Some(pos) = upper.find(q) {
            let before_ok = pos == 0
                || !upper.as_bytes()[pos - 1].is_ascii_alphanumeric();
            let end = pos + q.len();
            let after_ok = end >= upper.len()
                || !upper.as_bytes()[end].is_ascii_alphanumeric();
            if before_ok && after_ok {
                return Some(q.to_string());
            }
        }
    }
    None
}

fn detect_quant_from_content(content: &gguf_file::Content) -> Option<String> {
    let info = content.tensor_infos.get("blk.0.ffn_down.weight")
        .or_else(|| content.tensor_infos.get("blk.0.attn_q.weight"))?;
    Some(ggml_dtype_label(&format!("{:?}", info.ggml_dtype)))
}

fn ggml_dtype_label(debug: &str) -> String {
    match debug {
        "F32"  => "F32",
        "F16"  => "F16",
        "BF16" => "BF16",
        "Q4_0" => "Q4_0", "Q4_1" => "Q4_1",
        "Q5_0" => "Q5_0", "Q5_1" => "Q5_1",
        "Q8_0" => "Q8_0", "Q8_1" => "Q8_1",
        "Q2K"  => "Q2_K",
        "Q3K"  => "Q3_K",
        "Q4K"  => "Q4_K",
        "Q5K"  => "Q5_K",
        "Q6K"  => "Q6_K",
        "Q8K"  => "Q8_K",
        other  => other,
    }.to_string()
}


fn quant_accuracy(quant: &str) -> Option<(&'static str, &'static str)> {
    match quant.to_uppercase().as_str() {
        "IQ1_S" | "IQ1_M" => Some(("~92%", "extremely aggressive, strong degradation")),
        "IQ2_XXS" | "IQ2_XS" | "IQ2_S" | "IQ2_M" | "Q2_K" | "Q2_K_S" => {
            Some(("~95%", "aggressive, noticeable quality loss"))
        }
        "IQ3_XXS" | "IQ3_XS" | "IQ3_S" | "Q3_K_S" => {
            Some(("~96%", "compact, some quality loss"))
        }
        "Q3_K" | "Q3_K_M" | "IQ3_M" | "IQ3_K_S" => {
            Some(("~96.5%", "compact, moderate quality"))
        }
        "Q3_K_L" | "IQ3_K_M" => Some(("~97%", "compact, good quality")),
        "Q4_0" | "Q4_1" => Some(("~97.5%", "good compression")),
        "Q4_K_S" | "IQ4_XS" => Some(("~97.8%", "good balance")),
        "Q4_K" | "Q4_K_M" | "IQ4_NL" => Some(("~98.5%", "excellent balance — recommended")),
        "Q5_0" | "Q5_1" => Some(("~98.8%", "high quality")),
        "Q5_K" | "Q5_K_S" => Some(("~99%", "high quality")),
        "Q5_K_M" => Some(("~99.2%", "high quality")),
        "Q6_K" | "Q6_K_L" => Some(("~99.5%", "near-lossless")),
        "Q8_0" | "Q8_K" | "Q8_K_M" | "Q8_K_S" => Some(("~99.7%", "near-lossless")),
        "F16" | "BF16" => Some(("~99.9%", "floating point — baseline")),
        "F32" => Some(("100%", "full precision")),
        _ => None,
    }
}

pub fn quant_accuracy_str(quant: &str) -> &'static str {
    quant_accuracy(quant).map(|(pct, _)| pct).unwrap_or("?")
}

fn best_recommendation(files: &[(String, u64)]) -> Option<&str> {
    if let Some((name, _)) = files.iter().find(|(f, _)| {
        extract_quant_from_filename(f)
            .map(|q| q.eq_ignore_ascii_case("Q4_K_M"))
            .unwrap_or(false)
    }) {
        return Some(name.as_str());
    }
    files.iter().find_map(|(name, _)| {
        let q = extract_quant_from_filename(name)?;
        match q.to_uppercase().as_str() {
            "Q4_K_S" | "Q4_K" | "Q4_0" | "Q4_1" | "IQ4_NL" | "IQ4_XS" => Some(name.as_str()),
            _ => None,
        }
    })
}


fn meta_u32(content: &gguf_file::Content, key: &str) -> Option<u32> {
    content.metadata.get(key).and_then(|v| v.to_u32().ok())
}

fn meta_string(content: &gguf_file::Content, key: &str) -> Option<String> {
    content.metadata.get(key).and_then(|v| v.to_string().ok().cloned())
}

fn read_torch_dtype(config_path: &Path) -> Option<String> {
    let content = std::fs::read_to_string(config_path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;
    json["torch_dtype"]
        .as_str()
        .map(|s| s.to_uppercase())
}

fn fmt_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;
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
