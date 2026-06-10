use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const HF_ENDPOINT: &str = "https://huggingface.co";

pub struct PullConfig {
    pub repo_id: String,
    pub dest_name: String,
    pub models_dir: PathBuf,
    pub token: Option<String>,
    pub force: bool,
    pub variant: Option<String>,
}

struct GgufVariant {
    quant_name: String,
    files: Vec<(String, u64)>,
}

impl GgufVariant {
    fn total_size(&self) -> u64 {
        self.files.iter().map(|(_, s)| s).sum()
    }
    fn is_split(&self) -> bool {
        self.files.len() > 1
            || self
                .files
                .first()
                .map(|(f, _)| has_split_suffix(f))
                .unwrap_or(false)
    }
}

fn strip_split_suffix(stem: &str) -> &str {
    let parts: Vec<&str> = stem.split('-').collect();
    let n = parts.len();
    if n >= 3
        && parts[n - 2] == "of"
        && parts[n - 1].chars().all(|c| c.is_ascii_digit())
        && parts[n - 3].chars().all(|c| c.is_ascii_digit())
    {
        let trim = 1 + parts[n - 1].len() + 1 + parts[n - 2].len() + 1 + parts[n - 3].len();
        &stem[..stem.len() - trim]
    } else {
        stem
    }
}

fn has_split_suffix(filename: &str) -> bool {
    let stem = filename.strip_suffix(".gguf").unwrap_or(filename);
    stem != strip_split_suffix(stem)
}

fn group_gguf_variants(gguf_files: &[(String, u64)]) -> Vec<GgufVariant> {
    let mut groups: Vec<(String, Vec<(String, u64)>)> = Vec::new();

    for (name, size) in gguf_files {
        let stem = name.strip_suffix(".gguf").unwrap_or(name);
        let key = strip_split_suffix(stem).to_string();
        if let Some(g) = groups.iter_mut().find(|(k, _)| k == &key) {
            g.1.push((name.clone(), *size));
        } else {
            groups.push((key, vec![(name.clone(), *size)]));
        }
    }

    groups
        .into_iter()
        .map(|(key, mut files)| {
            files.sort_by_key(|(f, _)| f.clone());
            let quant_name = crate::models::estimate::extract_quant_from_filename(&key)
                .unwrap_or_else(|| key.clone());
            GgufVariant { quant_name, files }
        })
        .collect()
}

fn select_variant<'a>(
    variants: &'a [GgufVariant],
    preferred: Option<&str>,
    already_present: &std::collections::HashSet<String>,
) -> anyhow::Result<&'a GgufVariant> {
    if let Some(pref) = preferred {
        return variants
            .iter()
            .find(|v| v.quant_name.eq_ignore_ascii_case(pref))
            .ok_or_else(|| {
                let avail: Vec<&str> = variants.iter().map(|v| v.quant_name.as_str()).collect();
                anyhow::anyhow!(
                    "Variant '{}' not found in this repo. Available: {}",
                    pref,
                    avail.join(", ")
                )
            });
    }

    if variants.len() == 1 {
        return Ok(&variants[0]);
    }

    let present: Vec<&GgufVariant> = variants
        .iter()
        .filter(|v| already_present.contains(&v.quant_name))
        .collect();
    let available: Vec<&GgufVariant> = variants
        .iter()
        .filter(|v| !already_present.contains(&v.quant_name))
        .collect();

    println!("  Multiple GGUF variants available — choose one to download:\n");

    if !present.is_empty() {
        println!("  Already downloaded:");
        for v in &present {
            println!(
                "    \u{2713} {:<16}  {:>10}  {}",
                v.quant_name,
                fmt_size_f(v.total_size()),
                crate::models::estimate::quant_accuracy_str(&v.quant_name),
            );
        }
        println!();
    }

    if available.is_empty() {
        anyhow::bail!(
            "All variants of this model are already downloaded. Use --force to re-download."
        );
    }

    println!(
        "  {:>2}  {:<16}  {:>10}  {:>5}  Accuracy",
        "#", "Format", "Size", "Files"
    );
    println!("  {}", "─".repeat(56));

    let recommended_idx = best_variant_idx(&available);

    for (i, v) in available.iter().enumerate() {
        let star = if Some(i) == recommended_idx {
            " ★"
        } else {
            ""
        };
        let files_label = if v.is_split() {
            format!("{} shards", v.files.len())
        } else {
            "1".to_string()
        };
        println!(
            "  {:>2}  {:<16}  {:>10}  {:>5}  {}{}",
            i + 1,
            v.quant_name,
            fmt_size_f(v.total_size()),
            files_label,
            crate::models::estimate::quant_accuracy_str(&v.quant_name),
            star,
        );
    }
    println!();

    let default = recommended_idx.map(|i| i + 1).unwrap_or(1);

    use std::io::IsTerminal;
    if !std::io::stdin().is_terminal() {
        println!(
            "  Non-interactive — selecting {} (#{}).",
            available[default - 1].quant_name,
            default
        );
        return Ok(available[default - 1]);
    }

    loop {
        print!("  Select [1-{}] (default: {}): ", available.len(), default);
        std::io::stdout().flush().ok();
        let mut line = String::new();
        std::io::stdin().lock().read_line(&mut line)?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Ok(available[default - 1]);
        }
        match trimmed.parse::<usize>() {
            Ok(n) if n >= 1 && n <= available.len() => return Ok(available[n - 1]),
            _ => println!("  Please enter a number between 1 and {}.", available.len()),
        }
    }
}

fn best_variant_idx(variants: &[&GgufVariant]) -> Option<usize> {
    if let Some(i) = variants
        .iter()
        .position(|v| v.quant_name.eq_ignore_ascii_case("Q4_K_M"))
    {
        return Some(i);
    }
    variants.iter().position(|v| {
        matches!(
            v.quant_name.to_uppercase().as_str(),
            "Q4_K_S" | "Q4_K" | "Q4_0" | "Q4_1" | "IQ4_NL" | "IQ4_XS" | "Q5_K_M"
        )
    })
}

fn is_incomplete_download(dir: &Path) -> bool {
    let index_path = dir.join("gguf.index");
    if index_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&index_path) {
            let all_present = content
                .lines()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .all(|fname| dir.join(fname).exists());
            return !all_present;
        }
        return true;
    }

    // Sharded safetensors: every shard listed in the index must exist and the
    // on-disk bytes must cover the index's total_size — a bare existence check
    // would accept a shard truncated by an interrupted download.
    let st_index = dir.join("model.safetensors.index.json");
    if st_index.exists() {
        let Ok(raw) = std::fs::read_to_string(&st_index) else {
            return true;
        };
        let Ok(json) = serde_json::from_str::<serde_json::Value>(&raw) else {
            return true;
        };
        let Some(map) = json["weight_map"].as_object() else {
            return true;
        };
        let shards: std::collections::HashSet<&str> =
            map.values().filter_map(|v| v.as_str()).collect();
        let mut disk_bytes: u64 = 0;
        for shard in &shards {
            match dir.join(shard).metadata() {
                Ok(m) => disk_bytes += m.len(),
                Err(_) => return true,
            }
        }
        if let Some(total) = json["metadata"]["total_size"].as_u64() {
            return disk_bytes < total;
        }
        return false;
    }

    if dir.join("config.json").exists() {
        let has_weights = std::fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .any(|e| {
                e.path()
                    .extension()
                    .map(|x| x == "safetensors")
                    .unwrap_or(false)
            });
        return !has_weights;
    }

    true
}

pub fn pull(config: &PullConfig) -> anyhow::Result<()> {
    let dest = config.models_dir.join(&config.dest_name);

    println!("Repository : {}", config.repo_id);
    println!("Destination: {}", dest.display());
    if config.token.is_some() {
        println!("Auth       : token provided");
    }
    println!();

    let client = reqwest::blocking::Client::builder()
        .timeout(None)
        .user_agent(concat!("oxydllm/", env!("CARGO_PKG_VERSION")))
        .build()?;

    print!("Fetching file list...");
    std::io::stdout().flush().ok();
    let all_files = list_repo_files(&client, &config.repo_id, config.token.as_deref())?;
    println!();

    let (gguf_files, mut metadata_files): (Vec<_>, Vec<_>) = all_files
        .into_iter()
        .filter(|(f, _)| is_relevant_file(f))
        .partition(|(f, _)| f.to_lowercase().ends_with(".gguf"));

    let has_safetensors = metadata_files
        .iter()
        .any(|(f, _)| f.ends_with(".safetensors"));
    let mut download_safetensors = false;

    let gguf_to_download: Vec<String> = if gguf_files.is_empty() {
        download_safetensors = true;
        Vec::new()
    } else {
        let mut variants = group_gguf_variants(&gguf_files);
        if has_safetensors {
            let st_files: Vec<_> = metadata_files
                .iter()
                .filter(|(f, _)| f.ends_with(".safetensors"))
                .cloned()
                .collect();
            variants.insert(
                0,
                GgufVariant {
                    quant_name: "Safetensors".to_string(),
                    files: st_files,
                },
            );
        }

        println!();

        let target_variant_str = config.variant.as_deref().map(|s| {
            if s.eq_ignore_ascii_case("safetensors") {
                "Safetensors"
            } else {
                s
            }
        });

        let already_present: std::collections::HashSet<String> = if dest.exists() && !config.force {
            variants
                .iter()
                .filter(|v| v.quant_name != "Safetensors")
                .filter(|v| {
                    v.files.iter().all(|(fname, expected_size)| {
                        let p = dest.join(fname);
                        p.exists()
                            && (*expected_size == 0
                                || p.metadata().map(|m| m.len()).unwrap_or(0) >= *expected_size)
                    })
                })
                .map(|v| v.quant_name.clone())
                .collect()
        } else {
            std::collections::HashSet::new()
        };

        let chosen = select_variant(&variants, target_variant_str, &already_present)?;

        if chosen.quant_name == "Safetensors" {
            download_safetensors = true;
            println!(
                "  Selected: Safetensors ({})\n",
                fmt_size_f(chosen.total_size())
            );
            Vec::new()
        } else {
            let variant_filenames: Vec<String> =
                chosen.files.iter().map(|(f, _)| f.clone()).collect();

            if dest.exists() {
                for (f, _) in &chosen.files {
                    let _ = std::fs::remove_file(dest.join(f));
                }
                let _ = std::fs::remove_file(dest.join("gguf.index"));
            }

            if variants.len() > 1 {
                println!(
                    "  Downloading: {} ({}{})\n",
                    chosen.quant_name,
                    fmt_size_f(chosen.total_size()),
                    if chosen.is_split() {
                        format!(", {} shards", chosen.files.len())
                    } else {
                        String::new()
                    }
                );
            } else {
                println!(
                    "  Found: {} ({}{})\n",
                    chosen.quant_name,
                    fmt_size_f(chosen.total_size()),
                    if chosen.is_split() {
                        format!(", {} shards", chosen.files.len())
                    } else {
                        String::new()
                    }
                );
            }
            variant_filenames
        }
    };

    if !download_safetensors {
        metadata_files.retain(|(f, _)| !f.ends_with(".safetensors"));
    }

    if download_safetensors && dest.exists() {
        if config.force {
            println!("Removing existing model at {}...", dest.display());
            std::fs::remove_dir_all(&dest)?;
        } else if is_incomplete_download(&dest) {
            // Resume at file granularity: drop only files whose size doesn't
            // match the upstream listing; complete files are skipped by the
            // to_download filter below.
            println!("Resuming interrupted download...");
            for (name, size) in &metadata_files {
                let p = dest.join(name);
                if *size > 0 && p.exists() && p.metadata().map(|m| m.len()).unwrap_or(0) != *size {
                    let _ = std::fs::remove_file(&p);
                }
            }
        } else {
            anyhow::bail!(
                "A model named '{}' already exists at {}.\n\
                 Use --force to overwrite, or --name <name> to save under a different name.",
                config.dest_name,
                dest.display()
            );
        }
    }

    let mut to_download: Vec<String> = metadata_files
        .into_iter()
        .map(|(f, _)| f)
        .filter(|f| !dest.join(f).exists())
        .collect();
    to_download.extend(gguf_to_download.iter().cloned());
    to_download.sort_by_key(|f| if f.ends_with(".json") { 0u8 } else { 1u8 });

    if to_download.is_empty() {
        anyhow::bail!(
            "No compatible model files found in '{}'.\n\
             The repository may not contain safetensors or GGUF weights.",
            config.repo_id
        );
    }

    std::fs::create_dir_all(&dest)?;

    let mut downloaded_files: Vec<String> = Vec::new();
    let mut failed_file: Option<String> = None;
    for filename in &to_download {
        match download_file(
            &client,
            &config.repo_id,
            filename,
            &dest,
            config.token.as_deref(),
        ) {
            Ok(()) => downloaded_files.push(filename.clone()),
            Err(e) => {
                tracing::error!(filename = %filename, error = %e, "error downloading model file");
                failed_file = Some(filename.clone());
                break;
            }
        }
    }

    if let Some(failed) = failed_file {
        // Keep completed files so a retry resumes from them; drop only the
        // truncated in-flight file.
        let _ = std::fs::remove_file(dest.join(&failed));
        anyhow::bail!(
            "Download incomplete ({} of {} files done). Re-run the same pull to resume.",
            downloaded_files.len(),
            to_download.len()
        );
    }

    let new_shards: Vec<&str> = gguf_to_download
        .iter()
        .filter(|f| f.to_lowercase().ends_with(".gguf"))
        .map(|f| f.as_str())
        .collect();

    if !new_shards.is_empty() {
        let mut existing: Vec<String> = std::fs::read_dir(&dest)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let p = e.path();
                if p.extension().and_then(|x| x.to_str()) == Some("gguf") {
                    p.file_name().map(|n| n.to_string_lossy().to_string())
                } else {
                    None
                }
            })
            .collect();
        existing.sort();

        let index_path = dest.join("gguf.index");
        let mut index_file = std::fs::File::create(&index_path)?;
        for shard in &existing {
            writeln!(index_file, "{}", shard)?;
        }
    }

    println!("\nModel '{}' saved to {}", config.dest_name, dest.display());
    Ok(())
}

fn list_repo_files(
    client: &reqwest::blocking::Client,
    repo_id: &str,
    token: Option<&str>,
) -> anyhow::Result<Vec<(String, u64)>> {
    let url = format!("{}/api/models/{}?blobs=true", HF_ENDPOINT, repo_id);
    let mut builder = client.get(&url);
    if let Some(tok) = token {
        builder = builder.bearer_auth(tok);
    }
    let resp = builder.send()?;
    let status = resp.status().as_u16();
    check_status(status, repo_id)?;

    let json: serde_json::Value = resp.json()?;
    let siblings = json["siblings"].as_array().ok_or_else(|| {
        anyhow::anyhow!("Unexpected HuggingFace API response (missing 'siblings')")
    })?;

    Ok(siblings
        .iter()
        .filter_map(|s| {
            let name = s["rfilename"].as_str()?.to_string();
            let size = s["lfs"]["size"]
                .as_u64()
                .or_else(|| s["size"].as_u64())
                .unwrap_or(0);
            Some((name, size))
        })
        .collect())
}

fn is_relevant_file(f: &str) -> bool {
    if f.contains('/') || f.starts_with('.') {
        return false;
    }
    let l = f.to_lowercase();
    if l == "consolidated.safetensors"
        || l.ends_with(".pth")
        || l.ends_with(".pt")
        || l.ends_with(".bin")
    {
        return false;
    }
    l.ends_with(".json")
        || l.ends_with(".safetensors")
        || l.ends_with(".gguf")
        || l.ends_with(".model")
        || l.ends_with(".tiktoken")
        || l.ends_with(".jinja")
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

fn fmt_size_f(bytes: u64) -> String {
    const MB: u64 = 1024 * 1024;
    const GB: u64 = MB * 1024;
    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else {
        format!("{:.0} MB", bytes as f64 / MB as f64)
    }
}

fn check_status(status: u16, repo_id: &str) -> anyhow::Result<()> {
    match status {
        200..=299 => Ok(()),
        401 => anyhow::bail!(
            "Authentication required for '{repo_id}'.\n\
             Create a token at https://huggingface.co/settings/tokens, then:\n\
               oxydllm pull {repo_id} --token <TOKEN>\n\
             or set the HF_TOKEN environment variable."
        ),
        403 => anyhow::bail!(
            "Access denied to '{repo_id}' — this model requires accepting a license.\n\
             1. Visit https://huggingface.co/{repo_id} and accept the terms\n\
             2. Create a token at https://huggingface.co/settings/tokens\n\
             3. Run:  oxydllm pull {repo_id} --token <TOKEN>"
        ),
        404 => anyhow::bail!("Model '{repo_id}' not found on HuggingFace."),
        code => anyhow::bail!("HuggingFace returned HTTP {}.", code),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_split_suffix_removes_shard_numbers() {
        assert_eq!(
            strip_split_suffix("model-Q4_K_M-00001-of-00003"),
            "model-Q4_K_M"
        );
    }

    #[test]
    fn strip_split_suffix_keeps_non_shard_name() {
        assert_eq!(strip_split_suffix("model-Q4_K_M"), "model-Q4_K_M");
    }

    #[test]
    fn strip_split_suffix_keeps_partial_pattern() {
        assert_eq!(strip_split_suffix("model-00001"), "model-00001");
    }

    #[test]
    fn has_split_suffix_detects_shard_gguf() {
        assert!(has_split_suffix("model-Q4_K_M-00001-of-00002.gguf"));
    }

    #[test]
    fn has_split_suffix_rejects_plain_gguf() {
        assert!(!has_split_suffix("model-Q4_K_M.gguf"));
    }

    #[test]
    fn is_relevant_file_accepts_standard_formats() {
        assert!(is_relevant_file("config.json"));
        assert!(is_relevant_file("model.safetensors"));
        assert!(is_relevant_file("model.gguf"));
        assert!(is_relevant_file("tokenizer.model"));
    }

    #[test]
    fn is_relevant_file_rejects_binaries_and_nested_paths() {
        assert!(!is_relevant_file("model.bin"));
        assert!(!is_relevant_file("model.pt"));
        assert!(!is_relevant_file("model.pth"));
        assert!(!is_relevant_file("sub/config.json")); // nested
        assert!(!is_relevant_file(".hidden.json")); // dotfile
        assert!(!is_relevant_file("consolidated.safetensors")); // excluded by name
    }

    #[test]
    fn group_gguf_variants_merges_shards_into_one_variant() {
        let files = vec![
            ("model-Q4_K_M-00001-of-00002.gguf".to_string(), 100u64),
            ("model-Q4_K_M-00002-of-00002.gguf".to_string(), 80u64),
            ("model-Q8_0.gguf".to_string(), 200u64),
        ];
        let variants = group_gguf_variants(&files);
        assert_eq!(
            variants.len(),
            2,
            "expected 2 variants: Q4_K_M (sharded) and Q8_0"
        );
        let q4 = variants
            .iter()
            .find(|v| v.is_split())
            .expect("sharded variant");
        assert_eq!(q4.files.len(), 2);
        assert_eq!(q4.total_size(), 180);
    }

    #[test]
    fn group_gguf_variants_single_file_is_not_split() {
        let files = vec![("model-Q4_K_M.gguf".to_string(), 500u64)];
        let variants = group_gguf_variants(&files);
        assert_eq!(variants.len(), 1);
        assert!(!variants[0].is_split());
    }

    #[test]
    fn best_variant_idx_prefers_q4_k_m() {
        let v1 = GgufVariant {
            quant_name: "Q8_0".to_string(),
            files: vec![],
        };
        let v2 = GgufVariant {
            quant_name: "Q4_K_M".to_string(),
            files: vec![],
        };
        let refs = vec![&v1, &v2];
        assert_eq!(best_variant_idx(&refs), Some(1));
    }

    #[test]
    fn best_variant_idx_fallback_to_q4_k_s_when_no_q4_k_m() {
        let v1 = GgufVariant {
            quant_name: "Q2_K".to_string(),
            files: vec![],
        };
        let v2 = GgufVariant {
            quant_name: "Q4_K_S".to_string(),
            files: vec![],
        };
        let refs = vec![&v1, &v2];
        assert_eq!(best_variant_idx(&refs), Some(1));
    }

    #[test]
    fn best_variant_idx_returns_none_for_unrecognised_variants() {
        let v = GgufVariant {
            quant_name: "UNKNOWN_FORMAT".to_string(),
            files: vec![],
        };
        assert_eq!(best_variant_idx(&[&v]), None);
    }

    #[test]
    fn truncate_label_keeps_short_string() {
        assert_eq!(truncate_label("short", 20), "short");
    }

    #[test]
    fn truncate_label_truncates_long_string_with_ellipsis() {
        let label = truncate_label("this-is-a-very-long-filename.gguf", 20);
        assert_eq!(label.len(), 20);
        assert!(label.starts_with("..."));
    }

    #[test]
    fn fmt_size_formats_bytes_kb_mb_gb() {
        assert_eq!(fmt_size(512), "512B");
        assert_eq!(fmt_size(1024), "1KB");
        assert_eq!(fmt_size(1024 * 1024), "1.0MB");
        assert!(fmt_size(2 * 1024 * 1024 * 1024).contains("GB"));
    }

    #[test]
    fn is_incomplete_download_returns_true_for_missing_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("nonexistent");
        assert!(is_incomplete_download(&missing));
    }

    #[test]
    fn is_incomplete_download_returns_true_when_config_but_no_weights() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("config.json"), "{}").unwrap();
        assert!(is_incomplete_download(tmp.path()));
    }

    #[test]
    fn is_incomplete_download_returns_false_when_config_and_weights_present() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("config.json"), "{}").unwrap();
        std::fs::write(tmp.path().join("model.safetensors"), b"weights").unwrap();
        assert!(!is_incomplete_download(tmp.path()));
    }

    // Contract: a sharded safetensors dir is incomplete when a shard is missing
    // OR truncated (on-disk bytes < index total_size); complete otherwise.
    #[test]
    fn sharded_safetensors_truncated_shard_is_incomplete() {
        let tmp = tempfile::tempdir().unwrap();
        let index = serde_json::json!({
            "metadata": { "total_size": 16 },
            "weight_map": { "a": "model-00000-of-00002.safetensors",
                            "b": "model-00001-of-00002.safetensors" }
        });
        std::fs::write(
            tmp.path().join("model.safetensors.index.json"),
            index.to_string(),
        )
        .unwrap();

        // Missing second shard.
        std::fs::write(
            tmp.path().join("model-00000-of-00002.safetensors"),
            [0u8; 8],
        )
        .unwrap();
        assert!(is_incomplete_download(tmp.path()));

        // Both present but truncated (8 + 4 < 16).
        std::fs::write(
            tmp.path().join("model-00001-of-00002.safetensors"),
            [0u8; 4],
        )
        .unwrap();
        assert!(is_incomplete_download(tmp.path()));

        // Both present, full size.
        std::fs::write(
            tmp.path().join("model-00001-of-00002.safetensors"),
            [0u8; 8],
        )
        .unwrap();
        assert!(!is_incomplete_download(tmp.path()));
    }
}
