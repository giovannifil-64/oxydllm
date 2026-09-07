use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

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

/// Drops GGUF files that are not variants of the repository's model.
///
/// Quantization repositories ship companions next to the model: multimodal
/// projectors (`mmproj-BF16.gguf`), speculative-decoding modules, and whatever
/// comes next. They are valid GGUF and would otherwise be offered as choices,
/// so on unsloth/Qwen3.8-27B-GGUF the menu listed "BF16, 888 MB" for a 27B
/// model whose real BF16 is 55 GB, and picking it would download something that
/// cannot answer a request.
///
/// The rule names none of them: a repository's model variants share the model's
/// name, so the most common leading name wins and everything else is a
/// companion. Repositories with a single variant keep it either way.
fn keep_model_variants(files: &[(String, u64)]) -> Vec<(String, u64)> {
    let lead = |name: &str| -> String {
        name.rsplit('/')
            .next()
            .unwrap_or(name)
            .split('-')
            .next()
            .unwrap_or("")
            .to_lowercase()
    };

    let mut counts: Vec<(String, usize)> = Vec::new();
    for (name, _) in files {
        let l = lead(name);
        match counts.iter_mut().find(|(k, _)| *k == l) {
            Some(c) => c.1 += 1,
            None => counts.push((l, 1)),
        }
    }
    let Some((dominant, _)) = counts.into_iter().max_by_key(|(_, n)| *n) else {
        return files.to_vec();
    };
    let same_family: Vec<(String, u64)> = files
        .iter()
        .filter(|(name, _)| lead(name) == dominant)
        .cloned()
        .collect();

    // A variant carries a quantization in its name. Files that do not, like the
    // importance matrix bartowski ships as `<model>-imatrix.gguf`, share the
    // model's name and would otherwise be offered as a 10 MB choice. Only apply
    // this when some file does carry one, so a repository publishing a single
    // plain `model.gguf` keeps it.
    let labelled =
        |name: &str| crate::models::estimate::extract_quant_from_filename(name).is_some();
    if same_family.iter().any(|(n, _)| labelled(n)) {
        return same_family
            .into_iter()
            .filter(|(n, _)| labelled(n))
            .collect();
    }
    same_family
}

fn group_gguf_variants(gguf_files: &[(String, u64)]) -> Vec<GgufVariant> {
    let gguf_files = &keep_model_variants(gguf_files);
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

    println!("  Multiple GGUF variants available, choose one to download:\n");

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
        "  {:>2}  {:<16}  {:>10}  {:>5}  Quality",
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
    println!(
        "  Quality is typical for the label, not measured on this model; >= marks a suffix kept above its base. The chosen variant's header is read before download."
    );
    println!();

    let default = recommended_idx.map(|i| i + 1).unwrap_or(1);

    use std::io::IsTerminal;
    if !std::io::stdin().is_terminal() {
        println!(
            "  Non-interactive: selecting {} (#{}).",
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
    // on-disk bytes must cover the index's total_size; a bare existence check
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
    // The file bodies go through the asynchronous client because only its
    // builder has a read timeout, which is what turns a connection left
    // hanging by a sleep or a dropped network into an attempt to reconnect.
    let fetcher = reqwest::Client::builder()
        .read_timeout(Duration::from_secs(READ_STALL_SECS))
        .user_agent(concat!("oxydllm/", env!("CARGO_PKG_VERSION")))
        .build()?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
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

    let gguf_to_download: Vec<(String, u64)> = if gguf_files.is_empty() {
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
            let variant_files = chosen.files.clone();

            // The header states every tensor's quantization and sits at the
            // start of the file, so a range request settles in seconds what a
            // full download settles in half an hour. Publishers ship builds
            // that mix a handful of undecodable tensors into an otherwise
            // ordinary quantization, and refusing after 15 GB have arrived is
            // the worst moment to find out.
            if let Some((first, _)) = variant_files.first() {
                let url = format!("{}/{}/resolve/main/{}", HF_ENDPOINT, config.repo_id, first);
                let verdict =
                    crate::models::gguf_probe::probe_remote(&client, &url, config.token.as_deref());
                if let Some(line) = verdict.composition_line() {
                    println!("  Header: {line}");
                }
                if let Some(reason) = verdict.refusal() {
                    anyhow::bail!("{} cannot be used: {reason}", chosen.quant_name);
                }
            }

            // A file already there is picked up where it stopped, unless the
            // caller asked to start over.
            if dest.exists() && config.force {
                for (f, _) in &chosen.files {
                    let _ = std::fs::remove_file(dest.join(f));
                }
            }
            if dest.exists() {
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
            variant_files
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
            // A file shorter than upstream is picked up where it stopped; one
            // longer than upstream is not a prefix of anything and starts over.
            println!("Resuming interrupted download...");
            for (name, size) in &metadata_files {
                let p = dest.join(name);
                if *size > 0 && p.exists() && p.metadata().map(|m| m.len()).unwrap_or(0) > *size {
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

    let complete = |name: &str, size: u64| -> bool {
        let p = dest.join(name);
        p.exists() && (size == 0 || p.metadata().map(|m| m.len()).unwrap_or(0) == size)
    };
    let mut to_download: Vec<(String, u64)> = metadata_files
        .into_iter()
        .filter(|(f, size)| !complete(f, *size))
        .collect();
    to_download.extend(gguf_to_download.iter().cloned());
    to_download.sort_by_key(|(f, _)| if f.ends_with(".json") { 0u8 } else { 1u8 });

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
    for (filename, size) in &to_download {
        let url = format!(
            "{}/{}/resolve/main/{}",
            HF_ENDPOINT, config.repo_id, filename
        );
        match download_file(
            &runtime,
            &fetcher,
            &url,
            &config.repo_id,
            &dest.join(filename),
            (*size > 0).then_some(*size),
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

    if failed_file.is_some() {
        // Every file stays as it is, the interrupted one included: the next
        // run picks it up where it stopped.
        anyhow::bail!(
            "Download incomplete ({} of {} files done). Re-run the same pull to resume from where it stopped.",
            downloaded_files.len(),
            to_download.len()
        );
    }

    let new_shards: Vec<&str> = gguf_to_download
        .iter()
        .filter(|(f, _)| f.to_lowercase().ends_with(".gguf"))
        .map(|(f, _)| f.as_str())
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

/// Connection attempts per file. A body stops arriving when the machine
/// sleeps or the network drops; each attempt asks for the bytes past those
/// already on disk, so nothing already fetched is fetched twice.
const DOWNLOAD_ATTEMPTS: usize = 5;

/// Seconds without a byte after which the connection is given up and the
/// next attempt made.
const READ_STALL_SECS: u64 = 60;

/// Pause between attempts.
const RETRY_PAUSE: Duration = Duration::from_secs(2);

/// Why a fetch stopped: something a fresh connection can fix, or not.
enum FetchError {
    Transient(anyhow::Error),
    Fatal(anyhow::Error),
}

/// Fetches `url` into `dest_path`, resuming from the bytes already there.
///
/// A file whose length already equals `expected` is not requested at all.
/// Otherwise the request asks for the range past the file's end: a `206`
/// appends, a `200` from a server that ignored the range starts the file
/// over, and a `416` means the server holds nothing past that offset, so the
/// file is complete if its length is upstream's and wrong otherwise. A body
/// that stops short or a connection that goes quiet for [`READ_STALL_SECS`]
/// costs an attempt, not the bytes on disk.
///
/// ## Errors
/// Fails on an HTTP status that a retry cannot change, or when
/// [`DOWNLOAD_ATTEMPTS`] connections all stopped short.
fn download_file(
    runtime: &tokio::runtime::Runtime,
    client: &reqwest::Client,
    url: &str,
    repo_id: &str,
    dest_path: &Path,
    expected: Option<u64>,
    token: Option<&str>,
) -> anyhow::Result<()> {
    let label = truncate_label(
        &dest_path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        40,
    );
    let mut attempt = 1;
    loop {
        let have = dest_path.metadata().map(|m| m.len()).unwrap_or(0);
        if have > 0 && expected == Some(have) {
            print_progress(&label, have, expected, true);
            println!();
            return Ok(());
        }
        match runtime.block_on(fetch_from(
            client, url, repo_id, dest_path, have, expected, token, &label,
        )) {
            Ok(()) => return Ok(()),
            Err(FetchError::Fatal(e)) => return Err(e),
            Err(FetchError::Transient(e)) if attempt < DOWNLOAD_ATTEMPTS => {
                attempt += 1;
                println!();
                println!("  {label}: {e}; reconnecting ({attempt}/{DOWNLOAD_ATTEMPTS})");
                std::thread::sleep(RETRY_PAUSE);
            }
            Err(FetchError::Transient(e)) => {
                return Err(e.context(format!(
                    "{DOWNLOAD_ATTEMPTS} connections stopped short; what arrived is kept for the next run"
                )));
            }
        }
    }
}

/// The `total` of a `Content-Range: bytes a-b/total` header.
fn content_range_total(resp: &reqwest::Response) -> Option<u64> {
    resp.headers()
        .get(reqwest::header::CONTENT_RANGE)?
        .to_str()
        .ok()?
        .rsplit('/')
        .next()?
        .parse()
        .ok()
}

#[allow(clippy::too_many_arguments)]
async fn fetch_from(
    client: &reqwest::Client,
    url: &str,
    repo_id: &str,
    dest_path: &Path,
    have: u64,
    expected: Option<u64>,
    token: Option<&str>,
    label: &str,
) -> Result<(), FetchError> {
    let mut builder = client.get(url);
    if let Some(tok) = token {
        builder = builder.bearer_auth(tok);
    }
    if have > 0 {
        builder = builder.header(reqwest::header::RANGE, format!("bytes={have}-"));
    }
    let mut resp = builder
        .send()
        .await
        .map_err(|e| FetchError::Transient(e.into()))?;
    let status = resp.status().as_u16();
    let (mut out, mut downloaded, total) = match status {
        206 => (
            std::fs::OpenOptions::new()
                .append(true)
                .open(dest_path)
                .map_err(|e| FetchError::Fatal(e.into()))?,
            have,
            content_range_total(&resp).or(expected),
        ),
        416 => {
            let total = content_range_total(&resp);
            if total == Some(have) || (total.is_none() && expected == Some(have)) {
                print_progress(label, have, Some(have), true);
                println!();
                return Ok(());
            }
            let _ = std::fs::remove_file(dest_path);
            return Err(FetchError::Transient(anyhow::anyhow!(
                "upstream holds {} bytes and the file on disk {have}; starting it over",
                total.map_or_else(|| "?".to_string(), |t| t.to_string())
            )));
        }
        200 => (
            std::fs::File::create(dest_path).map_err(|e| FetchError::Fatal(e.into()))?,
            0,
            resp.content_length().or(expected),
        ),
        other => return Err(FetchError::Fatal(check_status(other, repo_id).unwrap_err())),
    };

    let mut last_tick = Instant::now();
    loop {
        match resp.chunk().await {
            Ok(Some(chunk)) => {
                out.write_all(&chunk)
                    .map_err(|e| FetchError::Fatal(e.into()))?;
                downloaded += chunk.len() as u64;
                if last_tick.elapsed().as_millis() >= 100 {
                    print_progress(label, downloaded, total, false);
                    last_tick = Instant::now();
                }
            }
            Ok(None) => break,
            Err(e) => {
                out.flush().ok();
                return Err(FetchError::Transient(anyhow::anyhow!(
                    "connection lost at {} of {}: {e}",
                    fmt_size(downloaded),
                    total.map_or_else(|| "?".to_string(), fmt_size)
                )));
            }
        }
    }
    out.flush().map_err(|e| FetchError::Fatal(e.into()))?;
    if let Some(t) = total
        && downloaded != t
    {
        return Err(FetchError::Transient(anyhow::anyhow!(
            "body ended at {} of {}",
            fmt_size(downloaded),
            fmt_size(t)
        )));
    }
    print_progress(label, downloaded, total, true);
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
            "Access denied to '{repo_id}': this model requires accepting a license.\n\
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

    /// Contract: a repository's companions are not offered as models. On
    /// unsloth/Qwen3.8-27B-GGUF the projectors are valid GGUF a tenth the size
    /// of the smallest real variant, and picking one downloads something that
    /// cannot answer a request.
    #[test]
    fn companion_files_are_not_offered_as_variants() {
        let files = vec![
            ("Qwen3.8-27B-UD-Q4_K_M.gguf".to_string(), 15_000u64),
            ("Qwen3.8-27B-UD-Q6_K_XL.gguf".to_string(), 22_000u64),
            ("Qwen3.8-27B-Q4_0.gguf".to_string(), 14_000u64),
            ("mmproj-BF16.gguf".to_string(), 888u64),
            ("mmproj-F16.gguf".to_string(), 885u64),
        ];
        let names: Vec<String> = group_gguf_variants(&files)
            .into_iter()
            .map(|v| v.quant_name)
            .collect();
        assert!(
            !names.iter().any(|n| n == "BF16" || n == "F16"),
            "{names:?}"
        );
        assert_eq!(names.len(), 3, "{names:?}");
    }

    /// Contract: files without a quantization in their name are not variants.
    /// bartowski ships `<model>-imatrix.gguf`, which shares the model's name
    /// and is 10 MB, and it used to be offered as something downloadable.
    #[test]
    fn unlabelled_files_are_not_offered_as_variants() {
        let files = vec![
            ("Qwen3.8-27B-Q2_K.gguf".to_string(), 11_000u64),
            ("Qwen3.8-27B-Q3_K_M.gguf".to_string(), 13_000u64),
            ("Qwen3.8-27B-imatrix.gguf".to_string(), 10u64),
        ];
        let names: Vec<String> = group_gguf_variants(&files)
            .into_iter()
            .map(|v| v.quant_name)
            .collect();
        assert_eq!(names.len(), 2, "{names:?}");
        assert!(!names.iter().any(|n| n.contains("imatrix")), "{names:?}");
    }

    /// Contract: a repository whose only file carries no label keeps it, so the
    /// rule cannot empty a listing.
    #[test]
    fn an_unlabelled_lone_file_survives() {
        let files = vec![("model.gguf".to_string(), 500u64)];
        assert_eq!(group_gguf_variants(&files).len(), 1);
    }

    /// Contract: a repository with only one family keeps it, whatever it is
    /// named, so the filter cannot empty a listing.
    #[test]
    fn a_single_family_survives_the_companion_filter() {
        let files = vec![("mmproj-BF16.gguf".to_string(), 888u64)];
        assert_eq!(group_gguf_variants(&files).len(), 1);
    }

    /// Contract: suffixes publishers invent survive instead of being truncated
    /// onto an existing label. `Q6_K`, `Q6_K_M` and `Q6_K_XL` ship side by side
    /// and used to collapse into three entries all called `Q6_K`.
    #[test]
    fn invented_quant_suffixes_stay_distinct() {
        let files = vec![
            ("m-UD-Q6_K.gguf".to_string(), 1u64),
            ("m-UD-Q6_K_M.gguf".to_string(), 2u64),
            ("m-UD-Q6_K_XL.gguf".to_string(), 3u64),
            ("m-UD-Q8_K_XL.gguf".to_string(), 4u64),
        ];
        let mut names: Vec<String> = group_gguf_variants(&files)
            .into_iter()
            .map(|v| v.quant_name)
            .collect();
        names.sort();
        assert_eq!(names, ["Q6_K", "Q6_K_M", "Q6_K_XL", "Q8_K_XL"]);
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

    /// A one-file HTTP server for the download contracts: serves `body` at
    /// any path, honouring `Range` unless told to ignore it, and once closes
    /// the connection after `cut_once` bytes of body. Records the offset each
    /// request asked to start from.
    fn serve(
        body: Vec<u8>,
        honour_range: bool,
        cut_once: Option<usize>,
    ) -> (String, std::sync::Arc<std::sync::Mutex<Vec<u64>>>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/model.gguf", listener.local_addr().unwrap());
        let asked = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen = asked.clone();
        std::thread::spawn(move || {
            let mut cut = cut_once;
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else {
                    break;
                };
                let mut reader = std::io::BufReader::new(stream.try_clone().unwrap());
                let mut start = 0u64;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                        break;
                    }
                    if let Some(v) = line.to_ascii_lowercase().strip_prefix("range: bytes=") {
                        start = v.trim().trim_end_matches('-').parse().unwrap();
                    }
                }
                seen.lock().unwrap().push(start);
                let len = body.len() as u64;
                let (head, payload): (String, &[u8]) = if !honour_range || start == 0 {
                    (
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n"
                        ),
                        &body[..],
                    )
                } else if start >= len {
                    (
                        format!(
                            "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Range: bytes */{len}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                        ),
                        &[],
                    )
                } else {
                    (
                        format!(
                            "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {start}-{}/{len}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            len - 1,
                            len - start
                        ),
                        &body[start as usize..],
                    )
                };
                stream.write_all(head.as_bytes()).unwrap();
                let n = match cut.take() {
                    Some(c) if c < payload.len() => c,
                    _ => payload.len(),
                };
                stream.write_all(&payload[..n]).unwrap();
                stream.flush().ok();
            }
        });
        (url, asked)
    }

    fn fetch(url: &str, dest: &Path, expected: Option<u64>) -> anyhow::Result<()> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let client = reqwest::Client::builder()
            .no_proxy()
            .read_timeout(Duration::from_secs(5))
            .build()
            .unwrap();
        download_file(&runtime, &client, url, "test/repo", dest, expected, None)
    }

    fn body() -> Vec<u8> {
        (0..5000u32).map(|i| (i * 7 % 251) as u8).collect()
    }

    /// Contract: a file left half done is completed by asking upstream only
    /// for the bytes past its end, so a pull that stopped resumes where it
    /// stopped instead of starting over.
    #[test]
    fn a_partial_file_is_completed_from_where_it_stopped() {
        let body = body();
        let (url, asked) = serve(body.clone(), true, None);
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("model.gguf");
        std::fs::write(&dest, &body[..1000]).unwrap();
        fetch(&url, &dest, Some(5000)).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), body);
        assert_eq!(*asked.lock().unwrap(), vec![1000]);
    }

    /// Contract: a file already of upstream's length is not requested at all.
    #[test]
    fn a_complete_file_is_not_fetched_again() {
        let body = body();
        let (url, asked) = serve(body.clone(), true, None);
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("model.gguf");
        std::fs::write(&dest, &body).unwrap();
        fetch(&url, &dest, Some(5000)).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), body);
        assert!(asked.lock().unwrap().is_empty());
    }

    /// Contract: a server that answers a range request with the whole file
    /// gets the file written from the start, not appended to.
    #[test]
    fn a_server_that_ignores_the_range_starts_the_file_over() {
        let body = body();
        let (url, asked) = serve(body.clone(), false, None);
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("model.gguf");
        std::fs::write(&dest, &body[..1000]).unwrap();
        fetch(&url, &dest, Some(5000)).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), body);
        assert_eq!(*asked.lock().unwrap(), vec![1000]);
    }

    /// Contract: a connection that drops mid-body, which is what a sleeping
    /// machine or a failing network looks like, is reopened for the bytes
    /// still missing and the file comes out whole.
    #[test]
    fn a_dropped_connection_is_picked_up_where_it_stopped() {
        let body = body();
        let (url, asked) = serve(body.clone(), true, Some(2000));
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("model.gguf");
        fetch(&url, &dest, Some(5000)).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), body);
        assert_eq!(*asked.lock().unwrap(), vec![0, 2000]);
    }

    /// Contract: a file longer than upstream's is no prefix of it and is
    /// fetched from the start rather than trusted.
    #[test]
    fn a_file_longer_than_upstream_starts_over() {
        let body = body();
        let (url, asked) = serve(body.clone(), true, None);
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("model.gguf");
        let mut longer = body.clone();
        longer.extend_from_slice(&[9u8; 100]);
        std::fs::write(&dest, &longer).unwrap();
        fetch(&url, &dest, Some(5000)).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), body);
        assert_eq!(*asked.lock().unwrap(), vec![5100, 0]);
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
