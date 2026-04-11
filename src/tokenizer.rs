use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    chat_template: Option<String>,
    special_tokens: HashMap<String, String>,
    special_token_ids: HashMap<String, u32>,
}

impl Tokenizer {
    pub fn from_dir(model_dir: &str) -> Result<Self> {
        let dir = Path::new(model_dir);

        let json_path = dir.join("tokenizer.json");
        if json_path.exists() {
            return Self::from_tokenizer_json(model_dir);
        }

        if let Some(gguf_path) = crate::models::loader::find_gguf_file(dir) {
            let gguf_path_str = gguf_path
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("Non-UTF8 GGUF path: {}", gguf_path.display()))?;

            match Self::from_gguf_file(gguf_path_str) {
                Ok(tok) => return Ok(tok),
                Err(gguf_err) => {
                    for fallback_dir in Self::gguf_fallback_tokenizer_dirs(dir) {
                        let Some(fallback_str) = fallback_dir.to_str() else {
                            continue;
                        };
                        match Self::from_tokenizer_json(fallback_str) {
                            Ok(tok) => {
                                eprintln!(
                                    "[gguf] GGUF tokenizer unsupported, using tokenizer.json fallback from '{}'",
                                    fallback_dir.display()
                                );
                                return Ok(tok);
                            }
                            Err(_) => continue,
                        }
                    }

                    return Err(gguf_err).with_context(|| {
                        format!(
                            "Failed to load tokenizer from GGUF in '{}', and no tokenizer.json fallback was usable",
                            dir.display()
                        )
                    });
                }
            }
        }

        anyhow::bail!(
            "No tokenizer found in '{}': expected tokenizer.json or a .gguf file",
            model_dir
        )
    }

    fn strip_gguf_suffix(name: &str) -> &str {
        let lower = name.to_ascii_lowercase();
        for suffix in ["-gguf", "_gguf", ".gguf", "gguf"] {
            if lower.ends_with(suffix) {
                let cut = name.len() - suffix.len();
                return &name[..cut];
            }
        }
        name
    }

    fn gguf_fallback_tokenizer_dirs(model_dir: &Path) -> Vec<std::path::PathBuf> {
        let mut exact = Vec::new();
        let mut related = Vec::new();

        let Some(parent) = model_dir.parent() else {
            return Vec::new();
        };

        let base_name = model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .map(Self::strip_gguf_suffix)
            .unwrap_or_default()
            .to_ascii_lowercase();

        let entries = match std::fs::read_dir(parent) {
            Ok(e) => e,
            Err(_) => return Vec::new(),
        };

        for entry in entries.flatten() {
            let p = entry.path();
            if p == model_dir || !p.is_dir() {
                continue;
            }
            if !p.join("tokenizer.json").exists() {
                continue;
            }

            let name = p
                .file_name()
                .and_then(|n| n.to_str())
                .map(|s| s.to_ascii_lowercase())
                .unwrap_or_default();

            if !base_name.is_empty() && name == base_name {
                exact.push(p);
            } else if !base_name.is_empty()
                && (name.starts_with(&base_name)
                    || base_name.starts_with(&name)
                    || (name.contains("phi-3") && base_name.contains("phi-3")))
            {
                related.push(p);
            }
        }

        exact.extend(related);
        exact
    }

    fn from_tokenizer_json(model_dir: &str) -> Result<Self> {
        let path = format!("{}/tokenizer.json", model_dir);
        let inner = tokenizers::Tokenizer::from_file(&path)
            .map_err(|e| anyhow::anyhow!("{}", e))
            .with_context(|| format!("Errore caricamento tokenizer da {}", path))?;

        let tokenizer_cfg_path = format!("{}/tokenizer_config.json", model_dir);
        let cfg: serde_json::Value = std::fs::read_to_string(&tokenizer_cfg_path)
            .ok()
            .and_then(|raw| serde_json::from_str(&raw).ok())
            .unwrap_or(serde_json::Value::Null);

        let parse_chat_template = |val: &serde_json::Value| -> Option<String> {
            match val {
                serde_json::Value::String(s) if !s.trim().is_empty() => Some(s.clone()),
                serde_json::Value::Array(arr) => arr
                    .iter()
                    .find(|v| v.get("name").and_then(|n| n.as_str()) == Some("default"))
                    .or_else(|| arr.first())
                    .and_then(|v| v.get("template").and_then(|t| t.as_str()))
                    .map(|s| s.to_string()),
                _ => None,
            }
        };

        let mut chat_template = cfg.get("chat_template").and_then(parse_chat_template);

        if chat_template.is_none() {
            let processor_cfg_path = format!("{}/processor_config.json", model_dir);
            let processor_cfg: serde_json::Value = std::fs::read_to_string(&processor_cfg_path)
                .ok()
                .and_then(|raw| serde_json::from_str(&raw).ok())
                .unwrap_or(serde_json::Value::Null);
            chat_template = processor_cfg
                .get("chat_template")
                .and_then(parse_chat_template);
        }

        if chat_template.is_none() {
            let template_path = format!("{}/chat_template.jinja", model_dir);
            if let Ok(raw) = std::fs::read_to_string(&template_path)
                && !raw.trim().is_empty()
            {
                chat_template = Some(raw);
            }
        }

        let mut special_tokens = HashMap::new();
        if let Some(obj) = cfg.as_object() {
            for (key, val) in obj {
                if !key.ends_with("_token") {
                    continue;
                }
                let s = match val {
                    serde_json::Value::String(s) => Some(s.clone()),
                    serde_json::Value::Object(o) => o
                        .get("content")
                        .and_then(|c| c.as_str())
                        .map(|s| s.to_string()),
                    _ => None,
                };
                if let Some(tok_str) = s
                    && !tok_str.is_empty()
                {
                    special_tokens.insert(key.clone(), tok_str);
                }
            }
        }

        let mut special_token_ids: HashMap<String, u32> = HashMap::new();
        if let Some(decoder) = cfg.get("added_tokens_decoder").and_then(|v| v.as_object()) {
            for (id_str, info) in decoder {
                let is_special = info
                    .get("special")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if is_special
                    && let (Ok(id), Some(content)) = (
                        id_str.parse::<u32>(),
                        info.get("content").and_then(|c| c.as_str()),
                    )
                {
                    special_token_ids.insert(content.to_string(), id);
                }
            }
        }

        for tok_str in special_tokens.values() {
            if !special_token_ids.contains_key(tok_str)
                && let Some(id) = inner.token_to_id(tok_str)
            {
                special_token_ids.insert(tok_str.clone(), id);
            }
        }

        Ok(Self {
            inner,
            chat_template,
            special_tokens,
            special_token_ids,
        })
    }

    pub fn from_gguf_file(gguf_path: &str) -> Result<Self> {
        use candle_core::quantized::{gguf_file, tokenizer::TokenizerFromGguf};

        let mut file = std::fs::File::open(gguf_path)
            .with_context(|| format!("Failed to open GGUF file: {}", gguf_path))?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow::anyhow!("Failed to parse GGUF: {}", e))?;

        let inner = tokenizers::Tokenizer::from_gguf(&content)
            .map_err(|e| anyhow::anyhow!("Failed to build tokenizer from GGUF: {}", e))?;

        let chat_template = content
            .metadata
            .get("tokenizer.chat_template")
            .and_then(|v| v.to_string().ok())
            .cloned();

        let mut special_tokens = HashMap::new();
        let mut special_token_ids: HashMap<String, u32> = HashMap::new();

        let tokens_arr: Vec<String> = content
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.to_vec().ok())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.to_string().ok().cloned())
                    .collect()
            })
            .unwrap_or_default();

        let get_token_str = |id: u32| -> Option<String> { tokens_arr.get(id as usize).cloned() };

        for (gguf_key, name) in [
            ("tokenizer.ggml.bos_token_id", "bos_token"),
            ("tokenizer.ggml.eos_token_id", "eos_token"),
            ("tokenizer.ggml.pad_token_id", "pad_token"),
            ("tokenizer.ggml.unk_token_id", "unk_token"),
        ] {
            if let Some(val) = content.metadata.get(gguf_key)
                && let Ok(id) = val.to_u32()
                && let Some(tok_str) = get_token_str(id)
            {
                special_tokens.insert(name.to_string(), tok_str.clone());
                special_token_ids.insert(tok_str, id);
            }
        }

        if let Some(type_arr) = content.metadata.get("tokenizer.ggml.token_type")
            && let Ok(arr) = type_arr.to_vec()
        {
            for (idx, v) in arr.iter().enumerate() {
                if let Ok(ty) = v.to_u32()
                    && matches!(ty, 2..=5)
                    && let Some(tok_str) = tokens_arr.get(idx)
                {
                    special_token_ids
                        .entry(tok_str.clone())
                        .or_insert(idx as u32);
                }
            }
        }

        println!(
            "[gguf] Tokenizer loaded from GGUF ({} tokens, template={})",
            tokens_arr.len(),
            if chat_template.is_some() { "yes" } else { "no" },
        );

        Ok(Self {
            inner,
            chat_template,
            special_tokens,
            special_token_ids,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("{}", e))
    }

    pub fn decode_with_special(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, false)
            .map_err(|e| anyhow::anyhow!("{}", e))
    }

    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    pub fn bos_token(&self) -> Option<&str> {
        self.special_tokens.get("bos_token").map(|s| s.as_str())
    }

    pub fn eos_token(&self) -> Option<&str> {
        self.special_tokens.get("eos_token").map(|s| s.as_str())
    }

    pub fn special_token_id(&self, content: &str) -> Option<u32> {
        self.special_token_ids.get(content).copied()
    }

    pub fn id_to_token(&self, token_id: u32) -> Option<String> {
        self.inner.id_to_token(token_id)
    }

    pub fn has_thinking_support(&self) -> bool {
        self.chat_template
            .as_deref()
            .map(|t| t.contains("enable_thinking"))
            .unwrap_or(false)
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token().and_then(|tok| self.special_token_id(tok))
    }

    pub fn stop_token_ids(&self) -> Vec<u32> {
        let mut ids = Vec::new();
        if let Some(id) = self.eos_token_id() {
            ids.push(id);
        }

        for key in &["eot_token", "eom_token", "end_of_turn_token"] {
            if let Some(content) = self.special_tokens.get(*key)
                && let Some(id) = self.special_token_id(content)
                && !ids.contains(&id)
            {
                ids.push(id);
            }
        }

        for content in &[
            "<|eot_id|>",
            "<|eom_id|>",
            "<|end_of_text|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<turn|>",
            "<end_of_turn>",
        ] {
            if let Some(id) = self.special_token_id(content)
                && !ids.contains(&id)
            {
                ids.push(id);
            }
        }
        ids
    }
}
