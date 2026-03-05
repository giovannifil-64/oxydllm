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
            return Self::from_gguf_file(gguf_path.to_str().unwrap());
        }

        anyhow::bail!(
            "No tokenizer found in '{}': expected tokenizer.json or a .gguf file",
            model_dir
        )
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

        let chat_template = match cfg.get("chat_template") {
            Some(serde_json::Value::String(s)) if !s.trim().is_empty() => Some(s.clone()),
            Some(serde_json::Value::Array(arr)) => {
                arr.iter()
                    .find(|v| v.get("name").and_then(|n| n.as_str()) == Some("default"))
                    .or_else(|| arr.first())
                    .and_then(|v| v.get("template").and_then(|t| t.as_str()))
                    .map(|s| s.to_string())
            }
            _ => None,
        };

        let mut special_tokens = HashMap::new();
        for key in &["bos_token", "eos_token", "pad_token", "unk_token"] {
            if let Some(val) = cfg.get(*key) {
                let s = match val {
                    serde_json::Value::String(s) => Some(s.clone()),
                    serde_json::Value::Object(obj) => {
                        obj.get("content").and_then(|c| c.as_str()).map(|s| s.to_string())
                    }
                    _ => None,
                };
                if let Some(tok_str) = s {
                    special_tokens.insert(key.to_string(), tok_str);
                }
            }
        }

        let mut special_token_ids: HashMap<String, u32> = HashMap::new();
        if let Some(decoder) = cfg.get("added_tokens_decoder").and_then(|v| v.as_object()) {
            for (id_str, info) in decoder {
                let is_special = info.get("special").and_then(|v| v.as_bool()).unwrap_or(false);
                if is_special
                    && let (Ok(id), Some(content)) = (
                        id_str.parse::<u32>(),
                        info.get("content").and_then(|c| c.as_str()),
                    ) {
                        special_token_ids.insert(content.to_string(), id);
                    }
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
            .and_then(|v| v.to_string().ok()).cloned();

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

        let get_token_str = |id: u32| -> Option<String> {
            tokens_arr.get(id as usize).cloned()
        };

        for (gguf_key, name) in [
            ("tokenizer.ggml.bos_token_id", "bos_token"),
            ("tokenizer.ggml.eos_token_id", "eos_token"),
            ("tokenizer.ggml.pad_token_id", "pad_token"),
            ("tokenizer.ggml.unk_token_id", "unk_token"),
        ] {
            if let Some(val) = content.metadata.get(gguf_key)
                && let Ok(id) = val.to_u32()
                    && let Some(tok_str) = get_token_str(id) {
                        special_tokens.insert(name.to_string(), tok_str.clone());
                        special_token_ids.insert(tok_str, id);
                    }
        }

        if let Some(type_arr) = content.metadata.get("tokenizer.ggml.token_type")
            && let Ok(arr) = type_arr.to_vec() {
                for (idx, v) in arr.iter().enumerate() {
                    if let Ok(ty) = v.to_u32()
                        && matches!(ty, 2..=5)
                            && let Some(tok_str) = tokens_arr.get(idx) {
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

    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token().and_then(|tok| self.special_token_id(tok))
    }

    pub fn stop_token_ids(&self) -> Vec<u32> {
        let mut ids = Vec::new();
        if let Some(id) = self.eos_token_id() {
            ids.push(id);
        }
        for content in &[
            "<|eot_id|>",
            "<|eom_id|>",
            "<|end_of_text|>",
            "<|im_end|>",
            "<|endoftext|>",
        ] {
            if let Some(id) = self.special_token_id(content)
                && !ids.contains(&id) {
                    ids.push(id);
                }
        }
        ids
    }
}
