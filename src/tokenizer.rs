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
                                tracing::warn!(
                                    fallback_dir = %fallback_dir.display(),
                                    "GGUF tokenizer unsupported, using tokenizer.json fallback"
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

    /// Repairs the pre-tokenizer that candle built for a GGUF, for the
    /// `tokenizer.ggml.pre` kinds candle mishandles.
    ///
    /// candle maps unknown pre kinds to `ByteLevel::default()`, whose
    /// `add_prefix_space = true` injects a spurious `Ġ` at the start of every
    /// text segment that follows a special token, so chat markup like
    /// `<|start_of_role|>user` encodes the role name as `Ġuser` instead of
    /// `user`. Two families need repair:
    ///
    /// - `refact` / `starcoder` (bigcode: Granite 3.x, granite-code) and plain
    ///   `gpt-2`: the reference pipeline is GPT-2 ByteLevel with
    ///   `add_prefix_space = false`.
    /// - `llama-bpe` (Llama 3 family): llama.cpp writes `llama-bpe` but candle
    ///   only matches `llama3`, so these fall through to the broken default;
    ///   the reference pipeline splits with the Llama 3 regex (digit runs
    ///   capped at 3) and then byte-encodes without a prefix space.
    fn fix_gguf_pre_tokenizer(inner: &mut tokenizers::Tokenizer, pre: &str) -> Result<()> {
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;
        use tokenizers::pre_tokenizers::sequence::Sequence;
        use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
        use tokenizers::tokenizer::SplitDelimiterBehavior;

        const REGEX_LLAMA3: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

        match pre {
            "refact" | "starcoder" | "gpt-2" => {
                inner.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));
            }
            "llama-bpe" => {
                let split = Split::new(
                    SplitPattern::Regex(REGEX_LLAMA3.to_string()),
                    SplitDelimiterBehavior::Isolated,
                    false,
                )
                .map_err(|e| anyhow::anyhow!("llama-bpe split regex: {e}"))?;
                inner.with_pre_tokenizer(Some(Sequence::new(vec![
                    split.into(),
                    ByteLevel::new(false, true, false).into(),
                ])));
            }
            _ => {}
        }
        Ok(())
    }

    pub fn from_gguf_file(gguf_path: &str) -> Result<Self> {
        use candle_core::quantized::gguf_file;

        let mut file = std::fs::File::open(gguf_path)
            .with_context(|| format!("Failed to open GGUF file: {}", gguf_path))?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow::anyhow!("Failed to parse GGUF: {}", e))?;
        Self::from_gguf_content(&content)
    }

    fn from_gguf_content(content: &candle_core::quantized::gguf_file::Content) -> Result<Self> {
        use candle_core::quantized::tokenizer::TokenizerFromGguf;

        let mut inner = tokenizers::Tokenizer::from_gguf(content)
            .map_err(|e| anyhow::anyhow!("Failed to build tokenizer from GGUF: {}", e))?;

        let pre_kind = content
            .metadata
            .get("tokenizer.ggml.pre")
            .and_then(|v| v.to_string().ok())
            .cloned()
            .unwrap_or_default();
        Self::fix_gguf_pre_tokenizer(&mut inner, &pre_kind)?;

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

        tracing::info!(
            tokens = tokens_arr.len(),
            has_template = chat_template.is_some(),
            "tokenizer loaded from GGUF"
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
        self.special_token_ids
            .get(content)
            .copied()
            .or_else(|| self.inner.token_to_id(content))
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

    pub fn stop_text_markers(&self) -> Vec<String> {
        let mut markers: Vec<String> = Vec::new();

        for key in &["eot_token", "eom_token", "end_of_turn_token"] {
            if let Some(content) = self.special_tokens.get(*key)
                && !markers.contains(content)
            {
                markers.push(content.clone());
            }
        }

        for content in &[
            "<|eot_id|>",
            "<|eom_id|>",
            "<|end_of_text|>",
            "<|im_end|>",
            "<|im_start|>",
            "<|endoftext|>",
            "<turn|>",
            "<end_of_turn>",
        ] {
            if !markers.iter().any(|m| m == content) {
                markers.push((*content).to_string());
            }
        }

        markers
    }

    pub fn stop_token_sequences(&self) -> Vec<Vec<u32>> {
        fn push_sequence(tok: &Tokenizer, out: &mut Vec<Vec<u32>>, content: &str) {
            let Ok(ids) = tok.encode(content) else {
                return;
            };
            if ids.is_empty() {
                return;
            }
            if let Ok(roundtrip) = tok.decode_with_special(&ids)
                && roundtrip != content
            {
                return;
            }
            if !out.contains(&ids) {
                out.push(ids);
            }
        }

        let mut seqs: Vec<Vec<u32>> = Vec::new();

        for content in self.stop_text_markers() {
            push_sequence(self, &mut seqs, &content);
            let with_newline = format!("{}\n", content);
            push_sequence(self, &mut seqs, &with_newline);
        }

        seqs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    fn build_test_tokenizer(chat_template: Option<&str>, eos_token: Option<&str>) -> Tokenizer {
        let tmp = tempfile::tempdir().unwrap();

        let model = WordLevel::builder()
            .vocab(
                [
                    ("[UNK]".to_string(), 0u32),
                    ("hello".to_string(), 1u32),
                    ("world".to_string(), 2u32),
                    ("<eos>".to_string(), 3u32),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let mut inner = tokenizers::Tokenizer::new(model);
        inner.with_pre_tokenizer(Some(Whitespace {}));
        inner
            .save(tmp.path().join("tokenizer.json"), false)
            .unwrap();

        let mut cfg = serde_json::json!({});
        if let Some(tmpl) = chat_template {
            cfg["chat_template"] = tmpl.into();
        }
        if let Some(eos) = eos_token {
            cfg["eos_token"] = eos.into();
        }
        std::fs::write(
            tmp.path().join("tokenizer_config.json"),
            serde_json::to_string(&cfg).unwrap(),
        )
        .unwrap();

        Tokenizer::from_dir(tmp.path().to_str().unwrap()).unwrap()
    }

    #[test]
    fn encode_decode_roundtrip() {
        let tok = build_test_tokenizer(None, None);
        let ids = tok.encode("hello world").unwrap();
        assert_eq!(ids.len(), 2);
        let text = tok.decode(&ids).unwrap();
        assert_eq!(text, "hello world");
    }

    #[test]
    fn has_thinking_support_false_without_template() {
        let tok = build_test_tokenizer(None, None);
        assert!(!tok.has_thinking_support());
    }

    #[test]
    fn has_thinking_support_true_with_keyword_in_template() {
        let tok = build_test_tokenizer(Some("{% if enable_thinking %}think{% endif %}"), None);
        assert!(tok.has_thinking_support());
    }

    #[test]
    fn eos_token_id_resolves_from_vocab() {
        let tok = build_test_tokenizer(None, Some("<eos>"));
        assert_eq!(
            tok.eos_token_id(),
            Some(3),
            "<eos> is at index 3 in the test vocab"
        );
    }

    /// Contract: with the GGUF pre-tokenizer kinds candle mishandles
    /// (`refact`/`starcoder`/`gpt-2` and `llama-bpe`), text after a special
    /// token must NOT receive a spurious leading space: `<|x|>user` encodes
    /// the role name as `user`, never `Ġuser`. Guards the override of
    /// candle's `ByteLevel::default()` fallback, whose
    /// `add_prefix_space = true` corrupts chat markup.
    #[test]
    fn gguf_repaired_pre_tokenizers_have_no_prefix_space() {
        for pre in ["refact", "starcoder", "gpt-2", "llama-bpe"] {
            run_gguf_pre_tokenizer_case(pre);
        }
    }

    fn run_gguf_pre_tokenizer_case(pre: &str) {
        use candle_core::quantized::gguf_file::{Content, Value, VersionedMagic};
        use std::collections::HashMap;

        let tokens = [
            "<|x|>", "Ġ", "u", "s", "e", "r", "us", "er", "user", "Ġuser",
        ];
        // Types per llama_token_type: 3 = control (special), 1 = normal.
        let token_types = [3u32, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let merges = ["u s", "e r", "us er", "Ġ user"];

        let mut metadata: HashMap<String, Value> = HashMap::new();
        metadata.insert("tokenizer.ggml.model".into(), Value::String("gpt2".into()));
        metadata.insert("tokenizer.ggml.pre".into(), Value::String(pre.into()));
        metadata.insert(
            "tokenizer.ggml.tokens".into(),
            Value::Array(tokens.iter().map(|t| Value::String((*t).into())).collect()),
        );
        metadata.insert(
            "tokenizer.ggml.token_type".into(),
            Value::Array(token_types.iter().map(|&t| Value::U32(t)).collect()),
        );
        metadata.insert(
            "tokenizer.ggml.merges".into(),
            Value::Array(merges.iter().map(|m| Value::String((*m).into())).collect()),
        );

        let content = Content {
            magic: VersionedMagic::GgufV3,
            metadata,
            tensor_infos: HashMap::new(),
            tensor_data_offset: 0,
        };

        let tok = Tokenizer::from_gguf_content(&content).unwrap();
        let ids = tok.encode("<|x|>user").unwrap();
        let user_id = 8u32;
        let prefixed_user_id = 9u32;
        assert_eq!(
            ids,
            vec![0, user_id],
            "[{pre}] expected [<|x|>, user], got ids {ids:?}: a {prefixed_user_id} here means \
             the pre-tokenizer injected a spurious leading space"
        );
    }

    #[test]
    fn stop_token_ids_includes_eos_and_has_no_duplicates() {
        let tok = build_test_tokenizer(None, Some("<eos>"));
        let ids = tok.stop_token_ids();
        assert!(ids.contains(&3), "<eos> (id=3) must be a stop token");
        let unique: std::collections::HashSet<u32> = ids.iter().copied().collect();
        assert_eq!(
            unique.len(),
            ids.len(),
            "stop token ids must not contain duplicates"
        );
    }
}
