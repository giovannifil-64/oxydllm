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

    pub fn from_gguf_file(gguf_path: &str) -> Result<Self> {
        let header = crate::common::gguf_header::GgufHeader::read(gguf_path)
            .map_err(|e| anyhow::anyhow!("Failed to parse GGUF: {e}"))?;
        Self::from_gguf_content(&header.content)
    }

    fn from_gguf_content(content: &candle_core::quantized::gguf_file::Content) -> Result<Self> {
        let inner = gguf::build_tokenizer(content)?;

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

    /// The full vocabulary size, added tokens included.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// The raw vocabulary form of a token id (byte-level BPE alphabet for
    /// BPE tokenizers), for building constrained-decoding byte tables.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// The ids of every registered special token.
    pub fn special_token_id_values(&self) -> Vec<u32> {
        self.special_token_ids.values().copied().collect()
    }

    /// Encodes with the tokenizer's special-token template applied (e.g.
    /// `<s> ... </s>` for the BERT/RoBERTa encoder models, whose embeddings
    /// are computed over the full templated sequence).
    pub fn encode_with_special_tokens(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, true)
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

    /// Whether the model can be asked to reason: its template takes
    /// `enable_thinking`, or it speaks harmony, where the analysis channel is
    /// the reasoning and `enable_thinking` decides whether the reply carries it.
    pub fn has_thinking_support(&self) -> bool {
        self.chat_template
            .as_deref()
            .is_some_and(|t| t.contains("enable_thinking"))
            || self.speaks_harmony()
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token().and_then(|tok| self.special_token_id(tok))
    }

    /// Whether the vocabulary speaks the harmony protocol, whose `<|end|>`
    /// closes one message of the turn and not the turn, which `<|return|>`
    /// and `<|call|>` do; gpt-oss stopped after its analysis message and
    /// answered nothing once `<|end|>` was a stop for Phi-3's sake.
    fn speaks_harmony(&self) -> bool {
        self.special_token_id("<|channel|>").is_some()
            && self.special_token_id("<|message|>").is_some()
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
            "<|end|>",
            "<turn|>",
            "<end_of_turn>",
        ] {
            if *content == "<|end|>" && self.speaks_harmony() {
                continue;
            }
            if let Some(id) = self.special_token_id(content)
                && !ids.contains(&id)
            {
                ids.push(id);
            }
        }
        if self.speaks_harmony()
            && let Some(id) = self.special_token_id("<|call|>")
            && !ids.contains(&id)
        {
            ids.push(id);
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
            "<|end|>",
            "<turn|>",
            "<end_of_turn>",
        ] {
            if *content == "<|end|>" && self.speaks_harmony() {
                continue;
            }
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

/// Builds a [`tokenizers::Tokenizer`] from GGUF metadata.
///
/// Ported from candle-core 0.10.2 `quantized::tokenizer` (Apache-2.0/MIT):
/// candle 0.11 pins tokenizers 0.22, so its `TokenizerFromGguf` impl targets
/// a different `Tokenizer` type than the 0.23 crate this project ships.
/// Owning the builder also lets the pre-tokenizer table cover the
/// `tokenizer.ggml.pre` kinds candle mishandles: its fallback is
/// `ByteLevel::default()`, whose `add_prefix_space = true` injects a spurious
/// `Ġ` at the start of every text segment that follows a special token, so
/// chat markup like `<|start_of_role|>user` encodes the role name as `Ġuser`
/// instead of `user`.
mod gguf {
    use anyhow::{Context, Result};
    use candle_core::quantized::gguf_file;
    use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
    use tokenizers::pre_tokenizers::{byte_level::ByteLevel as ByteLevelPre, sequence::Sequence};
    use tokenizers::tokenizer::SplitDelimiterBehavior;
    use tokenizers::{
        AddedToken, Tokenizer,
        decoders::{
            DecoderWrapper, byte_fallback::ByteFallback, byte_level::ByteLevel as ByteLevelDecoder,
            fuse::Fuse, sequence::Sequence as DecoderSequence, strip::Strip,
        },
        models::bpe::{BPE, Vocab},
        normalizers::{
            NormalizerWrapper, Sequence as NormalizerSequence, prepend::Prepend, replace::Replace,
            unicode::NFC,
        },
        pre_tokenizers::PreTokenizerWrapper,
        processors::sequence::Sequence as ProcessorSequence,
        processors::{PostProcessorWrapper, byte_level::ByteLevel as ByteLevelProcessor},
    };

    fn metadata_value<'a>(ct: &'a gguf_file::Content, key: &str) -> Result<&'a gguf_file::Value> {
        ct.metadata
            .get(key)
            .with_context(|| format!("missing GGUF metadata key `{key}`"))
    }

    fn value_to_u32(v: &gguf_file::Value) -> Result<u32> {
        use gguf_file::Value::*;
        match v {
            U8(v) => Ok(*v as u32),
            I8(v) => Ok(*v as u32),
            U16(v) => Ok(*v as u32),
            I16(v) => Ok(*v as u32),
            U32(v) => Ok(*v),
            I32(v) => Ok(*v as u32),
            U64(v) => Ok(*v as u32),
            I64(v) => Ok(*v as u32),
            _ => anyhow::bail!("expected numeric value for token type/id, got {v:?}"),
        }
    }

    fn value_to_string_array(v: &gguf_file::Value, name: &str) -> Result<Vec<String>> {
        let arr = v
            .to_vec()
            .map_err(|e| anyhow::anyhow!("`{name}` is not an array: {e}"))?;
        arr.iter()
            .map(|v| {
                v.to_string()
                    .map(|s| s.to_string())
                    .map_err(|e| anyhow::anyhow!("`{name}` element is not a string: {e}"))
            })
            .collect()
    }

    fn value_to_f32_array(v: &gguf_file::Value, name: &str) -> Result<Vec<f32>> {
        let arr = v
            .to_vec()
            .map_err(|e| anyhow::anyhow!("`{name}` is not an array: {e}"))?;
        arr.iter()
            .map(|v| {
                v.to_f32()
                    .map_err(|e| anyhow::anyhow!("`{name}` element is not a float: {e}"))
            })
            .collect()
    }

    /// The merge list a SentencePiece BPE vocabulary implies: every piece
    /// that splits into two pieces of the vocabulary is a merge, ordered by
    /// the merged piece's score from the highest down, which is the order the
    /// reference conversion (`SentencePieceExtractor`) writes into a
    /// `tokenizer.json`, so the ids come out as SentencePiece's own. Picking
    /// the pieces by maximum score sum instead, as a unigram model does,
    /// split the official Phi-3 file's `Hello` into six pieces.
    fn merges_from_scores(tokens: &[String], scores: &[f32]) -> Vec<(String, String)> {
        let ids: std::collections::HashMap<&str, usize> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.as_str(), i))
            .collect();
        let mut merges: Vec<((String, String), f32)> = Vec::new();
        for (piece, &score) in tokens.iter().zip(scores) {
            let mut local: Vec<(usize, usize, usize)> = piece
                .char_indices()
                .skip(1)
                .filter_map(|(at, _)| {
                    let (l, r) = piece.split_at(at);
                    Some((*ids.get(l)?, *ids.get(r)?, at))
                })
                .collect();
            local.sort_unstable();
            merges.extend(local.into_iter().map(|(_, _, at)| {
                let (l, r) = piece.split_at(at);
                ((l.to_string(), r.to_string()), score)
            }));
        }
        merges.sort_by(|a, b| b.1.total_cmp(&a.1));
        merges.into_iter().map(|(pair, _)| pair).collect()
    }

    fn merges_from_value(v: &gguf_file::Value) -> Result<Vec<(String, String)>> {
        value_to_string_array(v, "tokenizer.ggml.merges")?
            .into_iter()
            .map(|m| {
                m.split_once(' ')
                    .map(|(a, b)| (a.to_string(), b.to_string()))
                    .ok_or_else(|| anyhow::anyhow!("invalid merge entry `{m}`"))
            })
            .collect()
    }

    struct Pipeline {
        normalizer: Option<NormalizerWrapper>,
        pretokenizer: Option<PreTokenizerWrapper>,
        decoder: Option<DecoderWrapper>,
        post_processor: Option<PostProcessorWrapper>,
    }

    fn pre_tokenizer_sequence(
        regex: &str,
        byte_level: ByteLevelPre,
    ) -> Result<PreTokenizerWrapper> {
        let split = Split::new(
            SplitPattern::Regex(regex.to_string()),
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .map_err(|e| anyhow::anyhow!("pre-tokenizer split regex: {e}"))?;
        Ok(Sequence::new(vec![split.into(), byte_level.into()]).into())
    }

    /// The pipeline a SentencePiece-style BPE needs: spaces become `\u{2581}`
    /// on the way in and come back on the way out, with byte fallback for what
    /// the vocabulary does not cover.
    ///
    /// Taken from the canonical `tokenizer.json` of the checkpoints that use it
    /// rather than from memory, including the pre-tokenizer that splits on a
    /// space the normalizer has already replaced: it does nothing, and leaving
    /// it out is a difference nobody would notice until it mattered.
    /// The SentencePiece pipeline: a space is spelled U+2581, and with
    /// `add_space_prefix` one is put in front of the text, as SentencePiece's
    /// dummy prefix does, so the first word is the same piece it would be in
    /// the middle of a sentence; the decoder takes that one space back.
    fn sentencepiece_pipeline(add_space_prefix: bool) -> Result<Pipeline> {
        let replace = Replace::new(" ", "\u{2581}")
            .map_err(|e| anyhow::anyhow!("sentencepiece normalizer: {e}"))?;
        let normalizer: NormalizerWrapper = if add_space_prefix {
            NormalizerSequence::new(vec![
                Prepend::new("\u{2581}".to_string()).into(),
                replace.into(),
            ])
            .into()
        } else {
            replace.into()
        };
        let replace_back = tokenizers::normalizers::replace::Replace::new("\u{2581}", " ")
            .map_err(|e| anyhow::anyhow!("sentencepiece decoder: {e}"))?;
        let mut decoders = vec![
            DecoderWrapper::Replace(replace_back),
            DecoderWrapper::ByteFallback(ByteFallback::default()),
            DecoderWrapper::Fuse(Fuse::default()),
        ];
        if add_space_prefix {
            decoders.push(DecoderWrapper::Strip(Strip::new(' ', 1, 0)));
        }
        Ok(Pipeline {
            normalizer: Some(normalizer),
            pretokenizer: Some(
                Split::new(
                    SplitPattern::String(" ".to_string()),
                    SplitDelimiterBehavior::MergedWithPrevious,
                    false,
                )
                .map_err(|e| anyhow::anyhow!("sentencepiece pre-tokenizer: {e}"))?
                .into(),
            ),
            decoder: Some(DecoderWrapper::Sequence(DecoderSequence::new(decoders))),
            post_processor: None,
        })
    }

    fn pipeline_from_pre(pre: &str) -> Result<Pipeline> {
        const REGEX_QWEN2: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        const REGEX_LLAMA3: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

        Ok(match pre {
            // Matches Qwen2 tokenizer.json settings.
            "qwen2" => Pipeline {
                normalizer: Some(NFC.into()),
                pretokenizer: Some(pre_tokenizer_sequence(
                    REGEX_QWEN2,
                    ByteLevelPre::new(false, false, false),
                )?),
                decoder: Some(ByteLevelDecoder::new(false, false, false).into()),
                post_processor: Some(ByteLevelProcessor::new(false, false, false).into()),
            },
            // Llama 3 style byte-level BPE (digit runs capped at 3);
            // llama.cpp writes "llama-bpe" for the Llama 3 family.
            "smaug-bpe" | "lfm2" | "llama3" | "llama-bpe" => Pipeline {
                normalizer: None,
                pretokenizer: Some(pre_tokenizer_sequence(
                    REGEX_LLAMA3,
                    ByteLevelPre::new(false, true, false),
                )?),
                decoder: Some(ByteLevelDecoder::new(true, true, true).into()),
                post_processor: Some(ByteLevelProcessor::new(true, false, true).into()),
            },
            // bigcode family (Granite 3.x publishes "refact", granite-code
            // "starcoder") and plain GPT-2: ByteLevel WITHOUT the prefix
            // space, matching the reference tokenizer.json.
            "refact" | "starcoder" | "gpt-2" => Pipeline {
                normalizer: None,
                pretokenizer: Some(ByteLevelPre::new(false, true, true).into()),
                decoder: Some(ByteLevelDecoder::default().into()),
                post_processor: Some(ByteLevelProcessor::default().into()),
            },
            // Default GPT-2 style BPE.
            _ => Pipeline {
                normalizer: None,
                pretokenizer: Some(ByteLevelPre::default().into()),
                decoder: Some(ByteLevelDecoder::default().into()),
                post_processor: Some(ByteLevelProcessor::default().into()),
            },
        })
    }

    fn template_processor(
        tokens: &[String],
        bos_id: Option<u32>,
        eos_id: Option<u32>,
        add_bos: bool,
        add_eos: bool,
    ) -> Option<PostProcessorWrapper> {
        if (!add_bos && !add_eos) || tokens.is_empty() {
            return None;
        }

        let bos = bos_id.and_then(|id| tokens.get(id as usize)).cloned();
        let eos = eos_id.and_then(|id| tokens.get(id as usize)).cloned();

        let mut specials = Vec::new();
        if add_bos {
            specials.push((bos.clone()?, bos_id?));
        }
        if add_eos {
            specials.push((eos.clone()?, eos_id?));
        }

        let mut single = Vec::new();
        if add_bos {
            single.push(bos.clone()?);
        }
        single.push("$0".to_string());
        if add_eos {
            single.push(eos.clone()?);
        }

        let mut pair = Vec::new();
        if add_bos {
            pair.push(format!("{}:0", bos.clone()?));
        }
        pair.push("$A:0".to_string());
        if add_eos {
            pair.push(format!("{}:0", eos.clone()?));
        }
        if add_bos {
            pair.push(format!("{}:1", bos.clone()?));
        }
        pair.push("$B:1".to_string());
        if add_eos {
            pair.push(format!("{}:1", eos?));
        }

        let proc = tokenizers::processors::template::TemplateProcessing::builder()
            .try_single(single)
            .ok()?
            .try_pair(pair)
            .ok()?
            .special_tokens(specials)
            .build()
            .ok()?;

        Some(PostProcessorWrapper::Template(proc))
    }

    /// Builds the tokenizer a GGUF file describes.
    ///
    /// Token types follow `llama_token_type` the way llama.cpp renders them:
    /// unknown (2), control (3) and unused (5) tokens are special, added
    /// atomically and dropped from decoded text; user-defined tokens (4) are
    /// added atomically but survive decoding, since they are the chat and
    /// tool markup (`<tool_call>`, `<think>` on the Qwen files) that the
    /// server parses out of the generated text, exactly as the checkpoint's
    /// own `tokenizer.json` lists them as non-special added tokens.
    pub(super) fn build_tokenizer(ct: &gguf_file::Content) -> Result<Tokenizer> {
        let model_kind = metadata_value(ct, "tokenizer.ggml.model")?
            .to_string()
            .map_err(|e| anyhow::anyhow!("tokenizer.ggml.model: {e}"))?
            .to_lowercase();
        // `llama` is what the converters write for a SentencePiece vocabulary,
        // the one every Llama 1 and 2 era checkpoint carries. It reaches the
        // same pipeline `gemma4` does, since both spell a space as U+2581 and
        // fall back to bytes; what differs is only which of the two models the
        // file gives enough to build.
        let sentencepiece = model_kind == "gemma4" || model_kind == "llama";
        if model_kind != "gpt2" && !sentencepiece {
            anyhow::bail!("unsupported tokenizer model `{model_kind}`");
        }

        let tokens = value_to_string_array(
            metadata_value(ct, "tokenizer.ggml.tokens")?,
            "tokenizer.ggml.tokens",
        )?;
        // Either spelling of the key: the converters disagree on it.
        let unk_token = [
            "tokenizer.ggml.unk_token_id",
            "tokenizer.ggml.unknown_token_id",
        ]
        .iter()
        .find_map(|key| metadata_value(ct, key).ok())
        .and_then(|val| value_to_u32(val).ok())
        .and_then(|id| tokens.get(id as usize).cloned());
        let byte_fallback = match metadata_value(ct, "tokenizer.ggml.byte_fallback") {
            Ok(val) => val.to_bool().map_err(|e| anyhow::anyhow!("{e}"))?,
            // A vocabulary that spells the bytes it cannot compose falls back
            // to them whether or not the file says so.
            Err(_) => tokens.iter().any(|t| t == "<0x0A>"),
        };

        let mut tokenizer =
            match metadata_value(ct, "tokenizer.ggml.merges").and_then(merges_from_value) {
                Ok(merges) => {
                    let vocab: Vocab = tokens
                        .iter()
                        .enumerate()
                        .map(|(i, t)| (t.clone(), i as u32))
                        .collect();
                    let mut builder = BPE::builder()
                        .vocab_and_merges(vocab, merges)
                        .byte_fallback(byte_fallback);
                    if let Some(token) = unk_token.clone() {
                        builder = builder.unk_token(token);
                    }
                    if let Ok(val) = metadata_value(ct, "tokenizer.ggml.ignore_merges") {
                        builder = builder
                            .ignore_merges(val.to_bool().map_err(|e| anyhow::anyhow!("{e}"))?);
                    }
                    Tokenizer::new(
                        builder
                            .build()
                            .map_err(|e| anyhow::anyhow!("BPE build: {e}"))?,
                    )
                }
                // No merge list: a SentencePiece file states the scores instead,
                // and a piece's score is its rank in the merge order, so the
                // merges are rebuilt from the pieces the way the reference
                // conversion rebuilds them.
                Err(e) if sentencepiece => {
                    let scores = value_to_f32_array(
                        metadata_value(ct, "tokenizer.ggml.scores")
                            .map_err(|_| anyhow::anyhow!("neither merges nor scores: {e}"))?,
                        "tokenizer.ggml.scores",
                    )?;
                    if scores.len() != tokens.len() {
                        anyhow::bail!(
                            "tokenizer.ggml.scores has {} entries for {} tokens",
                            scores.len(),
                            tokens.len()
                        );
                    }
                    let vocab: Vocab = tokens
                        .iter()
                        .enumerate()
                        .map(|(i, t)| (t.clone(), i as u32))
                        .collect();
                    let mut builder = BPE::builder()
                        .vocab_and_merges(vocab, merges_from_scores(&tokens, &scores))
                        .byte_fallback(byte_fallback)
                        .fuse_unk(true);
                    if let Some(token) = unk_token.clone() {
                        builder = builder.unk_token(token);
                    }
                    Tokenizer::new(
                        builder
                            .build()
                            .map_err(|e| anyhow::anyhow!("BPE build: {e}"))?,
                    )
                }
                Err(e) => return Err(e),
            };

        let pipeline = if sentencepiece {
            let add_space_prefix = match metadata_value(ct, "tokenizer.ggml.add_space_prefix") {
                Ok(val) => val.to_bool().map_err(|e| anyhow::anyhow!("{e}"))?,
                Err(_) => model_kind == "llama",
            };
            sentencepiece_pipeline(add_space_prefix)?
        } else {
            let pre = metadata_value(ct, "tokenizer.ggml.pre")
                .and_then(|v| v.to_string().map_err(|e| anyhow::anyhow!("{e}")))
                .map(|s| s.to_string())
                .unwrap_or_else(|_| "gpt2".to_string());
            pipeline_from_pre(pre.as_str())?
        };
        let post_processor_base = pipeline.post_processor.clone();

        let add_bos = metadata_value(ct, "tokenizer.ggml.add_bos_token")
            .and_then(|v| v.to_bool().map_err(|e| anyhow::anyhow!("{e}")))
            .unwrap_or(false);
        let add_eos = metadata_value(ct, "tokenizer.ggml.add_eos_token")
            .and_then(|v| v.to_bool().map_err(|e| anyhow::anyhow!("{e}")))
            .unwrap_or(false);
        let bos_id = metadata_value(ct, "tokenizer.ggml.bos_token_id")
            .and_then(value_to_u32)
            .ok();
        let eos_id = metadata_value(ct, "tokenizer.ggml.eos_token_id")
            .and_then(value_to_u32)
            .ok();

        if let Some(norm) = pipeline.normalizer {
            let _ = tokenizer.with_normalizer(Some(norm));
        }
        if let Some(pt) = pipeline.pretokenizer {
            let _ = tokenizer.with_pre_tokenizer(Some(pt));
        }
        if let Some(dec) = pipeline.decoder {
            let _ = tokenizer.with_decoder(Some(dec));
        }

        // Compose the byte-level post-processor with a BOS/EOS template one.
        let template_pp = template_processor(&tokens, bos_id, eos_id, add_bos, add_eos);
        let mut steps = Vec::new();
        if let Some(pp) = post_processor_base {
            steps.push(pp);
        }
        if let Some(tp) = template_pp {
            steps.push(tp);
        }
        if !steps.is_empty() {
            let pp = if steps.len() == 1 {
                steps.pop().unwrap()
            } else {
                ProcessorSequence::new(steps).into()
            };
            let _ = tokenizer.with_post_processor(Some(pp));
        }

        if let Ok(gguf_file::Value::Array(arr)) = metadata_value(ct, "tokenizer.ggml.token_type") {
            let mut specials = Vec::new();
            let mut user_defined = Vec::new();
            for (idx, v) in arr.iter().enumerate() {
                let Some(tok) = tokens.get(idx) else {
                    continue;
                };
                match value_to_u32(v)? {
                    4 => user_defined.push(AddedToken::from(tok.clone(), false).normalized(false)),
                    2 | 3 | 5 => specials.push(AddedToken::from(tok.clone(), true)),
                    _ => {}
                }
            }
            if !specials.is_empty() {
                let _ = tokenizer.add_special_tokens(specials);
            }
            if !user_defined.is_empty() {
                let _ = tokenizer.add_tokens(user_defined);
            }
        }

        let mut explicit_specials = std::collections::HashSet::new();
        for key in [
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.pad_token_id",
            "tokenizer.ggml.sep_token_id",
            "tokenizer.ggml.unk_token_id",
        ] {
            if let Ok(val) = metadata_value(ct, key) {
                explicit_specials.insert(value_to_u32(val)?);
            }
        }
        let specials: Vec<_> = explicit_specials
            .into_iter()
            .filter_map(|id| tokens.get(id as usize))
            .map(|tok| AddedToken::from(tok.clone(), true))
            .collect();
        if !specials.is_empty() {
            let _ = tokenizer.add_special_tokens(specials);
        }

        Ok(tokenizer)
    }
}

#[cfg(test)]
mod gemma4_tokenizer_tests {
    use super::*;

    /// Contract: a SentencePiece vocabulary from a GGUF round-trips the text it
    /// is given and spells a space the way SentencePiece does.
    ///
    /// `llama` is the model name every Llama 1 and 2 era converter writes, and
    /// refusing it left those checkpoints unloadable however complete they
    /// were. Round-tripping is the property that catches the ways this can go
    /// wrong without a reference vocabulary at hand: a normalizer that keeps
    /// U+2581 in the output, a missing byte fallback that drops what no token
    /// spells, an off-by-one between tokens and scores.
    ///
    /// Needs the checkpoint; run with
    /// `cargo test a_sentencepiece_gguf_vocabulary_round_trips -- --ignored`.
    #[test]
    #[ignore]
    fn a_sentencepiece_gguf_vocabulary_round_trips() {
        let gguf = format!(
            "{}/.oxydllm/models/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
            std::env::var("HOME").unwrap()
        );
        if !std::path::Path::new(&gguf).exists() {
            eprintln!("checkpoint assente, salto");
            return;
        }
        let tok = Tokenizer::from_gguf_file(&gguf).expect("tokenizer from a SentencePiece GGUF");
        for text in [
            "Hello world",
            "double  space and punctuation!",
            "Il pesce e' ottimo, pero' costa caro.",
            "numbers 1234567890 and 3.14159",
            "emoji \u{1f600} and symbols \u{2603}",
        ] {
            let ids = tok.encode(text).expect("encode");
            assert!(!ids.is_empty(), "no ids for {text:?}");
            let back = tok.decode(&ids).expect("decode");
            assert_eq!(back.trim(), text.trim(), "round trip differs for {text:?}");
        }
    }

    /// Contract: the `llama` kind gives the ids llama.cpp gives, on a file that
    /// carries merges (TinyLlama) and on one that carries scores alone (the
    /// official Phi-3), both of which spell the first word with the dummy
    /// prefix. Reference ids from `llama-server /tokenize` with
    /// `add_special: false`.
    ///
    /// Needs the checkpoints; run with
    /// `cargo test a_llama_gguf_tokenizer_matches_llama_cpp -- --ignored`.
    #[test]
    #[ignore]
    fn a_llama_gguf_tokenizer_matches_llama_cpp() {
        let home = std::env::var("HOME").unwrap();
        let files = [
            format!(
                "{home}/.oxydllm/models/microsoft/Phi-3-mini-4k-instruct-gguf/Phi-3-mini-4k-instruct-q4.gguf"
            ),
            format!(
                "{home}/.oxydllm/models/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
            ),
        ];
        for gguf in files {
            if !std::path::Path::new(&gguf).exists() {
                eprintln!("checkpoint assente, salto: {gguf}");
                continue;
            }
            let tok = Tokenizer::from_gguf_file(&gguf).expect("tokenizer from a llama GGUF");
            assert_eq!(
                tok.encode("Hello world, how are you?").unwrap(),
                vec![15043, 3186, 29892, 920, 526, 366, 29973],
                "{gguf}"
            );
            assert_eq!(
                tok.decode(&[15043, 3186, 29892, 920, 526, 366, 29973])
                    .unwrap(),
                "Hello world, how are you?",
                "{gguf}"
            );
        }
    }

    /// Contract: the tokenizer built from a GGUF gives the same token ids as the
    /// checkpoint's own `tokenizer.json`.
    ///
    /// Ollama reads this tokenizer out of the GGUF and needs nothing else; until
    /// this passed, oxydLLM needed a `tokenizer.json` beside the file, which
    /// somebody who pulled only the GGUF does not have. A tokenizer that is
    /// nearly right is worse than one that refuses, so the comparison is exact
    /// ids on text that exercises the parts most likely to differ: leading and
    /// repeated spaces, punctuation, non-Latin scripts, and bytes with no token
    /// of their own.
    ///
    /// One case is left out on purpose: the literal string `<eos>`. This file
    /// types that token as normal rather than control, so a tokenizer that
    /// follows the file splits it into pieces while the reference, which lists
    /// it among its added tokens, does not. llama.cpp reads the same field the
    /// same way, so following the file is also what matches the engine this is
    /// compared against.
    ///
    /// Needs the checkpoint; run with
    /// `cargo test gemma4_gguf_tokenizer_matches_the_reference -- --ignored`.
    #[test]
    #[ignore]
    fn gemma4_gguf_tokenizer_matches_the_reference() {
        let dir = format!(
            "{}/.oxydllm/models/unsloth/gemma-4-12b-it-GGUF",
            std::env::var("HOME").unwrap()
        );
        let gguf = format!("{dir}/gemma-4-12b-it-Q4_K_M.gguf");
        if !std::path::Path::new(&gguf).exists() {
            eprintln!("checkpoint assente, salto");
            return;
        }
        let from_gguf = Tokenizer::from_gguf_file(&gguf).expect("tokenizer from GGUF");
        let reference = Tokenizer::from_dir(&dir).expect("tokenizer.json reference");

        let corpus = [
            "Hello world",
            " leading space",
            "double  space and\ttab",
            "What is the capital of France?",
            "Il pesce è ottimo, però costa caro.",
            "日本語のテキストも試す",
            "emoji \u{1f600} and symbols \u{2603}",
            "<|turn>user\nhello<turn|>",
            "<|tool>name<tool|> and <channel|>",
            "numbers 1234567890 and 3.14159",
            "",
        ];
        for text in corpus {
            let a = from_gguf.encode(text).expect("gguf encode");
            let b = reference.encode(text).expect("reference encode");
            assert_eq!(a, b, "ids differ for {text:?}");
        }
    }
}

#[cfg(test)]
#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    /// Contract: `<|end|>` ends the turn for Phi-3, where it is the end of
    /// turn, and not for a harmony vocabulary, where it ends one message and
    /// `<|return|>` ends the turn; with it as a stop, gpt-oss stopped after
    /// its analysis and answered nothing.
    #[test]
    fn the_harmony_message_end_is_not_a_stop() {
        fn tokenizer_with(specials: &[&str]) -> Tokenizer {
            let tmp = tempfile::tempdir().unwrap();
            let vocab: Vec<(String, u32)> = ["[UNK]", "hi"]
                .iter()
                .chain(specials.iter())
                .enumerate()
                .map(|(i, t)| (t.to_string(), i as u32))
                .collect();
            let model = WordLevel::builder()
                .vocab(vocab.iter().cloned().collect())
                .unk_token("[UNK]".to_string())
                .build()
                .unwrap();
            tokenizers::Tokenizer::new(model)
                .save(tmp.path().join("tokenizer.json"), false)
                .unwrap();
            let decoder: serde_json::Map<String, serde_json::Value> = vocab
                .iter()
                .filter(|(t, _)| specials.contains(&t.as_str()))
                .map(|(t, id)| {
                    (
                        id.to_string(),
                        serde_json::json!({"content": t, "special": true}),
                    )
                })
                .collect();
            let cfg = serde_json::json!({
                "eos_token": specials[0],
                "added_tokens_decoder": decoder,
            });
            std::fs::write(tmp.path().join("tokenizer_config.json"), cfg.to_string()).unwrap();
            Tokenizer::from_dir(tmp.path().to_str().unwrap()).unwrap()
        }

        let harmony = tokenizer_with(&[
            "<|return|>",
            "<|end|>",
            "<|channel|>",
            "<|message|>",
            "<|call|>",
        ]);
        assert_eq!(
            harmony.stop_token_ids(),
            vec![
                harmony.special_token_id("<|return|>").unwrap(),
                harmony.special_token_id("<|call|>").unwrap()
            ],
            "the turn ends at <|return|> or at a function call"
        );
        assert!(!harmony.stop_text_markers().iter().any(|m| m == "<|end|>"));
        assert!(
            harmony.has_thinking_support(),
            "the analysis channel is reasoning on request"
        );

        let phi = tokenizer_with(&["<|endoftext|>", "<|end|>"]);
        assert!(
            phi.stop_token_ids()
                .contains(&phi.special_token_id("<|end|>").unwrap())
        );
    }

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
        let _ = inner.with_pre_tokenizer(Some(Whitespace {}));
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

    /// Contract: a SentencePiece GGUF that carries scores and no merge list,
    /// as the official Phi-3 files do, builds its tokenizer from the scores.
    ///
    /// The scores branch existed but sat behind a `?` on the merge-list key,
    /// so a file without that key failed to load before reaching it; only a
    /// file that had merges as well, like TinyLlama's, ever loaded.
    #[test]
    fn a_sentencepiece_gguf_without_merges_builds_from_scores() {
        use candle_core::quantized::gguf_file::{Content, Value, VersionedMagic};
        use std::collections::HashMap;

        let tokens = [
            "<unk>",
            "<s>",
            "</s>",
            "\u{2581}hi",
            "\u{2581}h",
            "h",
            "i",
            "\u{2581}yo",
            "\u{2581}y",
            "y",
            "o",
            "\u{2581}",
        ];
        let scores = [
            0.0f32, 0.0, 0.0, -1.0, -2.0, -3.0, -3.0, -1.5, -2.5, -3.0, -3.0, -3.0,
        ];
        let token_types = [2u32, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let mut metadata: HashMap<String, Value> = HashMap::new();
        metadata.insert("tokenizer.ggml.model".into(), Value::String("llama".into()));
        metadata.insert(
            "tokenizer.ggml.tokens".into(),
            Value::Array(tokens.iter().map(|t| Value::String((*t).into())).collect()),
        );
        metadata.insert(
            "tokenizer.ggml.scores".into(),
            Value::Array(scores.iter().map(|&s| Value::F32(s)).collect()),
        );
        metadata.insert(
            "tokenizer.ggml.token_type".into(),
            Value::Array(token_types.iter().map(|&t| Value::U32(t)).collect()),
        );
        metadata.insert("tokenizer.ggml.unknown_token_id".into(), Value::U32(0));
        let content = Content {
            magic: VersionedMagic::GgufV3,
            metadata,
            tensor_infos: HashMap::new(),
            tensor_data_offset: 0,
        };
        let tok =
            Tokenizer::from_gguf_content(&content).expect("a scores-only SentencePiece file loads");
        let ids = tok.encode("hi yo").unwrap();
        assert_eq!(ids, vec![3, 7], "the pieces are merged in score order");
        assert_eq!(tok.decode(&ids).unwrap(), "hi yo");
    }

    /// Contract: a GGUF token of type user-defined (4) is one token in the
    /// input and survives decoding, while a control token (3) is dropped.
    ///
    /// Every non-normal type was marked special, so `<tool_call>` and
    /// `<think>`, which the Qwen converters type as user-defined, vanished
    /// from the generated text on GGUF models: the tool-call parser never
    /// saw the tags that the same model's `tokenizer.json` keeps.
    #[test]
    fn gguf_user_defined_tokens_survive_decoding() {
        use candle_core::quantized::gguf_file::{Content, Value, VersionedMagic};
        use std::collections::HashMap;

        let tokens = [
            "<|x|>",
            "Ġ",
            "u",
            "s",
            "e",
            "r",
            "us",
            "er",
            "user",
            "Ġuser",
            "<tool_call>",
            "<|end|>",
        ];
        let token_types = [3u32, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4];
        let merges = ["u s", "e r", "us er", "Ġ user"];

        let mut metadata: HashMap<String, Value> = HashMap::new();
        metadata.insert("tokenizer.ggml.model".into(), Value::String("gpt2".into()));
        metadata.insert("tokenizer.ggml.pre".into(), Value::String("gpt-2".into()));
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

        assert_eq!(tok.encode("<tool_call>user").unwrap(), vec![10, 8]);
        assert_eq!(tok.decode(&[10, 8]).unwrap(), "<tool_call>user");
        assert_eq!(
            tok.decode(&[0, 8]).unwrap(),
            "user",
            "control tokens stay hidden"
        );
        assert_eq!(tok.special_token_id("<tool_call>"), Some(10));
        assert!(
            tok.stop_token_ids().contains(&11),
            "Phi-3 ends its turn with the user-defined <|end|>, which llama.cpp knows by name"
        );
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
