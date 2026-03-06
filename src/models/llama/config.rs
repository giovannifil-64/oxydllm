use serde::{Deserialize, Deserializer};

fn deserialize_eos_ids<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u32>, D::Error> {
    let v: serde_json::Value = serde::Deserialize::deserialize(d)?;
    match &v {
        serde_json::Value::Number(n) => n
            .as_u64()
            .map(|x| vec![x as u32])
            .ok_or_else(|| serde::de::Error::custom("eos_token_id is not a valid u32")),
        serde_json::Value::Array(arr) => {
            let ids: Vec<u32> = arr
                .iter()
                .filter_map(|x| x.as_u64())
                .map(|x| x as u32)
                .collect();
            if ids.is_empty() {
                Err(serde::de::Error::custom("eos_token_id array is empty or invalid"))
            } else {
                Ok(ids)
            }
        }
        other => Err(serde::de::Error::custom(format!(
            "unexpected eos_token_id type: {other}"
        ))),
    }
}

fn default_rope_theta() -> f64 {
    500_000.0
}

fn default_eos_token_ids() -> Vec<u32> {
    vec![128_001]
}

#[derive(Debug, Deserialize, Clone)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(
        default = "default_eos_token_ids",
        deserialize_with = "deserialize_eos_ids"
    )]
    pub eos_token_ids: Vec<u32>,
}

impl LlamaConfig {
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&raw)?)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

impl From<LlamaConfig> for crate::common::config::StandardTransformerConfig {
    fn from(cfg: LlamaConfig) -> Self {
        let mut eos_token_ids = cfg.eos_token_ids.clone();
        for &extra in &[128009u32, 128008u32] {
            if !eos_token_ids.contains(&extra) {
                eos_token_ids.push(extra);
            }
        }
        Self {
            vocab_size: cfg.vocab_size,
            num_hidden_layers: cfg.num_hidden_layers,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim(),
            rms_norm_eps: cfg.rms_norm_eps,
            rope_theta: cfg.rope_theta,
            max_position_embeddings: cfg.max_position_embeddings,
            qk_norm: false,
            tie_word_embeddings: cfg.tie_word_embeddings,
            attention_scale: None,
            eos_token_ids,
        }
    }
}
