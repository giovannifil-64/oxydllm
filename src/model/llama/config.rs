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
    pub intermediate_size: usize,
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

    pub fn primary_eos_token_id(&self) -> u32 {
        *self.eos_token_ids.last().unwrap_or(&128_001)
    }
}
