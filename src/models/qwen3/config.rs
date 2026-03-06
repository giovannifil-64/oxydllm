use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct Qwen3Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    pub vocab_size: usize,
    pub head_dim: Option<usize>,
    pub max_position_embeddings: usize,
}

fn default_rope_theta() -> f64 {
    1_000_000.0
}

impl Qwen3Config {
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&raw)?)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

impl From<Qwen3Config> for crate::common::config::StandardTransformerConfig {
    fn from(cfg: Qwen3Config) -> Self {
        Self {
            vocab_size: cfg.vocab_size,
            num_hidden_layers: cfg.num_hidden_layers,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim(),
            rms_norm_eps: cfg.rms_norm_eps,
            rope_theta: cfg.rope_theta,
            max_position_embeddings: cfg.max_position_embeddings,
            qk_norm: true,
            tie_word_embeddings: false,
            attention_scale: None,
            eos_token_ids: vec![151645],
        }
    }
}
