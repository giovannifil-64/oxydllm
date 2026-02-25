use super::ffn::Activation;
pub struct BlockConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub qk_norm: bool,
    pub sliding_window: Option<usize>,
    pub activation: Activation,
}
