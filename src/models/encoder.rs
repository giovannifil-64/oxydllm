//! Encoder-only models (BERT / RoBERTa family) for text embeddings.
//!
//! A self-contained runtime, deliberately separate from the decoder engine:
//! encoders need bidirectional attention with no KV cache, LayerNorm with
//! bias, absolute position embeddings, and exact (erf) GELU, none of which
//! the paged decode path provides. [`EncoderModel::embed`] runs one sequence
//! through the stack and returns the pooled, L2-normalised sentence
//! embedding as sentence-transformers computes it (CLS pooling for the
//! granite-embedding r1 family).
//!
//! Config parsing lives here rather than in `hf_parser`, which remains the
//! single source of truth for decoder blueprints only.

use crate::common::linear::{Linear, softmax_last_dim};
use crate::common::weights::ModelWeights;
use anyhow::{Context, Result};
use candle_core::{D, DType, Device, Tensor};

/// Mean-subtracting LayerNorm with bias, computed in F32.
struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    fn load(weights: &ModelWeights, prefix: &str, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: weights.get(&format!("{prefix}.weight"))?.clone(),
            bias: weights.get(&format!("{prefix}.bias"))?.clone(),
            eps,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let mean = x.mean_keepdim(D::Minus1)?;
        let centered = x.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = centered.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        Ok(normed
            .broadcast_mul(&self.weight.to_dtype(DType::F32)?)?
            .broadcast_add(&self.bias.to_dtype(DType::F32)?)?
            .to_dtype(dtype)?)
    }
}

struct EncoderLayer {
    query: Linear,
    key: Linear,
    value: Linear,
    attn_out: Linear,
    attn_ln: LayerNorm,
    ffn_up: Linear,
    ffn_down: Linear,
    ffn_ln: LayerNorm,
}

/// How the token embeddings pool into one sentence vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pooling {
    /// The first (`[CLS]` / `<s>`) token's hidden state.
    Cls,
    /// The mean over all token hidden states.
    Mean,
}

/// A loaded encoder-only embedding model.
pub struct EncoderModel {
    word_embeddings: Tensor,
    position_embeddings: Tensor,
    token_type_embeddings: Tensor,
    embed_ln: LayerNorm,
    layers: Vec<EncoderLayer>,
    n_heads: usize,
    head_dim: usize,
    /// RoBERTa offsets positions by `pad_token_id + 1` (2); BERT starts at 0.
    position_offset: usize,
    max_positions: usize,
    pooling: Pooling,
    device: Device,
    dtype: DType,
}

impl EncoderModel {
    /// Loads an encoder checkpoint (`RobertaModel` / `BertModel`) from
    /// `model_dir`. Pooling comes from the sentence-transformers
    /// `1_Pooling/config.json` when present, defaulting to CLS.
    ///
    /// ## Errors
    /// Fails on unsupported architectures, missing tensors, or load errors.
    pub fn load(model_dir: &str, device: &Device, dtype: DType) -> Result<Self> {
        let config_raw = std::fs::read_to_string(format!("{model_dir}/config.json"))
            .with_context(|| format!("read {model_dir}/config.json"))?;
        let config: serde_json::Value = serde_json::from_str(&config_raw)?;
        let model_type = config["model_type"].as_str().unwrap_or_default();
        if !matches!(model_type, "roberta" | "bert" | "xlm-roberta") {
            anyhow::bail!("encoder: unsupported model_type '{model_type}'");
        }
        let hidden = config["hidden_size"].as_u64().context("hidden_size")? as usize;
        let n_layers = config["num_hidden_layers"].as_u64().context("layers")? as usize;
        let n_heads = config["num_attention_heads"].as_u64().context("heads")? as usize;
        let eps = config["layer_norm_eps"].as_f64().unwrap_or(1e-12);
        let position_offset = if model_type == "bert" {
            0
        } else {
            config["pad_token_id"].as_u64().unwrap_or(1) as usize + 1
        };
        let max_positions = config["max_position_embeddings"].as_u64().unwrap_or(512) as usize;

        let pooling = match std::fs::read_to_string(format!("{model_dir}/1_Pooling/config.json")) {
            Ok(raw) => {
                let v: serde_json::Value = serde_json::from_str(&raw)?;
                if v["pooling_mode_mean_tokens"].as_bool().unwrap_or(false) {
                    Pooling::Mean
                } else {
                    Pooling::Cls
                }
            }
            Err(_) => Pooling::Cls,
        };

        let weight_path = format!("{model_dir}/model.safetensors");
        let weights = ModelWeights::load(&[weight_path.as_str()], device, dtype, None)?;

        let lin = |name: &str| -> Result<Linear> {
            let w = weights.get(&format!("{name}.weight"))?.clone();
            let b = weights.get(&format!("{name}.bias"))?.clone();
            Ok(Linear::new(w, Some(b))?)
        };

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("encoder.layer.{i}");
            layers.push(EncoderLayer {
                query: lin(&format!("{p}.attention.self.query"))?,
                key: lin(&format!("{p}.attention.self.key"))?,
                value: lin(&format!("{p}.attention.self.value"))?,
                attn_out: lin(&format!("{p}.attention.output.dense"))?,
                attn_ln: LayerNorm::load(
                    &weights,
                    &format!("{p}.attention.output.LayerNorm"),
                    eps,
                )?,
                ffn_up: lin(&format!("{p}.intermediate.dense"))?,
                ffn_down: lin(&format!("{p}.output.dense"))?,
                ffn_ln: LayerNorm::load(&weights, &format!("{p}.output.LayerNorm"), eps)?,
            });
        }

        Ok(Self {
            word_embeddings: weights.get("embeddings.word_embeddings.weight")?.clone(),
            position_embeddings: weights
                .get("embeddings.position_embeddings.weight")?
                .clone(),
            token_type_embeddings: weights
                .get("embeddings.token_type_embeddings.weight")?
                .clone(),
            embed_ln: LayerNorm::load(&weights, "embeddings.LayerNorm", eps)?,
            layers,
            n_heads,
            head_dim: hidden / n_heads,
            position_offset,
            max_positions,
            pooling,
            device: device.clone(),
            dtype,
        })
    }

    /// Embeds one tokenized sequence (special tokens included) and returns
    /// the pooled, L2-normalised sentence vector in F32.
    ///
    /// ## Errors
    /// Fails if the sequence is empty, longer than the position table, or a
    /// tensor op fails.
    pub fn embed(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let n = ids.len();
        if n == 0 {
            anyhow::bail!("encoder: empty input");
        }
        if n + self.position_offset > self.max_positions {
            anyhow::bail!(
                "encoder: input of {n} tokens exceeds the model's {} positions",
                self.max_positions - self.position_offset
            );
        }

        let ids_t = Tensor::from_vec(ids.to_vec(), (n,), &self.device)?;
        let pos: Vec<u32> = (0..n as u32)
            .map(|i| i + self.position_offset as u32)
            .collect();
        let pos_t = Tensor::from_vec(pos, (n,), &self.device)?;

        let mut x = (self
            .word_embeddings
            .index_select(&ids_t, 0)?
            .add(&self.position_embeddings.index_select(&pos_t, 0)?)?
            .broadcast_add(&self.token_type_embeddings.narrow(0, 0, 1)?)?)
        .to_dtype(self.dtype)?;
        x = self.embed_ln.forward(&x)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        for layer in &self.layers {
            let split = |t: Tensor| -> Result<Tensor> {
                Ok(t.reshape((n, self.n_heads, self.head_dim))?
                    .transpose(0, 1)?
                    .contiguous()?)
            };
            let q = split(layer.query.forward(&x)?)?;
            let k = split(layer.key.forward(&x)?)?;
            let v = split(layer.value.forward(&x)?)?;

            let scores = (q.matmul(&k.transpose(1, 2)?)? * scale)?;
            let probs = softmax_last_dim(&scores.to_dtype(DType::F32)?)?.to_dtype(self.dtype)?;
            let ctx = probs
                .matmul(&v)?
                .transpose(0, 1)?
                .reshape((n, self.n_heads * self.head_dim))?;
            let attn = layer.attn_out.forward(&ctx)?;
            x = layer.attn_ln.forward(&(x + attn)?)?;

            let up = layer.ffn_up.forward(&x)?.gelu_erf()?;
            let ffn = layer.ffn_down.forward(&up)?;
            x = layer.ffn_ln.forward(&(x + ffn)?)?;
        }

        let pooled = match self.pooling {
            Pooling::Cls => x.narrow(0, 0, 1)?.squeeze(0)?,
            Pooling::Mean => x.mean(0)?,
        }
        .to_dtype(DType::F32)?;
        let norm = pooled.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        let out = (pooled / norm as f64)?;
        Ok(out.to_vec1()?)
    }

    /// The embedding dimension of the pooled vectors.
    pub fn hidden_size(&self) -> usize {
        self.n_heads * self.head_dim
    }

    /// The device this encoder runs on, for GPU-lock scoping.
    pub fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::Tokenizer;

    const FIXTURES: &str = include_str!("encoder_fixtures.json");
    const MODEL_DIR: &str = concat!(
        env!("HOME"),
        "/.oxydllm/models/ibm-granite/granite-embedding-125m-english"
    );

    /// Contract: tokenization and the full encoder forward match the
    /// transformers reference: identical input ids, and cosine similarity
    /// above 0.9995 between our F32-on-CPU embedding and the reference for
    /// every fixture sentence. Skips when the local checkpoint is absent
    /// (CI has no model files).
    #[test]
    fn encoder_matches_transformers_reference() {
        if !std::path::Path::new(MODEL_DIR)
            .join("model.safetensors")
            .exists()
        {
            return;
        }
        let fixtures: serde_json::Value = serde_json::from_str(FIXTURES).unwrap();
        let tokenizer = Tokenizer::from_dir(MODEL_DIR).unwrap();
        let model = EncoderModel::load(MODEL_DIR, &Device::Cpu, DType::F32).unwrap();
        assert_eq!(model.hidden_size(), 768);

        for f in fixtures.as_array().unwrap() {
            let text = f["text"].as_str().unwrap();
            let want_ids: Vec<u32> = f["input_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            let want: Vec<f32> = f["embedding"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();

            let ids = tokenizer.encode_with_special_tokens(text).unwrap();
            assert_eq!(ids, want_ids, "tokenization of {text:?}");

            let got = model.embed(&ids).unwrap();
            let dot: f32 = got.iter().zip(&want).map(|(a, b)| a * b).sum();
            assert!(
                dot > 0.9995,
                "cosine {dot} for {text:?} (both sides are unit-norm)"
            );
        }
    }
}
