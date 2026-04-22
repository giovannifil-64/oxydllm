//
// arch_regression.rs — Architecture regression test matrix
//
// Each test builds a tiny StandardTransformer (hidden=32, 2 layers, 4 heads)
// with random weights matching a specific architecture's feature combination,
// runs a prefill + decode forward pass, and verifies:
//   1. Output shape is correct
//   2. No NaN/Inf in logits
//   3. Decode after prefill doesn't crash
//
// This catches regressions when modifying shared code paths (attention, FFN,
// block, norm) — if adding Qwen5 breaks Gemma2's softcap+sliding window
// path, the test fails immediately.
//
// Usage in `src/models/mod.rs`:
//   #[cfg(test)]
//   mod arch_regression;
//

#[cfg(test)]
mod tests {
    use crate::common::block::TransformerBlock;
    use crate::common::config::*;
    use crate::common::linear::*;
    use crate::common::norm::RMSNorm;
    use crate::common::paged::*;
    use crate::common::rope::*;
    use crate::models::gguf_model::StandardTransformer;
    use crate::models::traits::BatchModel;
    use candle_core::*;
    use rustc_hash::FxHashMap;
    use std::sync::{Arc, Mutex};

    const HIDDEN: usize = 32;
    const HEADS: usize = 4;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 8;
    const INTER: usize = 64;
    const VOCAB: usize = 64;
    const LAYERS: usize = 2;
    const MAX_SEQ: usize = 64;
    const KV_BLOCKS: usize = 32;

    struct ArchSpec {
        name: &'static str,
        activation: Activation,
        norm_type: NormType,
        qk_norm: bool,
        v_norm: bool,
        has_ffn_norms: bool,
        embed_scale: Option<f64>,
        attn_softcap: Option<f64>,
        logit_softcap: Option<f64>,
        sliding_window: Option<usize>,
        attention_scale: Option<f64>,
        tie_weights: bool,
        rope_theta: f64,
    }

    impl Default for ArchSpec {
        fn default() -> Self {
            Self {
                name: "default",
                activation: Activation::SiLU,
                norm_type: NormType::Standard,
                qk_norm: false,
                v_norm: false,
                has_ffn_norms: false,
                embed_scale: None,
                attn_softcap: None,
                logit_softcap: None,
                sliding_window: None,
                attention_scale: None,
                tie_weights: false,
                rope_theta: 10_000.0,
            }
        }
    }

    //  Weight factory
    fn rand_t(dims: &[usize], dev: &Device) -> Tensor {
        Tensor::randn(0f32, 0.02, dims, dev).unwrap()
    }

    fn ones_t(dim: usize, dev: &Device) -> Tensor {
        Tensor::ones(&[dim], DType::F32, dev).unwrap()
    }

    fn build_weights(spec: &ArchSpec, dev: &Device) -> FxHashMap<String, Tensor> {
        let mut t = FxHashMap::default();
        let h = HIDDEN;
        let q_dim = HEADS * HEAD_DIM;
        let kv_dim = KV_HEADS * HEAD_DIM;

        t.insert("model.embed_tokens.weight".into(), rand_t(&[VOCAB, h], dev));
        if !spec.tie_weights {
            t.insert("lm_head.weight".into(), rand_t(&[VOCAB, h], dev));
        }
        t.insert("model.norm.weight".into(), ones_t(h, dev));

        for i in 0..LAYERS {
            let p = format!("model.layers.{i}");

            // Attention projections
            t.insert(
                format!("{p}.self_attn.q_proj.weight"),
                rand_t(&[q_dim, h], dev),
            );
            t.insert(
                format!("{p}.self_attn.k_proj.weight"),
                rand_t(&[kv_dim, h], dev),
            );
            t.insert(
                format!("{p}.self_attn.v_proj.weight"),
                rand_t(&[kv_dim, h], dev),
            );
            t.insert(
                format!("{p}.self_attn.o_proj.weight"),
                rand_t(&[h, q_dim], dev),
            );

            // QK norm (Qwen3, Gemma3, Gemma4)
            if spec.qk_norm {
                t.insert(
                    format!("{p}.self_attn.q_norm.weight"),
                    ones_t(HEAD_DIM, dev),
                );
                t.insert(
                    format!("{p}.self_attn.k_norm.weight"),
                    ones_t(HEAD_DIM, dev),
                );
            }

            // Layer norms
            t.insert(format!("{p}.input_layernorm.weight"), ones_t(h, dev));
            t.insert(
                format!("{p}.post_attention_layernorm.weight"),
                ones_t(h, dev),
            );

            // Extra FFN norms (Gemma2, Gemma3, Gemma4)
            if spec.has_ffn_norms {
                t.insert(
                    format!("{p}.pre_feedforward_layernorm.weight"),
                    ones_t(h, dev),
                );
                t.insert(
                    format!("{p}.post_feedforward_layernorm.weight"),
                    ones_t(h, dev),
                );
            }

            // FFN (gate+up fused layout — standard for all supported archs)
            t.insert(
                format!("{p}.mlp.gate_proj.weight"),
                rand_t(&[INTER, h], dev),
            );
            t.insert(format!("{p}.mlp.up_proj.weight"), rand_t(&[INTER, h], dev));
            t.insert(
                format!("{p}.mlp.down_proj.weight"),
                rand_t(&[h, INTER], dev),
            );
        }

        t
    }

    // Model builder
    fn build_model(spec: ArchSpec) -> Result<StandardTransformer> {
        let dev = Device::Cpu;
        let dtype = DType::F32;
        let tensors = build_weights(&spec, &dev);
        let weights = crate::common::weights::ModelWeights::from_tensors(tensors);

        let block_cfg = BlockConfig {
            n_heads: HEADS,
            n_kv_heads: KV_HEADS,
            head_dim: HEAD_DIM,
            rms_norm_eps: 1e-5,
            qk_norm: spec.qk_norm,
            attention_scale: spec.attention_scale,
            activation: spec.activation,
            norm_type: spec.norm_type,
            attn_softcap: spec.attn_softcap,
            v_norm: spec.v_norm,
            has_ffn_norms: spec.has_ffn_norms,
            sliding_window: spec.sliding_window,
        };

        let blocks = (0..LAYERS)
            .map(|i| TransformerBlock::load(&block_cfg, i, &weights))
            .collect::<Result<Vec<_>>>()?;

        let norm = RMSNorm::load(&weights, "model.norm", 1e-5, spec.norm_type)?;
        let embed_tokens = Embedding::new(weights.get("model.embed_tokens.weight")?.clone());

        let lm_head = if spec.tie_weights {
            AnyLinear::from_weight(weights.get("model.embed_tokens.weight")?.clone(), None)
        } else {
            AnyLinear::from_weight(weights.get("lm_head.weight")?.clone(), None)
        };

        let ropes = (0..LAYERS)
            .map(|_| RotaryEmbedding::new(HEAD_DIM, MAX_SEQ, spec.rope_theta, dtype, &dev))
            .collect::<Result<Vec<_>>>()?;

        let allocators = (0..LAYERS)
            .map(|_| -> Result<SharedBlockAllocator> {
                Ok(Arc::new(Mutex::new(BlockAllocator::new(
                    KV_BLOCKS,
                    DEFAULT_BLOCK_SIZE,
                    KV_HEADS,
                    HEAD_DIM,
                    dtype,
                    &dev,
                    None,
                )?)))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(StandardTransformer {
            embed_tokens,
            blocks,
            norm,
            lm_head,
            ropes,
            allocators,
            device: dev,
            stop_token_ids: vec![2],
            vocab_size: VOCAB,
            max_seq_len: MAX_SEQ,
            embed_scale: spec.embed_scale,
            logit_softcap: spec.logit_softcap,
            per_layer_input_embed: None,
            per_layer_input_embed_scale: None,
            per_layer_model_projection: None,
            per_layer_model_projection_scale: None,
            per_layer_projection_norm: None,
            per_layer_input_scale: None,
            kv_shared_layer_map: None,
        })
    }

    //  Forward pass helpers

    fn run_prefill(
        model: &StandardTransformer,
        caches: &mut Vec<PagedKvCache>,
        seq_len: usize,
    ) -> Result<Tensor> {
        let dev = model.device();
        let tokens: Vec<u32> = (1..=seq_len as u32).collect();
        let positions: Vec<u32> = (0..seq_len as u32).collect();
        let input = Tensor::from_vec(tokens, (1, seq_len), dev)?;
        let pos = Tensor::from_vec(positions, (seq_len,), dev)?;

        let mut slices: Vec<&mut [PagedKvCache]> = vec![caches.as_mut_slice()];
        model.forward_batch(&input, &pos, &mut slices, &[seq_len])
    }

    fn run_decode(
        model: &StandardTransformer,
        caches: &mut Vec<PagedKvCache>,
        position: usize,
    ) -> Result<Tensor> {
        let dev = model.device();
        let token_id = ((position % (VOCAB - 1)) + 1) as u32;
        let input = Tensor::from_vec(vec![token_id], (1, 1), dev)?;
        let pos = Tensor::from_vec(vec![position as u32], (1,), dev)?;

        let mut slices: Vec<&mut [PagedKvCache]> = vec![caches.as_mut_slice()];
        model.forward_batch(&input, &pos, &mut slices, &[1])
    }

    fn make_caches(model: &StandardTransformer) -> Vec<PagedKvCache> {
        model
            .allocators()
            .iter()
            .map(|a| PagedKvCache::new(Arc::clone(a)))
            .collect()
    }

    fn clear_caches(caches: &mut Vec<PagedKvCache>) {
        for c in caches.iter_mut() {
            c.clear();
        }
    }

    //  Assertions

    fn assert_logits_ok(logits: &Tensor, seq_len: usize, arch_name: &str) {
        let dims = logits.dims();
        assert_eq!(
            dims,
            &[1, seq_len, VOCAB],
            "[{arch_name}] logits shape mismatch: expected [1, {seq_len}, {VOCAB}], got {dims:?}"
        );

        let flat: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            flat.iter().all(|v| v.is_finite()),
            "[{arch_name}] logits contain NaN or Inf"
        );

        let any_nonzero = flat.iter().any(|v| v.abs() > 1e-10);
        assert!(
            any_nonzero,
            "[{arch_name}] logits are all zeros — likely a weight loading bug"
        );
    }

    fn run_arch_test(spec: ArchSpec) {
        let name = spec.name;
        let model =
            build_model(spec).unwrap_or_else(|e| panic!("[{name}] model build failed: {e}"));
        let mut caches = make_caches(&model);

        // Prefill: 8 tokens
        let prefill_logits = run_prefill(&model, &mut caches, 8)
            .unwrap_or_else(|e| panic!("[{name}] prefill failed: {e}"));
        assert_logits_ok(&prefill_logits, 8, name);

        // Decode: 3 steps
        for step in 0..3 {
            let decode_logits = run_decode(&model, &mut caches, 8 + step)
                .unwrap_or_else(|e| panic!("[{name}] decode step {step} failed: {e}"));
            assert_logits_ok(&decode_logits, 1, name);
        }

        clear_caches(&mut caches);
    }

    //
    // Architecture tests
    //
    // Each test exercises a unique combination of feature flags.
    // The comment lists which code paths are specifically activated.
    //

    #[test]
    fn arch_llama() {
        run_arch_test(ArchSpec {
            name: "Llama",
            activation: Activation::SiLU,
            rope_theta: 500_000.0,
            ..Default::default()
        });
    }

    /// MistralForCausalLM — SiLU + sliding window attention.
    /// Exercises: truncate_kv_window in decode, causal_mask_prefixed in prefill.
    #[test]
    fn arch_mistral() {
        run_arch_test(ArchSpec {
            name: "Mistral",
            activation: Activation::SiLU,
            sliding_window: Some(16),
            ..Default::default()
        });
    }

    /// Qwen2ForCausalLM — SiLU, high rope theta, otherwise standard.
    #[test]
    fn arch_qwen2() {
        run_arch_test(ArchSpec {
            name: "Qwen2",
            activation: Activation::SiLU,
            rope_theta: 1_000_000.0,
            ..Default::default()
        });
    }

    /// Qwen3ForCausalLM — SiLU + QK norm.
    /// Exercises: q_norm/k_norm paths in Attention.
    #[test]
    fn arch_qwen3() {
        run_arch_test(ArchSpec {
            name: "Qwen3",
            activation: Activation::SiLU,
            qk_norm: true,
            rope_theta: 1_000_000.0,
            ..Default::default()
        });
    }

    /// GemmaForCausalLM — GeLU + Gemma norm (+1 offset) + embed_scale + tied weights.
    /// Exercises: NormType::Gemma (weight+1), gelu_tanh activation, embed_scale multiply.
    #[test]
    fn arch_gemma() {
        run_arch_test(ArchSpec {
            name: "Gemma",
            activation: Activation::GeLUTanh,
            norm_type: NormType::Gemma,
            embed_scale: Some((HIDDEN as f64).sqrt()),
            tie_weights: true,
            ..Default::default()
        });
    }

    /// Gemma2ForCausalLM — GeLU + Gemma norm + FFN norms + attn softcap + logit softcap
    /// + sliding window (alternating via Gemma norm type).
    /// Exercises: pre/post_feedforward norms, softcap in attention scores, logit capping, compute_sliding_window Gemma odd-layer disable.
    #[test]
    fn arch_gemma2() {
        run_arch_test(ArchSpec {
            name: "Gemma2",
            activation: Activation::GeLUTanh,
            norm_type: NormType::Gemma,
            has_ffn_norms: true,
            embed_scale: Some((HIDDEN as f64).sqrt()),
            attn_softcap: Some(50.0),
            logit_softcap: Some(30.0),
            sliding_window: Some(16),
            tie_weights: true,
            ..Default::default()
        });
    }

    /// Gemma3ForCausalLM — GeLU + Gemma norm + QK norm + FFN norms + embed_scale.
    /// Exercises: combined qk_norm + ffn_norms (both active simultaneously).
    #[test]
    fn arch_gemma3() {
        run_arch_test(ArchSpec {
            name: "Gemma3",
            activation: Activation::GeLUTanh,
            norm_type: NormType::Gemma,
            qk_norm: true,
            has_ffn_norms: true,
            embed_scale: Some((HIDDEN as f64).sqrt()),
            tie_weights: true,
            ..Default::default()
        });
    }

    /// Gemma4ForConditionalGeneration — GeLU + Standard norm + QK norm + V norm
    /// + FFN norms + logit softcap + attention_scale override.
    /// Exercises: v_norm (rms_norm_no_weight), attention_scale override, Standard+qk_norm combo.
    #[test]
    fn arch_gemma4() {
        run_arch_test(ArchSpec {
            name: "Gemma4",
            activation: Activation::GeLUTanh,
            norm_type: NormType::Standard,
            qk_norm: true,
            v_norm: true,
            has_ffn_norms: true,
            embed_scale: Some((HIDDEN as f64).sqrt()),
            logit_softcap: Some(30.0),
            attention_scale: Some(1.0),
            tie_weights: true,
            ..Default::default()
        });
    }

    /// Phi3ForCausalLM — SiLU, standard norm (similar to Llama baseline).
    #[test]
    fn arch_phi3() {
        run_arch_test(ArchSpec {
            name: "Phi3",
            activation: Activation::SiLU,
            ..Default::default()
        });
    }

    //
    // Edge case / combination tests
    //

    /// Verify that sliding window + softcap together don't interfere.
    /// This is the Gemma2 path but with more aggressive window to force truncation.
    #[test]
    fn edge_sliding_window_plus_softcap() {
        run_arch_test(ArchSpec {
            name: "SlidingWindow+Softcap",
            activation: Activation::GeLUTanh,
            norm_type: NormType::Gemma,
            attn_softcap: Some(50.0),
            sliding_window: Some(4), // Very small → forces truncation even with 8-token prefill
            tie_weights: true,
            embed_scale: Some((HIDDEN as f64).sqrt()),
            ..Default::default()
        });
    }

    /// All features enabled simultaneously — catches interaction bugs.
    #[test]
    fn edge_all_features() {
        run_arch_test(ArchSpec {
            name: "AllFeatures",
            activation: Activation::GeLUTanh,
            norm_type: NormType::Standard,
            qk_norm: true,
            v_norm: true,
            has_ffn_norms: true,
            embed_scale: Some(5.0),
            attn_softcap: Some(30.0),
            logit_softcap: Some(20.0),
            sliding_window: Some(8),
            attention_scale: Some(0.5),
            tie_weights: true,
            rope_theta: 500_000.0,
        });
    }

    /// Minimal features — catches "default path" regressions that might be
    /// masked by feature-specific tests always having something enabled.
    #[test]
    fn edge_minimal() {
        run_arch_test(ArchSpec {
            name: "Minimal",
            ..Default::default()
        });
    }
}
