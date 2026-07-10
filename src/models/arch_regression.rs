//
// arch_regression.rs: Architecture regression test matrix
//
// Each test builds a tiny StandardTransformer (hidden=32, 2 layers, 4 heads)
// with random weights matching a specific architecture's feature combination,
// runs a prefill + decode forward pass, and verifies:
//   1. Output shape is correct
//   2. No NaN/Inf in logits
//   3. Decode after prefill doesn't crash
//
// This catches regressions when modifying shared code paths (attention, FFN,
// block, norm): if adding Qwen5 breaks Gemma2's softcap+sliding window
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
        residual_multiplier: Option<f64>,
        logits_scaling: Option<f64>,
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
                residual_multiplier: None,
                logits_scaling: None,
                tie_weights: false,
                rope_theta: 10_000.0,
            }
        }
    }

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

            t.insert(format!("{p}.input_layernorm.weight"), ones_t(h, dev));
            t.insert(
                format!("{p}.post_attention_layernorm.weight"),
                ones_t(h, dev),
            );

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

    fn build_model(spec: ArchSpec) -> Result<StandardTransformer> {
        let dev = Device::Cpu;
        let tensors = build_weights(&spec, &dev);
        build_model_from_tensors(&spec, tensors, None)
    }

    fn build_model_from_tensors(
        spec: &ArchSpec,
        tensors: rustc_hash::FxHashMap<String, Tensor>,
        kv_quantizer: Option<Arc<crate::common::kv_quant::KvQuantizer>>,
    ) -> Result<StandardTransformer> {
        let dev = Device::Cpu;
        let dtype = DType::F32;
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
            residual_multiplier: spec.residual_multiplier,
            v_norm: spec.v_norm,
            has_ffn_norms: spec.has_ffn_norms,
            sliding_window: spec.sliding_window,
            moe: None,
            linear_attn: None,
            attn_output_gate: false,
            rotary_dim: None,
            gguf_qk_permuted: false,
        };

        let blocks = (0..LAYERS)
            .map(|i| TransformerBlock::load(&block_cfg, i, &weights))
            .collect::<Result<Vec<_>>>()?;

        let norm = RMSNorm::load(&weights, "model.norm", 1e-5, spec.norm_type)?;
        let embed_tokens = Embedding::new(weights.get("model.embed_tokens.weight")?.clone());

        let lm_head = if spec.tie_weights {
            AnyLinear::from_weight(weights.get("model.embed_tokens.weight")?.clone(), None)?
        } else {
            AnyLinear::from_weight(weights.get("lm_head.weight")?.clone(), None)?
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
                    kv_quantizer.clone(),
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
            logits_scaling: spec.logits_scaling,
            per_layer_input_embed: None,
            per_layer_input_embed_scale: None,
            per_layer_model_projection: None,
            per_layer_model_projection_scale: None,
            per_layer_projection_norm: None,
            per_layer_input_scale: None,
            kv_shared_layer_map: None,
            has_recurrent_state: false,
        })
    }

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
        let logits = model.forward_batch(&input, &pos, &mut slices, &[seq_len])?;
        crate::common::block::flush_caches(&mut slices)?;
        Ok(logits)
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

        let prefill_logits = run_prefill(&model, &mut caches, 8)
            .unwrap_or_else(|e| panic!("[{name}] prefill failed: {e}"));
        assert_logits_ok(&prefill_logits, 8, name);

        for step in 0..3 {
            let decode_logits = run_decode(&model, &mut caches, 8 + step)
                .unwrap_or_else(|e| panic!("[{name}] decode step {step} failed: {e}"));
            assert_logits_ok(&decode_logits, 1, name);
        }

        clear_caches(&mut caches);
    }

    #[test]
    fn arch_llama() {
        run_arch_test(ArchSpec {
            name: "Llama",
            activation: Activation::SiLU,
            rope_theta: 500_000.0,
            ..Default::default()
        });
    }

    #[test]
    fn arch_mistral() {
        run_arch_test(ArchSpec {
            name: "Mistral",
            activation: Activation::SiLU,
            sliding_window: Some(16),
            ..Default::default()
        });
    }

    #[test]
    fn arch_qwen2() {
        run_arch_test(ArchSpec {
            name: "Qwen2",
            activation: Activation::SiLU,
            rope_theta: 1_000_000.0,
            ..Default::default()
        });
    }

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

    #[test]
    fn arch_granite() {
        run_arch_test(ArchSpec {
            name: "Granite",
            activation: Activation::SiLU,
            embed_scale: Some(12.0),
            attention_scale: Some(1.0 / HEAD_DIM as f64),
            residual_multiplier: Some(0.22),
            logits_scaling: Some(8.0),
            tie_weights: true,
            rope_theta: 10_000_000.0,
            ..Default::default()
        });
    }

    /// Contract: `logits_scaling = s` divides the final logits by exactly `s`
    /// (Granite semantics), leaving everything else untouched.
    #[test]
    fn granite_logits_scaling_divides() {
        let spec_off = ArchSpec {
            name: "logits_scaling_off",
            ..Default::default()
        };
        let dev = Device::Cpu;
        let tensors = build_weights(&spec_off, &dev);
        let tensors2: rustc_hash::FxHashMap<String, Tensor> = tensors
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let model_off = build_model_from_tensors(&spec_off, tensors, None).unwrap();
        let spec_on = ArchSpec {
            name: "logits_scaling_on",
            logits_scaling: Some(8.0),
            ..Default::default()
        };
        let model_on = build_model_from_tensors(&spec_on, tensors2, None).unwrap();

        let mut caches_off = make_caches(&model_off);
        let mut caches_on = make_caches(&model_on);
        let logits_off = run_prefill(&model_off, &mut caches_off, 8).unwrap();
        let logits_on = run_prefill(&model_on, &mut caches_on, 8).unwrap();

        let off: Vec<f32> = logits_off.flatten_all().unwrap().to_vec1().unwrap();
        let on: Vec<f32> = logits_on.flatten_all().unwrap().to_vec1().unwrap();
        let max_diff = off
            .iter()
            .zip(on.iter())
            .map(|(a, b)| (a / 8.0 - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-6,
            "logits_scaling must divide logits by exactly 8: max_abs_diff = {max_diff:.8}"
        );
    }

    #[test]
    fn arch_phi3() {
        run_arch_test(ArchSpec {
            name: "Phi3",
            activation: Activation::SiLU,
            ..Default::default()
        });
    }

    #[test]
    fn edge_sliding_window_plus_softcap() {
        run_arch_test(ArchSpec {
            name: "SlidingWindow+Softcap",
            activation: Activation::GeLUTanh,
            norm_type: NormType::Gemma,
            attn_softcap: Some(50.0),
            sliding_window: Some(4),
            tie_weights: true,
            embed_scale: Some((HIDDEN as f64).sqrt()),
            ..Default::default()
        });
    }

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
            residual_multiplier: Some(0.5),
            logits_scaling: Some(2.0),
            tie_weights: true,
            rope_theta: 500_000.0,
        });
    }

    #[test]
    fn edge_minimal() {
        run_arch_test(ArchSpec {
            name: "Minimal",
            ..Default::default()
        });
    }

    fn softmax(v: &[f32]) -> Vec<f32> {
        let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = v.iter().map(|x| (x - max).exp()).collect();
        let s: f32 = e.iter().sum();
        e.iter().map(|x| x / s).collect()
    }

    fn softmax_l1(a: &[f32], b: &[f32]) -> f32 {
        softmax(a)
            .iter()
            .zip(softmax(b))
            .map(|(x, y)| (x - y).abs())
            .sum()
    }

    fn softmax_l2(a: &[f32], b: &[f32]) -> f32 {
        softmax(a)
            .iter()
            .zip(softmax(b))
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    #[test]
    fn kv_quant_lossless_finite_and_close() {
        use crate::common::kv_quant::KvQuantizer;

        let spec = ArchSpec {
            name: "kv_quant_lossless",
            ..Default::default()
        };
        let dev = Device::Cpu;

        let tensors = build_weights(&spec, &dev);
        let tensors2: rustc_hash::FxHashMap<String, Tensor> = tensors
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let model_off = build_model_from_tensors(&spec, tensors, None)
            .expect("kv_quant: failed to build unquantized model");

        let quantizer = Arc::new(KvQuantizer::new_with_qjl(4, HEAD_DIM, true));
        let model_quant = build_model_from_tensors(&spec, tensors2, Some(quantizer))
            .expect("kv_quant: failed to build quantized model");

        let mut caches_off = make_caches(&model_off);
        let mut caches_quant = make_caches(&model_quant);

        run_prefill(&model_off, &mut caches_off, 8).expect("kv_quant: prefill failed (off)");
        run_prefill(&model_quant, &mut caches_quant, 8)
            .expect("kv_quant: prefill failed (lossless)");

        let off_logits =
            run_decode(&model_off, &mut caches_off, 8).expect("kv_quant: decode failed (off)");
        let quant_logits = run_decode(&model_quant, &mut caches_quant, 8)
            .expect("kv_quant: decode failed (lossless)");

        assert_logits_ok(&off_logits, 1, "kv_quant/off");
        assert_logits_ok(&quant_logits, 1, "kv_quant/lossless");

        let off_flat: Vec<f32> = off_logits.flatten_all().unwrap().to_vec1().unwrap();
        let quant_flat: Vec<f32> = quant_logits.flatten_all().unwrap().to_vec1().unwrap();
        let l1 = softmax_l1(&off_flat, &quant_flat);
        let l2 = softmax_l2(&off_flat, &quant_flat);

        assert!(
            l1 < 0.5,
            "kv_quant/lossless: softmax L1 distance {l1:.4} exceeds threshold 0.5 — \
             quantization is introducing excessive error in the KV read-back path"
        );
        assert!(
            l2 < 0.35,
            "kv_quant/lossless: softmax L2 distance {l2:.4} exceeds threshold 0.35 — \
             quantization is introducing large per-token error in the KV read-back path"
        );
    }

    #[test]
    fn kv_quant_balanced_vs_off() {
        use crate::common::kv_quant::KvQuantizer;

        let spec = ArchSpec {
            name: "kv_quant_balanced_vs_off",
            ..Default::default()
        };
        let dev = Device::Cpu;

        let tensors = build_weights(&spec, &dev);
        let tensors2: rustc_hash::FxHashMap<String, Tensor> = tensors
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let model_off = build_model_from_tensors(&spec, tensors, None)
            .expect("kv_quant/balanced_vs_off: failed to build unquantized model");

        let quantizer = Arc::new(KvQuantizer::new_with_qjl(3, HEAD_DIM, true));
        let model_quant = build_model_from_tensors(&spec, tensors2, Some(quantizer))
            .expect("kv_quant/balanced_vs_off: failed to build balanced model");

        let mut caches_off = make_caches(&model_off);
        let mut caches_quant = make_caches(&model_quant);

        run_prefill(&model_off, &mut caches_off, 8)
            .expect("kv_quant/balanced_vs_off: prefill failed (off)");
        run_prefill(&model_quant, &mut caches_quant, 8)
            .expect("kv_quant/balanced_vs_off: prefill failed (balanced)");

        let mut off_logits = None;
        let mut quant_logits = None;
        for step in 0..3 {
            off_logits = Some(
                run_decode(&model_off, &mut caches_off, 8 + step).unwrap_or_else(|e| {
                    panic!("kv_quant/balanced_vs_off: decode {step} failed (off): {e}")
                }),
            );
            quant_logits = Some(
                run_decode(&model_quant, &mut caches_quant, 8 + step).unwrap_or_else(|e| {
                    panic!("kv_quant/balanced_vs_off: decode {step} failed (balanced): {e}")
                }),
            );
        }

        let off_logits = off_logits.unwrap();
        let quant_logits = quant_logits.unwrap();
        assert_logits_ok(&off_logits, 1, "kv_quant/balanced_vs_off/off");
        assert_logits_ok(&quant_logits, 1, "kv_quant/balanced_vs_off/balanced");

        let off_flat: Vec<f32> = off_logits.flatten_all().unwrap().to_vec1().unwrap();
        let quant_flat: Vec<f32> = quant_logits.flatten_all().unwrap().to_vec1().unwrap();
        let l1 = softmax_l1(&off_flat, &quant_flat);
        let l2 = softmax_l2(&off_flat, &quant_flat);

        assert!(
            l1 < 0.7,
            "kv_quant/balanced: softmax L1 distance {l1:.4} exceeds threshold 0.7 — \
             3-bit quantization is introducing excessive error in the KV read-back path"
        );
        assert!(
            l2 < 0.5,
            "kv_quant/balanced: softmax L2 distance {l2:.4} exceeds threshold 0.5 — \
             3-bit quantization is introducing large per-token error in the KV read-back path"
        );
    }

    /// Single packed forward over `[seq_a; seq_b]` must match two sequential
    /// per-seq forwards concatenated along the seq dim; the engine relies on
    /// this when splitting prefill into multiple `forward_batch` calls.
    #[test]
    fn chunked_forward_matches_single_packed_forward() {
        let model = build_model(ArchSpec {
            name: "chunked_parity",
            ..ArchSpec::default()
        })
        .unwrap();
        let dev = model.device().clone();

        let len_a = 7usize;
        let len_b = 11usize;
        let tokens_a: Vec<u32> = (1..=len_a as u32).collect();
        let tokens_b: Vec<u32> = (10..10 + len_b as u32).collect();
        let pos_a: Vec<u32> = (0..len_a as u32).collect();
        let pos_b: Vec<u32> = (0..len_b as u32).collect();

        let mut caches_a_pkd = make_caches(&model);
        let mut caches_b_pkd = make_caches(&model);
        let combined_tokens: Vec<u32> = tokens_a.iter().chain(tokens_b.iter()).copied().collect();
        let combined_pos: Vec<u32> = pos_a.iter().chain(pos_b.iter()).copied().collect();
        let input_pkd = Tensor::from_vec(combined_tokens, (1, len_a + len_b), &dev).unwrap();
        let pos_pkd = Tensor::from_vec(combined_pos, (len_a + len_b,), &dev).unwrap();
        let mut slices_pkd: Vec<&mut [PagedKvCache]> =
            vec![caches_a_pkd.as_mut_slice(), caches_b_pkd.as_mut_slice()];
        let logits_pkd = model
            .forward_batch(&input_pkd, &pos_pkd, &mut slices_pkd, &[len_a, len_b])
            .unwrap();
        crate::common::block::flush_caches(&mut slices_pkd).unwrap();

        let mut caches_a_chk = make_caches(&model);
        let mut caches_b_chk = make_caches(&model);
        let input_a = Tensor::from_vec(tokens_a.clone(), (1, len_a), &dev).unwrap();
        let pos_a_t = Tensor::from_vec(pos_a.clone(), (len_a,), &dev).unwrap();
        let mut slices_a: Vec<&mut [PagedKvCache]> = vec![caches_a_chk.as_mut_slice()];
        let logits_a = model
            .forward_batch(&input_a, &pos_a_t, &mut slices_a, &[len_a])
            .unwrap();

        let input_b = Tensor::from_vec(tokens_b.clone(), (1, len_b), &dev).unwrap();
        let pos_b_t = Tensor::from_vec(pos_b.clone(), (len_b,), &dev).unwrap();
        let mut slices_b: Vec<&mut [PagedKvCache]> = vec![caches_b_chk.as_mut_slice()];
        let logits_b = model
            .forward_batch(&input_b, &pos_b_t, &mut slices_b, &[len_b])
            .unwrap();

        let logits_chk = Tensor::cat(&[&logits_a, &logits_b], 1).unwrap();

        let mut all_chk: Vec<&mut [PagedKvCache]> =
            vec![caches_a_chk.as_mut_slice(), caches_b_chk.as_mut_slice()];
        crate::common::block::flush_caches(&mut all_chk).unwrap();

        let p = logits_pkd.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let c = logits_chk.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(p.len(), c.len(), "chunked vs packed shape mismatch");
        let max_diff = p
            .iter()
            .zip(c.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "chunked forward diverges from packed: max_abs_diff = {max_diff:.6}"
        );
    }
}
