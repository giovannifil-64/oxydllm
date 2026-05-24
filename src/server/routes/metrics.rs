use std::sync::{Arc, LazyLock};

use axum::extract::State;
use axum::http::header;
use axum::response::IntoResponse;
use prometheus::{
    CounterVec, Encoder, Gauge, GaugeVec, HistogramVec, TextEncoder, register_counter_vec,
    register_gauge, register_gauge_vec, register_histogram_vec,
};

use super::AppState;

pub static TTFT_HISTOGRAM: LazyLock<HistogramVec> = LazyLock::new(|| {
    register_histogram_vec!(
        "oxydllm_ttft_milliseconds",
        "Time-to-first-token in milliseconds. Includes prefill and queue wait time. \
         Labeled by model.",
        &["model"],
        vec![10.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]
    )
    .expect("failed to register oxydllm_ttft_milliseconds")
});

pub static TPS_HISTOGRAM: LazyLock<HistogramVec> = LazyLock::new(|| {
    register_histogram_vec!(
        "oxydllm_tokens_per_second",
        "Decode throughput in tokens/s from first token to completion. \
         Labeled by model.",
        &["model"],
        vec![1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    )
    .expect("failed to register oxydllm_tokens_per_second")
});

pub static REQUESTS_TOTAL: LazyLock<CounterVec> = LazyLock::new(|| {
    register_counter_vec!(
        "oxydllm_requests_total",
        "Total completed chat completion requests. Labeled by model and status (ok/error).",
        &["model", "status"]
    )
    .expect("failed to register oxydllm_requests_total")
});

pub static QUEUE_DEPTH: LazyLock<Gauge> = LazyLock::new(|| {
    register_gauge!(
        "oxydllm_queue_depth",
        "Current number of sequences in the inference engine (waiting + running). \
         Updated each engine step."
    )
    .expect("failed to register oxydllm_queue_depth")
});

pub static PREFIX_CACHE_REQUESTS: LazyLock<CounterVec> = LazyLock::new(|| {
    register_counter_vec!(
        "oxydllm_prefix_cache_requests_total",
        "Prefix KV cache lookups, split by result (hit/miss). \
         Compute hit ratio as rate(hit) / rate(hit+miss). Labeled by model.",
        &["model", "result"]
    )
    .expect("failed to register oxydllm_prefix_cache_requests_total")
});

pub static MODEL_WEIGHTS_BYTES: LazyLock<GaugeVec> = LazyLock::new(|| {
    register_gauge_vec!(
        "oxydllm_model_weights_bytes",
        "Model weight memory in bytes (set at load, cleared at unload). \
         On Apple Silicon this is unified memory, not a discrete VRAM pool. \
         Labeled by model.",
        &["model"]
    )
    .expect("failed to register oxydllm_model_weights_bytes")
});

pub static KV_CACHE_ALLOCATED_BYTES: LazyLock<GaugeVec> = LazyLock::new(|| {
    register_gauge_vec!(
        "oxydllm_kv_cache_allocated_bytes",
        "KV cache memory reserved in bytes at model load time. \
         Not the dynamically occupied portion — see queue_depth for utilisation. \
         On Apple Silicon this is unified memory, not a discrete VRAM pool. \
         Labeled by model.",
        &["model"]
    )
    .expect("failed to register oxydllm_kv_cache_allocated_bytes")
});

pub static VRAM_USED_BYTES: LazyLock<Gauge> = LazyLock::new(|| {
    register_gauge!(
        "oxydllm_vram_used_bytes",
        "Total inference memory in bytes: model weights + KV cache for all loaded models. \
         On Apple Silicon this is unified system memory, not a discrete VRAM pool. \
         For per-model breakdown see oxydllm_model_weights_bytes and \
         oxydllm_kv_cache_allocated_bytes."
    )
    .expect("failed to register oxydllm_vram_used_bytes")
});

pub(super) async fn serve_metrics(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    // Touch all statics so they appear in output even before any request comes in.
    let _ = &*TTFT_HISTOGRAM;
    let _ = &*TPS_HISTOGRAM;
    let _ = &*REQUESTS_TOTAL;
    let _ = &*QUEUE_DEPTH;
    let _ = &*PREFIX_CACHE_REQUESTS;
    let _ = &*MODEL_WEIGHTS_BYTES;
    let _ = &*KV_CACHE_ALLOCATED_BYTES;
    let _ = &*VRAM_USED_BYTES;

    let running = state.manager.lock().await.list_running();
    let mut total_bytes: u64 = 0;
    for info in &running {
        MODEL_WEIGHTS_BYTES
            .with_label_values(&[&info.id])
            .set(info.weights_size_bytes as f64);
        KV_CACHE_ALLOCATED_BYTES
            .with_label_values(&[&info.id])
            .set(info.kv_cache_bytes as f64);
        total_bytes += (info.weights_size_bytes + info.kv_cache_bytes) as u64;
    }
    VRAM_USED_BYTES.set(total_bytes as f64);

    let encoder = TextEncoder::new();
    let mut buf = Vec::new();
    if let Err(e) = encoder.encode(&prometheus::gather(), &mut buf) {
        tracing::warn!(error = %e, "failed to encode prometheus metrics");
    }
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        buf,
    )
}
