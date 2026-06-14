use std::sync::{Arc, LazyLock, Mutex};

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

static MEMORY_GAUGE_LOCK: Mutex<()> = Mutex::new(());

fn refresh_memory_gauges<'a>(models: impl Iterator<Item = (&'a str, u64, u64)>) {
    MODEL_WEIGHTS_BYTES.reset();
    KV_CACHE_ALLOCATED_BYTES.reset();
    let mut total_bytes: u64 = 0;
    for (id, weights_bytes, kv_bytes) in models {
        MODEL_WEIGHTS_BYTES
            .with_label_values(&[id])
            .set(weights_bytes as f64);
        KV_CACHE_ALLOCATED_BYTES
            .with_label_values(&[id])
            .set(kv_bytes as f64);
        total_bytes += weights_bytes + kv_bytes;
    }
    VRAM_USED_BYTES.set(total_bytes as f64);
}

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

    let buf = {
        let _guard = MEMORY_GAUGE_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        refresh_memory_gauges(running.iter().map(|info| {
            (
                info.id.as_str(),
                info.weights_size_bytes as u64,
                info.kv_cache_bytes as u64,
            )
        }));
        let encoder = TextEncoder::new();
        let mut buf = Vec::new();
        if let Err(e) = encoder.encode(&prometheus::gather(), &mut buf) {
            tracing::warn!(error = %e, "failed to encode prometheus metrics");
        }
        buf
    };
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        buf,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gauge_value(metric_name: &str, model: &str) -> Option<f64> {
        prometheus::gather()
            .into_iter()
            .find(|mf| mf.name() == metric_name)
            .and_then(|mf| {
                mf.metric.iter().find_map(|m| {
                    let matches = m
                        .label
                        .iter()
                        .any(|l| l.name() == "model" && l.value() == model);
                    matches.then(|| m.gauge.value())
                })
            })
    }

    #[test]
    fn unloaded_model_gauge_series_is_dropped() {
        let model = "stale-series-contract-model";
        let _guard = MEMORY_GAUGE_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        refresh_memory_gauges([(model, 4096u64, 1024u64)].into_iter());
        assert_eq!(
            gauge_value("oxydllm_model_weights_bytes", model),
            Some(4096.0),
            "series must be exported while the model is loaded"
        );
        assert_eq!(
            gauge_value("oxydllm_kv_cache_allocated_bytes", model),
            Some(1024.0),
        );

        refresh_memory_gauges(std::iter::empty());
        assert_eq!(
            gauge_value("oxydllm_model_weights_bytes", model),
            None,
            "stale weights series must be removed after the model is unloaded"
        );
        assert_eq!(
            gauge_value("oxydllm_kv_cache_allocated_bytes", model),
            None,
            "stale kv-cache series must be removed after the model is unloaded"
        );
    }
}
