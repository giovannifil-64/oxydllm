//! OpenTelemetry tracing wiring, additive to the existing Prometheus metrics
//! and `tracing` logs.
//!
//! When an OTLP endpoint is configured (via `--otel-endpoint`, `OXYDLLM_OTEL_ENDPOINT`,
//! or the standard `OTEL_EXPORTER_OTLP_ENDPOINT`), per-request inference spans are
//! exported over OTLP/HTTP to a collector or backend (Grafana Tempo, Jaeger, etc.).
//! Without an endpoint the server behaves exactly as before: logs only, no exporter.

use opentelemetry::Context;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_http::HeaderExtractor;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::SdkTracerProvider;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::prelude::*;

/// Resolve the OTLP traces endpoint from CLI args (`--otel-endpoint <url>`) or,
/// failing that, the `OXYDLLM_OTEL_ENDPOINT` / `OTEL_EXPORTER_OTLP_ENDPOINT`
/// environment variables. Returns `None` when tracing export is not requested.
pub fn resolve_endpoint(args: &[String]) -> Option<String> {
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--otel-endpoint" {
            return args.get(i + 1).cloned();
        }
        i += 1;
    }
    std::env::var("OXYDLLM_OTEL_ENDPOINT")
        .or_else(|_| std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT"))
        .ok()
        .filter(|s| !s.trim().is_empty())
}

/// Install the global tracing subscriber: a fmt layer (compact, or JSON when
/// `LOG_FORMAT=json`) plus, when `otel_endpoint` is set, an OpenTelemetry layer
/// exporting spans over OTLP/HTTP. Returns the tracer provider, which the caller
/// must hand to [`shutdown`] on exit to flush buffered spans.
pub fn init(otel_endpoint: Option<&str>) -> Option<SdkTracerProvider> {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("oxydllm=info,hyper=warn,tower=warn"));

    // LOG_FORMAT=json emits one JSON object per line for Loki/Datadog/`jq`.
    // RUST_LOG governs only the log (fmt) layer; the OTLP layer below keeps its own
    // filter so traces are not silently dropped under RUST_LOG=warn (the default in
    // the install.sh service templates).
    let json = std::env::var("LOG_FORMAT").as_deref() == Ok("json");
    let fmt_layer = if json {
        tracing_subscriber::fmt::layer()
            .json()
            .with_target(false)
            .with_filter(env_filter)
            .boxed()
    } else {
        tracing_subscriber::fmt::layer()
            .compact()
            .with_target(false)
            .with_filter(env_filter)
            .boxed()
    };

    let (otel_layer, provider) = match otel_endpoint {
        Some(endpoint) => match build_provider(endpoint) {
            Ok(provider) => {
                // W3C Trace Context propagator so an upstream `traceparent` header
                // links the request's spans into the caller's distributed trace.
                opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());
                let tracer = provider.tracer("oxydllm");
                (
                    Some(tracing_opentelemetry::layer().with_tracer(tracer)),
                    Some(provider),
                )
            }
            Err(e) => {
                eprintln!(
                    "oxydllm: OpenTelemetry exporter init failed ({e}); continuing without trace export"
                );
                (None, None)
            }
        },
        None => (None, None),
    };

    let _ = tracing_subscriber::registry()
        .with(fmt_layer)
        .with(otel_layer.map(|layer| layer.with_filter(EnvFilter::new("oxydllm=info"))))
        .try_init();

    if let Some(endpoint) = otel_endpoint.filter(|_| provider.is_some()) {
        tracing::info!(endpoint, "OpenTelemetry OTLP trace export enabled");
    }
    provider
}

fn build_provider(endpoint: &str) -> anyhow::Result<SdkTracerProvider> {
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .with_endpoint(traces_endpoint(endpoint))
        .build()?;
    Ok(SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(Resource::builder().with_service_name("oxydllm").build())
        .build())
}

/// Treat the configured endpoint as the OTLP base (e.g. `http://localhost:4318`)
/// and target the traces signal path, matching the `OTEL_EXPORTER_OTLP_ENDPOINT`
/// convention. A builder-set endpoint is otherwise used verbatim, so collectors
/// listening on `/v1/traces` would never be hit.
fn traces_endpoint(endpoint: &str) -> String {
    let trimmed = endpoint.trim_end_matches('/');
    if trimmed.ends_with("/v1/traces") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/v1/traces")
    }
}

/// Extract the upstream trace context from incoming request headers (W3C
/// `traceparent` / `tracestate`). Returns an empty context when no valid header
/// is present or tracing is disabled, in which case the request span starts a
/// fresh trace.
pub fn extract_context(headers: &axum::http::HeaderMap) -> Context {
    opentelemetry::global::get_text_map_propagator(|propagator| {
        propagator.extract(&HeaderExtractor(headers))
    })
}

/// Flush and shut down the tracer provider so buffered spans are exported before
/// the process exits. Safe to call once; logs a warning on failure.
pub fn shutdown(provider: SdkTracerProvider) {
    if let Err(e) = provider.shutdown() {
        tracing::warn!(error = %e, "OpenTelemetry flush/shutdown failed");
    }
}

#[cfg(test)]
mod tests {
    use super::traces_endpoint;

    // Contract: a base OTLP endpoint is sent the traces signal path so real
    // collectors (which serve /v1/traces) receive the export, while an
    // already-qualified endpoint is left untouched (no double /v1/traces).
    #[test]
    fn base_endpoint_gets_traces_signal_path() {
        assert_eq!(
            traces_endpoint("http://localhost:4318"),
            "http://localhost:4318/v1/traces"
        );
        assert_eq!(
            traces_endpoint("http://localhost:4318/"),
            "http://localhost:4318/v1/traces"
        );
        assert_eq!(
            traces_endpoint("http://localhost:4318/v1/traces"),
            "http://localhost:4318/v1/traces"
        );
    }
}
