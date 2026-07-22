mod chat;
mod embeddings;
mod engine_loop;
mod handlers;
#[cfg(test)]
mod http_compat_tests;
mod metrics;
mod types;

pub use chat::apply_chat_template;
pub use engine_loop::engine_loop;
pub use types::{ChatMessage, IncomingRequest};

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};

async fn wait_for_shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut sigterm =
            signal(SignalKind::terminate()).expect("failed to install SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = sigterm.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    }
}

use crate::models::manager::{self, ModelManager, ModelManagerConfig, SharedModelManager};

pub(super) struct AppState {
    pub(super) manager: SharedModelManager,
    /// Optional API bearer key. When `Some`, requests must include
    /// `Authorization: Bearer <key>` (or `X-API-Key: <key>`) on protected
    /// endpoints. `None` disables authentication entirely.
    pub(super) api_key: Option<String>,
    /// Wall-clock per-request timeout. `None` disables the timeout.
    pub(super) request_timeout: Option<Duration>,
}

/// Constant-time slice comparison used for API-key checks.
///
/// Always inspects every byte of the longer slice so the running time does not
/// leak the position of the first mismatching byte.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Extract a presented API key from `Authorization: Bearer <key>` or the
/// `X-API-Key` header. The bearer scheme is preferred; the custom header is
/// kept for clients that cannot set `Authorization`.
fn extract_api_key(req: &Request<Body>) -> Option<String> {
    if let Some(auth) = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        && let Some(rest) = auth
            .strip_prefix("Bearer ")
            .or_else(|| auth.strip_prefix("bearer "))
    {
        return Some(rest.trim().to_string());
    }
    req.headers()
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim().to_string())
}

async fn require_api_key(
    State(state): State<Arc<AppState>>,
    req: Request<Body>,
    next: Next,
) -> Response {
    let Some(expected) = state.api_key.as_deref() else {
        return next.run(req).await;
    };
    let presented = extract_api_key(&req);
    let ok = presented
        .as_deref()
        .map(|p| constant_time_eq(p.as_bytes(), expected.as_bytes()))
        .unwrap_or(false);
    if !ok {
        return error_response(
            StatusCode::UNAUTHORIZED,
            "missing or invalid API key",
            "invalid_api_key",
        )
        .into_response();
    }
    next.run(req).await
}

fn build_router(state: Arc<AppState>) -> Router {
    let protected = Router::new()
        .route("/metrics", get(metrics::serve_metrics))
        .route("/v1/models", get(handlers::list_models))
        .route("/v1/models/running", get(handlers::list_running_models))
        .route("/v1/models/{*model_id}", get(handlers::get_model))
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route("/v1/embeddings", post(embeddings::embeddings))
        .route_layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            require_api_key,
        ));

    Router::new()
        .route("/health", get(handlers::health))
        .merge(protected)
        .with_state(state)
}

fn error_response(
    status: StatusCode,
    message: impl Into<String>,
    error_type: &str,
) -> (StatusCode, Json<serde_json::Value>) {
    (
        status,
        Json(serde_json::json!({
            "error": {
                "message": message.into(),
                "type": error_type,
                "param": null,
                "code": null,
            }
        })),
    )
}

pub struct StartServerArgs {
    pub models_dir: PathBuf,
    pub port: u16,
    pub keep_alive: Duration,
    pub shutdown_timeout: Duration,
    pub memory_budget_bytes: Option<usize>,
    pub cuda_devices: Vec<usize>,
    pub max_context_len: usize,
    pub kv_quant: crate::common::kv_quant::KvQuantMode,
    pub qjl_quantization: bool,
    pub require_gpu: bool,
    pub max_num_seqs: Option<usize>,
    pub max_queued_requests: usize,
    /// Optional API bearer key. When `Some`, the HTTP API requires
    /// `Authorization: Bearer <key>` (or `X-API-Key: <key>`) on every endpoint
    /// except `/health`.
    pub api_key: Option<String>,
    /// Wall-clock per-request timeout. `None` disables the timeout.
    pub request_timeout: Option<Duration>,
    /// When set, MoE expert weights stream from the checkpoint mmap through an
    /// LRU pool of this many megabytes instead of loading resident.
    pub expert_stream_mb: Option<usize>,
    /// Optional speculative-decoding draft model id, applied to every loaded
    /// model whose vocab matches.
    pub draft_model: Option<String>,
}

pub fn start_server(args: StartServerArgs) -> anyhow::Result<()> {
    let StartServerArgs {
        models_dir,
        port,
        keep_alive,
        shutdown_timeout,
        memory_budget_bytes,
        cuda_devices,
        max_context_len,
        kv_quant,
        qjl_quantization,
        require_gpu,
        max_num_seqs,
        max_queued_requests,
        api_key,
        request_timeout,
        expert_stream_mb,
        draft_model,
    } = args;

    if !models_dir.exists() {
        std::fs::create_dir_all(&models_dir)?;
        tracing::info!(path = %models_dir.display(), "created models directory");
    }
    let available = crate::models::loader::discover_models(&models_dir);
    tracing::info!(path = %models_dir.display(), "models directory");
    tracing::info!(
        discovered_models = available.len(),
        "discovered local models"
    );
    for m in &available {
        tracing::info!(model_id = %m.id, architecture = %m.architecture, "discovered model");
    }

    let manager = Arc::new(tokio::sync::Mutex::new(ModelManager::new(
        ModelManagerConfig {
            models_dir,
            keep_alive,
            memory_budget_bytes,
            cuda_devices,
            max_context_len,
            kv_quant,
            qjl_quantization,
            require_gpu,
            max_num_seqs,
            max_queued_requests,
            draft_model,
            expert_stream_mb,
        },
    )));

    let state = Arc::new(AppState {
        manager: Arc::clone(&manager),
        api_key,
        request_timeout,
    });

    let app = build_router(state);

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        manager::spawn_eviction_task(manager);

        let addr = format!("0.0.0.0:{}", port);
        let api_endpoint = format!("http://localhost:{port}/v1/chat/completions");
        let models_endpoint = format!("http://localhost:{port}/v1/models");
        let running_models_endpoint = format!("http://localhost:{port}/v1/models/running");
        let health_endpoint = format!("http://localhost:{port}/health");
        let metrics_endpoint = format!("http://localhost:{port}/metrics");

        tracing::info!(address = %addr, "server listening");
        tracing::info!(method = "POST", endpoint = %api_endpoint, "API endpoint");
        tracing::info!(method = "GET", endpoint = %models_endpoint, "models endpoint");
        tracing::info!(
            method = "GET",
            endpoint = %running_models_endpoint,
            "running models endpoint"
        );
        tracing::info!(method = "GET", endpoint = %health_endpoint, "health endpoint");
        tracing::info!(method = "GET", endpoint = %metrics_endpoint, "Prometheus metrics endpoint");
        tracing::info!(
            keep_alive_s = keep_alive.as_secs(),
            "models evicted after keep-alive timeout"
        );
        match memory_budget_bytes {
            Some(b) => tracing::info!(
                memory_budget_gb = ((b as f64 / 1_073_741_824.0) * 10.0).round() / 10.0,
                "memory budget configured (LRU eviction when exceeded)"
            ),
            None => tracing::info!("memory budget: unlimited"),
        }
        tracing::info!(
            max_context_len,
            "max context length configured (tokens per sequence)"
        );

        let listener = tokio::net::TcpListener::bind(&addr).await?;

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        tokio::spawn(async move {
            wait_for_shutdown_signal().await;
            tracing::info!(
                drain_timeout_s = shutdown_timeout.as_secs(),
                "shutdown signal received, draining in-flight requests"
            );
            let _ = shutdown_tx.send(());
            tokio::time::sleep(shutdown_timeout).await;
            tracing::warn!("graceful shutdown timed out, forcing exit");
            std::process::exit(1);
        });

        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await?;

        tracing::info!("server shut down cleanly");
        Ok::<_, anyhow::Error>(())
    })?;

    Ok(())
}
