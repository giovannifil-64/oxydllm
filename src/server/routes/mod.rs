mod chat;
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
use axum::http::StatusCode;
use axum::response::Json;
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

struct AppState {
    manager: SharedModelManager,
}

fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(handlers::health))
        .route("/metrics", get(metrics::serve_metrics))
        .route("/v1/models", get(handlers::list_models))
        .route("/v1/models/running", get(handlers::list_running_models))
        .route("/v1/models/{*model_id}", get(handlers::get_model))
        .route("/v1/chat/completions", post(chat::chat_completions))
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
        },
    )));

    let state = Arc::new(AppState {
        manager: Arc::clone(&manager),
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
