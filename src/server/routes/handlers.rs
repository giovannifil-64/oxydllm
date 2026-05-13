use std::sync::Arc;

use axum::extract::{Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json};

use super::AppState;
use super::error_response;

pub(super) async fn health() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok"}))
}

pub(super) async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mgr = state.manager.lock().await;
    let models_dir = mgr.models_dir().clone();
    let registry = mgr.list_registry().clone();
    drop(mgr);

    let discovered = crate::models::loader::discover_models(&models_dir);

    let data: Vec<serde_json::Value> = discovered
        .iter()
        .map(|m| {
            let size_bytes = registry.get(&m.id).map(|e| e.size_bytes).unwrap_or(0);
            let last_used_secs = registry.get(&m.id).map(|e| e.last_used_secs).unwrap_or(0);
            let owned_by =
                m.id.split_once('/')
                    .map(|(ns, _)| ns.to_string())
                    .unwrap_or_else(|| "local".to_string());
            serde_json::json!({
                "id": m.id,
                "object": "model",
                "created": m.created_at,
                "owned_by": owned_by,
                "architecture": m.architecture,
                "vocab_size": m.vocab_size,
                "num_layers": m.num_layers,
                "size_bytes": size_bytes,
                "size_gb": (size_bytes as f64 / 1_073_741_824.0 * 100.0).round() / 100.0,
                "last_used_secs": last_used_secs,
            })
        })
        .collect();

    Json(serde_json::json!({
        "object": "list",
        "data": data
    }))
}

pub(super) async fn get_model(
    State(state): State<Arc<AppState>>,
    AxumPath(model_id): AxumPath<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let mgr = state.manager.lock().await;
    let models_dir = mgr.models_dir().clone();
    let registry = mgr.list_registry().clone();
    drop(mgr);

    let discovered = crate::models::loader::discover_models(&models_dir);
    let model = discovered.iter().find(|m| m.id == model_id);

    match model {
        Some(m) => {
            let size_bytes = registry.get(&m.id).map(|e| e.size_bytes).unwrap_or(0);
            let owned_by =
                m.id.split_once('/')
                    .map(|(ns, _)| ns.to_string())
                    .unwrap_or_else(|| "local".to_string());
            Ok(Json(serde_json::json!({
                "id": m.id,
                "object": "model",
                "created": m.created_at,
                "owned_by": owned_by,
                "architecture": m.architecture,
                "vocab_size": m.vocab_size,
                "num_layers": m.num_layers,
                "size_bytes": size_bytes,
                "size_gb": (size_bytes as f64 / 1_073_741_824.0 * 100.0).round() / 100.0,
            })))
        }
        None => Err(error_response(
            StatusCode::NOT_FOUND,
            format!("The model '{}' does not exist", model_id),
            "invalid_request_error",
        )),
    }
}

pub(super) async fn list_running_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mgr = state.manager.lock().await;
    let running = mgr.list_running();
    let budget_bytes = mgr.memory_budget_bytes();
    let total_loaded = mgr.total_loaded_bytes();
    drop(mgr);

    let data: Vec<serde_json::Value> = running
        .iter()
        .map(|m| {
            let total = m.weights_size_bytes + m.kv_cache_bytes;
            serde_json::json!({
                "id": m.id,
                "object": "model",
                "architecture": m.architecture,
                "vocab_size": m.vocab_size,
                "num_layers": m.num_layers,
                "idle_seconds": m.idle_seconds,
                "weights_size_bytes": m.weights_size_bytes,
                "kv_cache_bytes": m.kv_cache_bytes,
                "total_size_bytes": total,
                "total_size_gb": (total as f64 / 1_073_741_824.0 * 100.0).round() / 100.0,
            })
        })
        .collect();

    let mut resp = serde_json::json!({
        "object": "list",
        "data": data,
        "total_loaded_bytes": total_loaded,
        "total_loaded_gb": (total_loaded as f64 / 1_073_741_824.0 * 100.0).round() / 100.0,
    });

    if let Some(budget) = budget_bytes {
        resp["memory_budget_bytes"] = budget.into();
        resp["memory_budget_gb"] =
            ((budget as f64 / 1_073_741_824.0 * 100.0).round() / 100.0).into();
        resp["memory_free_bytes"] = budget.saturating_sub(total_loaded).into();
    }

    Json(resp)
}
