//! The OpenAI-compatible `/v1/embeddings` endpoint.
//!
//! Serves encoder-only embedding models ([`crate::models::encoder`]). Encoders
//! load on first use and stay resident in the manager's encoder cache; the
//! forward passes run on a blocking thread under the device GPU lock, so they
//! interleave safely with decoder traffic.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use candle_core::DType;
use serde::Deserialize;

use super::{AppState, error_response};
use crate::models::manager::EncoderHandle;

#[derive(Deserialize)]
pub(super) struct EmbeddingsRequest {
    model: String,
    input: EmbeddingsInput,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum EmbeddingsInput {
    One(String),
    Many(Vec<String>),
}

pub(super) async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(body): Json<EmbeddingsRequest>,
) -> Response {
    let inputs = match body.input {
        EmbeddingsInput::One(s) => vec![s],
        EmbeddingsInput::Many(v) => v,
    };
    if inputs.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "'input' must contain at least one string",
            "invalid_request_error",
        )
        .into_response();
    }

    let handle = match get_or_load_encoder(&state, &body.model).await {
        Ok(h) => h,
        Err(resp) => return resp,
    };

    let batch = inputs.clone();
    let handle_for_embed = handle.clone();
    let embedded =
        tokio::task::spawn_blocking(move || -> anyhow::Result<(Vec<Vec<f32>>, usize)> {
            let gpu = crate::gpu_lock::gpu_lock_for(handle_for_embed.model.device());
            let _gpu = gpu.acquire();
            let mut vectors = Vec::with_capacity(batch.len());
            let mut prompt_tokens = 0usize;
            for text in &batch {
                let ids = handle_for_embed
                    .tokenizer
                    .encode_with_special_tokens(text)?;
                prompt_tokens += ids.len();
                vectors.push(handle_for_embed.model.embed(&ids)?);
            }
            Ok((vectors, prompt_tokens))
        })
        .await;

    let (vectors, prompt_tokens) = match embedded {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!("embedding failed: {e:#}"),
                "invalid_request_error",
            )
            .into_response();
        }
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("embedding task failed: {e}"),
                "server_error",
            )
            .into_response();
        }
    };

    let data: Vec<serde_json::Value> = vectors
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| {
            serde_json::json!({
                "object": "embedding",
                "index": index,
                "embedding": embedding,
            })
        })
        .collect();
    Json(serde_json::json!({
        "object": "list",
        "model": body.model,
        "data": data,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
        }
    }))
    .into_response()
}

/// Returns the cached encoder for `model_id`, loading it on a blocking thread
/// on first use. Concurrent first requests may both load; the second insert
/// wins and the duplicate is dropped, which is benign for these model sizes.
async fn get_or_load_encoder(
    state: &Arc<AppState>,
    model_id: &str,
) -> Result<EncoderHandle, Response> {
    if let Some(h) = state.manager.lock().await.encoder_cached(model_id) {
        return Ok(h);
    }
    let (dir, device_idx, require_gpu) =
        match state.manager.lock().await.encoder_load_info(model_id) {
            Ok(info) => info,
            Err(e) => {
                return Err(
                    error_response(StatusCode::NOT_FOUND, e, "model_not_found").into_response()
                );
            }
        };

    let loaded = tokio::task::spawn_blocking(move || -> anyhow::Result<EncoderHandle> {
        let device = crate::models::loader::select_device_at(device_idx.unwrap_or(0), require_gpu)?;
        let dtype = if device.is_cpu() {
            DType::F32
        } else {
            DType::BF16
        };
        let model = crate::models::encoder::EncoderModel::load(&dir, &device, dtype)?;
        let tokenizer = crate::tokenizer::Tokenizer::from_dir(&dir)?;
        Ok(EncoderHandle {
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
        })
    })
    .await;

    match loaded {
        Ok(Ok(handle)) => {
            state
                .manager
                .lock()
                .await
                .insert_encoder(model_id, handle.clone());
            tracing::info!(
                model_id,
                dimensions = handle.model.hidden_size(),
                "encoder model loaded"
            );
            Ok(handle)
        }
        Ok(Err(e)) => Err(error_response(
            StatusCode::BAD_REQUEST,
            format!("failed to load encoder model: {e:#}"),
            "invalid_request_error",
        )
        .into_response()),
        Err(e) => Err(error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("encoder load task failed: {e}"),
            "server_error",
        )
        .into_response()),
    }
}
