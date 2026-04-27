use serde::{Deserialize, Serialize};
use tokio::sync::mpsc as tokio_mpsc;

use crate::sampling::SamplingParams;

#[derive(Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: Option<bool>,
}

#[derive(Deserialize, Clone)]
pub struct JsonSchemaSpec {
    pub name: String,
    #[serde(default)]
    pub schema: Option<serde_json::Value>,
    #[serde(default)]
    pub strict: Option<bool>,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Deserialize, Clone)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    #[serde(default)]
    pub json_schema: Option<JsonSchemaSpec>,
}

#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub n: Option<usize>,
    #[serde(default)]
    pub stop: Option<StopParam>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub top_logprobs: Option<usize>,
    #[serde(default)]
    pub logit_bias: Option<serde_json::Value>,
    // Extensions (non-OpenAI)
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub repetition_window: Option<usize>,
    #[serde(default)]
    pub keep_alive: Option<u64>,
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum StopParam {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub reasoning_content: Option<String>,
}

/// Pre-decoded logprob entry for a single generated token.
pub struct EngineLogprobEntry {
    pub token_str: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
    /// Top-k alternatives: (token_str, logprob, bytes).
    pub top_logprobs: Vec<(String, f32, Vec<u8>)>,
}

pub struct IncomingRequest {
    pub prompt_tokens: Vec<u32>,
    pub sampling_params: SamplingParams,
    pub max_tokens: usize,
    pub response_tx: tokio_mpsc::UnboundedSender<EngineEvent>,
    pub model_id: String,
    pub enqueued_at: std::time::Instant,
    pub enable_thinking: bool,
    pub extra_stop_token_ids: Vec<u32>,
}

pub enum EngineEvent {
    Token {
        text: String,
        logprob_entries: Vec<EngineLogprobEntry>,
    },
    ReasoningToken(String),
    Finish {
        finish_reason: String,
        completion_tokens: usize,
    },
    StreamEnd,
    Error(String),
}
