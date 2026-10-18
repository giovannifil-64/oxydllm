//! OpenAI-compatible `POST /v1/chat/completions` handler and its helpers.
//!
//! [`chat_completions`] validates the request, resolves the model through the
//! [`ModelManager`](crate::models::manager::ModelManager), renders the chat
//! template ([`apply_chat_template`]), and fans out `n` sequences to the
//! engine. Around it live:
//!
//! - Tool calling: [`build_tool_config`] validates `tools`/`tool_choice`,
//!   [`tools_system_instruction`] injects the calling convention into the
//!   system prompt, and [`try_parse_tool_calls`] recovers calls from model
//!   output in both tag-block and bare-JSON shapes.
//! - Structured output: [`validate_schema_shape`] checks request-supplied
//!   schemas (including OpenAI strict-mode rules) and
//!   [`validate_against_schema`] post-validates model output against a JSON
//!   Schema subset.
//! - Serde types mirroring the OpenAI response and streaming-chunk wire
//!   shapes.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json};
use serde::Serialize;
use tokio::sync::mpsc as tokio_mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing_opentelemetry::OpenTelemetrySpanExt;

use super::AppState;
use super::error_response;
use super::types::{
    ChatCompletionRequest, ChatMessage, EngineEvent, EngineLogprobEntry, IncomingRequest,
    ResponseFormat, StopParam, ToolCall, ToolCallFunction, ToolDefinition,
};
use crate::chat_template;
use crate::models::manager::GetResult;
use crate::sampling::SamplingParams;
use crate::tokenizer::Tokenizer;

/// One top-k alternative inside a logprobs entry (OpenAI wire shape).
#[derive(Serialize, Clone)]
struct TopLogprobItem {
    token: String,
    logprob: f32,
    bytes: Option<Vec<u8>>,
}

/// Logprob record for one emitted token, with its top-k alternatives.
#[derive(Serialize, Clone)]
struct TokenLogprob {
    token: String,
    logprob: f32,
    bytes: Option<Vec<u8>>,
    top_logprobs: Vec<TopLogprobItem>,
}

/// The `logprobs` object attached to a choice or streaming chunk.
#[derive(Serialize, Clone)]
struct Logprobs {
    content: Vec<TokenLogprob>,
    refusal: Option<String>,
}

/// Assistant message in a non-streaming response. Kept separate from
/// [`ChatMessage`] so `content` serializes as an explicit `null` when
/// `tool_calls` is present, as the OpenAI API requires.
#[derive(Serialize)]
struct ResponseMessage {
    role: String,
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
}

/// Top-level non-streaming response body (`object: "chat.completion"`).
#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
    system_fingerprint: Option<String>,
}

/// One completion choice in a non-streaming response.
#[derive(Serialize)]
struct Choice {
    index: usize,
    message: ResponseMessage,
    finish_reason: String,
    logprobs: Option<Logprobs>,
}

/// The `completion_tokens_details` usage object, reporting reasoning tokens.
#[derive(Serialize)]
struct CompletionTokensDetails {
    reasoning_tokens: usize,
}

/// Token accounting for a response; `completion_tokens_details` appears only
/// when reasoning tokens were produced.
#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    completion_tokens_details: Option<CompletionTokensDetails>,
}

/// One SSE event in a streaming response (`object: "chat.completion.chunk"`).
#[derive(Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Usage>,
    system_fingerprint: Option<String>,
}

/// One choice delta within a streaming chunk.
#[derive(Serialize)]
struct ChunkChoice {
    index: usize,
    delta: Delta,
    finish_reason: Option<String>,
    logprobs: Option<Logprobs>,
}

/// Function name/arguments fragment inside a streamed tool-call delta.
#[derive(Serialize)]
struct ToolCallFunctionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    arguments: String,
}

/// Streamed tool-call fragment: the first chunk carries `id`, `type`, and
/// the function name; later chunks append arguments.
#[derive(Serialize)]
struct ToolCallDelta {
    index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    call_type: Option<String>,
    function: ToolCallFunctionDelta,
}

/// Message delta within a chunk; only populated fields are serialized.
#[derive(Serialize)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCallDelta>>,
}

/// Requested tool-selection behavior from the `tool_choice` parameter:
/// `None` disables tools, `Auto` lets the model decide, `Required` forces at
/// least one call, and `ForcedFunction` forces exactly the named function.
#[derive(Clone, Debug, PartialEq, Eq)]
enum ToolChoiceMode {
    None,
    Auto,
    Required,
    ForcedFunction { name: String },
}

/// Validated tool setup for one request: the effective tool list (already
/// filtered down by `tool_choice`), the choice mode, and whether parallel
/// calls are allowed.
#[derive(Clone, Debug)]
struct ToolConfig {
    tools: Vec<ToolDefinition>,
    choice_mode: ToolChoiceMode,
    parallel_tool_calls: bool,
}

/// Schema validation level: `Strict` applies the OpenAI strict-mode rules
/// (every object closed with `additionalProperties: false` and `required`
/// covering all properties), `BestEffort` checks structure only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SchemaStrictness {
    BestEffort,
    Strict,
}

fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Stable `system_fingerprint` value derived from the model id and the crate
/// version.
fn system_fingerprint(model_id: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    model_id.hash(&mut h);
    env!("CARGO_PKG_VERSION").hash(&mut h);
    format!("fp_{:012x}", h.finish() & 0xFFFF_FFFF_FFFF)
}

/// Generates a unique `chatcmpl-` id from the current time and a
/// process-wide counter.
fn make_chat_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("chatcmpl-{:x}{:x}-{}", t.as_secs(), t.subsec_nanos(), seq)
}

/// Strips a wrapping Markdown code fence (with an optional `json` tag) that
/// models often emit around JSON output, returning the inner text.
fn strip_json_fences(s: &str) -> &str {
    let s = s.trim();
    let inner = s
        .strip_prefix("```json")
        .or_else(|| s.strip_prefix("```JSON"))
        .or_else(|| s.strip_prefix("```"))
        .and_then(|t| t.trim_start_matches('\n').strip_suffix("```"))
        .map(|t| t.trim_end_matches('\n').trim());
    inner.unwrap_or(s)
}

/// Generates a unique id for a synthesized tool call: nine alphanumeric
/// characters, the one shape every template accepts, since Mistral's refuse
/// any other length when the call comes back in the next turn.
fn make_tool_call_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("call{:05x}", seq % 0x10_0000)
}

/// The id as a template that insists on nine alphanumeric characters sees
/// it: kept when it already is one, otherwise a stable nine-character hash,
/// so a call and its result keep pointing at each other.
fn nine_char_id(id: &str) -> String {
    if id.len() == 9 && id.bytes().all(|b| b.is_ascii_alphanumeric()) {
        return id.to_string();
    }
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    id.hash(&mut hasher);
    let mut n = hasher.finish();
    let digits = b"0123456789abcdefghijklmnopqrstuvwxyz";
    (0..9)
        .map(|_| {
            let d = digits[(n % 36) as usize] as char;
            n /= 36;
            d
        })
        .collect()
}

/// Messages whose tool-call ids satisfy a template that checks their length,
/// as Mistral's do with `tool_call.id|length != 9`; other templates get the
/// messages as they came.
fn messages_for_template<'a>(
    template: &str,
    messages: &'a [ChatMessage],
) -> std::borrow::Cow<'a, [ChatMessage]> {
    if !template.contains("|length != 9") {
        return std::borrow::Cow::Borrowed(messages);
    }
    let mut out = messages.to_vec();
    for m in &mut out {
        if let Some(calls) = &mut m.tool_calls {
            for c in calls {
                c.id = nine_char_id(&c.id);
            }
        }
        if let Some(id) = &m.tool_call_id {
            m.tool_call_id = Some(nine_char_id(id));
        }
    }
    std::borrow::Cow::Owned(out)
}

/// Checks the OpenAI function-name rule: 1 to 64 characters, each ASCII
/// alphanumeric, `_`, or `-`.
fn is_valid_function_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
}

/// Resolves a local `$ref` (`#` or a `#/...` pointer) against the root
/// schema; `None` when there is no `$ref` or it points outside the document.
fn resolve_schema_ref<'a>(
    schema: &'a serde_json::Value,
    root: &'a serde_json::Value,
) -> Option<&'a serde_json::Value> {
    let ref_str = schema.get("$ref")?.as_str()?;
    if ref_str == "#" {
        Some(root)
    } else if let Some(pointer) = ref_str.strip_prefix('#') {
        root.pointer(pointer)
    } else {
        None
    }
}

/// Whether the schema describes an object, by `type` or implicitly through
/// object keywords (`properties`, `required`, `additionalProperties`).
fn is_object_schema(schema_obj: &serde_json::Map<String, serde_json::Value>) -> bool {
    match schema_obj.get("type") {
        Some(serde_json::Value::String(t)) => t == "object",
        Some(serde_json::Value::Array(arr)) => arr.iter().any(|v| v.as_str() == Some("object")),
        _ => {
            schema_obj.contains_key("properties")
                || schema_obj.contains_key("required")
                || schema_obj.contains_key("additionalProperties")
        }
    }
}

/// Whether the schema describes an array, by `type` or an `items` keyword.
fn is_array_schema(schema_obj: &serde_json::Map<String, serde_json::Value>) -> bool {
    match schema_obj.get("type") {
        Some(serde_json::Value::String(t)) => t == "array",
        Some(serde_json::Value::Array(arr)) => arr.iter().any(|v| v.as_str() == Some("array")),
        _ => schema_obj.contains_key("items"),
    }
}

/// Validates a `type` keyword: a known type name or a non-empty array of
/// them.
///
/// ## Errors
///
/// Returns a message naming `path` when the keyword is malformed or contains
/// an unsupported type name.
fn validate_schema_type_keyword(ty: &serde_json::Value, path: &str) -> Result<(), String> {
    let valid_type = |name: &str| {
        matches!(
            name,
            "object" | "array" | "string" | "number" | "integer" | "boolean" | "null"
        )
    };
    match ty {
        serde_json::Value::String(name) if valid_type(name) => Ok(()),
        serde_json::Value::Array(arr) if !arr.is_empty() => {
            for item in arr {
                let Some(name) = item.as_str() else {
                    return Err(format!("{path}.type entries must be strings"));
                };
                if !valid_type(name) {
                    return Err(format!("{path}.type contains unsupported type '{name}'"));
                }
            }
            Ok(())
        }
        serde_json::Value::String(name) => {
            Err(format!("{path}.type contains unsupported type '{name}'"))
        }
        _ => Err(format!(
            "{path}.type must be a string or non-empty array of strings"
        )),
    }
}

/// Recursive worker for [`validate_schema_shape`]; `path` locates errors and
/// `depth` caps `$ref` and nesting recursion at 64 levels.
///
/// ## Errors
///
/// Returns the first structural violation found, prefixed with the JSON path
/// of the offending keyword.
fn validate_schema_shape_inner(
    schema: &serde_json::Value,
    root: &serde_json::Value,
    path: &str,
    strictness: SchemaStrictness,
    depth: usize,
) -> Result<(), String> {
    if depth > 64 {
        return Err("schema is too deeply nested".to_string());
    }

    if let Some(resolved) = resolve_schema_ref(schema, root) {
        return validate_schema_shape_inner(resolved, root, path, strictness, depth + 1);
    }

    let Some(obj) = schema.as_object() else {
        return Err(format!("{path} must be a JSON object"));
    };

    if let Some(ty) = obj.get("type") {
        validate_schema_type_keyword(ty, path)?;
    }

    if let Some(required) = obj.get("required") {
        let Some(arr) = required.as_array() else {
            return Err(format!("{path}.required must be an array"));
        };
        for item in arr {
            if item.as_str().is_none() {
                return Err(format!("{path}.required entries must be strings"));
            }
        }
    }

    if let Some(enum_vals) = obj.get("enum") {
        let Some(arr) = enum_vals.as_array() else {
            return Err(format!("{path}.enum must be an array"));
        };
        if arr.is_empty() {
            return Err(format!("{path}.enum must not be empty"));
        }
    }

    if let Some(any_of) = obj.get("anyOf") {
        let Some(arr) = any_of.as_array() else {
            return Err(format!("{path}.anyOf must be an array"));
        };
        if arr.is_empty() {
            return Err(format!("{path}.anyOf must not be empty"));
        }
        for (idx, branch) in arr.iter().enumerate() {
            validate_schema_shape_inner(
                branch,
                root,
                &format!("{path}.anyOf[{idx}]"),
                strictness,
                depth + 1,
            )?;
        }
    }

    if let Some(props) = obj.get("properties") {
        let Some(props_obj) = props.as_object() else {
            return Err(format!("{path}.properties must be an object"));
        };
        for (key, prop_schema) in props_obj {
            validate_schema_shape_inner(
                prop_schema,
                root,
                &format!("{path}.properties.{key}"),
                strictness,
                depth + 1,
            )?;
        }
    }

    if let Some(items) = obj.get("items") {
        validate_schema_shape_inner(items, root, &format!("{path}.items"), strictness, depth + 1)?;
    }

    if let Some(additional) = obj.get("additionalProperties") {
        match additional {
            serde_json::Value::Bool(_) => {}
            serde_json::Value::Object(_) => validate_schema_shape_inner(
                additional,
                root,
                &format!("{path}.additionalProperties"),
                strictness,
                depth + 1,
            )?,
            _ => {
                return Err(format!(
                    "{path}.additionalProperties must be a boolean or schema object"
                ));
            }
        }
    }

    if let Some(defs) = obj.get("$defs") {
        let Some(defs_obj) = defs.as_object() else {
            return Err(format!("{path}.$defs must be an object"));
        };
        for (key, def_schema) in defs_obj {
            validate_schema_shape_inner(
                def_schema,
                root,
                &format!("{path}.$defs.{key}"),
                strictness,
                depth + 1,
            )?;
        }
    }

    if strictness == SchemaStrictness::Strict && is_object_schema(obj) {
        match obj.get("additionalProperties") {
            Some(serde_json::Value::Bool(false)) => {}
            _ => {
                return Err(format!(
                    "{path} must set additionalProperties to false in strict mode"
                ));
            }
        }
        let props = obj
            .get("properties")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();
        let required = obj
            .get("required")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("{path}.required is required in strict mode"))?;
        let required_names: HashSet<&str> = required.iter().filter_map(|v| v.as_str()).collect();
        for key in props.keys() {
            if !required_names.contains(key.as_str()) {
                return Err(format!(
                    "{path}.required must include property '{key}' in strict mode"
                ));
            }
        }
    }

    Ok(())
}

/// Validates that a request-supplied JSON Schema uses only the supported
/// subset, applying strict-mode rules when requested.
///
/// ## Errors
///
/// Returns a human-readable message with the JSON path of the first invalid
/// keyword: an unsupported `type`, malformed `required`/`enum`/`anyOf`/
/// `properties`/`items`/`additionalProperties`/`$defs`, nesting deeper than
/// 64 levels, or a strict-mode violation (an object without
/// `additionalProperties: false`, or `required` not covering every
/// property).
fn validate_schema_shape(
    schema: &serde_json::Value,
    strictness: SchemaStrictness,
    path: &str,
) -> Result<(), String> {
    validate_schema_shape_inner(schema, schema, path, strictness, 0)
}

/// Whether `value` matches a JSON Schema `type` name; unknown names match
/// anything.
fn matches_json_type(value: &serde_json::Value, ty: &str) -> bool {
    match ty {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        _ => true,
    }
}

/// Recursive worker for [`validate_against_schema`]; fails closed on
/// non-object schemas and on nesting deeper than 64 levels.
fn validate_against_schema_inner(
    value: &serde_json::Value,
    schema: &serde_json::Value,
    root: &serde_json::Value,
    depth: usize,
) -> bool {
    if depth > 64 {
        return false;
    }

    let schema = resolve_schema_ref(schema, root).unwrap_or(schema);
    let Some(obj) = schema.as_object() else {
        return false;
    };

    if let Some(any_of) = obj.get("anyOf").and_then(|v| v.as_array())
        && !any_of
            .iter()
            .any(|branch| validate_against_schema_inner(value, branch, root, depth + 1))
    {
        return false;
    }

    if let Some(const_val) = obj.get("const")
        && const_val != value
    {
        return false;
    }

    if let Some(enum_vals) = obj.get("enum").and_then(|e| e.as_array())
        && !enum_vals.contains(value)
    {
        return false;
    }

    if let Some(ty) = obj.get("type") {
        let type_ok = match ty {
            serde_json::Value::String(name) => matches_json_type(value, name),
            serde_json::Value::Array(arr) => arr
                .iter()
                .filter_map(|v| v.as_str())
                .any(|name| matches_json_type(value, name)),
            _ => false,
        };
        if !type_ok {
            return false;
        }
    }

    if let Some(val_obj) = value.as_object() {
        if let Some(required) = obj.get("required").and_then(|r| r.as_array()) {
            for req in required {
                if let Some(field) = req.as_str()
                    && !val_obj.contains_key(field)
                {
                    return false;
                }
            }
        }

        let props = obj.get("properties").and_then(|p| p.as_object());
        match obj.get("additionalProperties") {
            Some(serde_json::Value::Bool(false)) => {
                if let Some(props) = props {
                    for key in val_obj.keys() {
                        if !props.contains_key(key) {
                            return false;
                        }
                    }
                }
            }
            Some(serde_json::Value::Object(_)) => {
                if let Some(extra_schema) = obj.get("additionalProperties") {
                    for (key, prop_val) in val_obj {
                        let is_declared = props.is_some_and(|p| p.contains_key(key));
                        if !is_declared
                            && !validate_against_schema_inner(
                                prop_val,
                                extra_schema,
                                root,
                                depth + 1,
                            )
                        {
                            return false;
                        }
                    }
                }
            }
            _ => {}
        }

        if let Some(props) = props {
            for (key, prop_schema) in props {
                if let Some(prop_val) = val_obj.get(key)
                    && !validate_against_schema_inner(prop_val, prop_schema, root, depth + 1)
                {
                    return false;
                }
            }
        }
    }

    if is_array_schema(obj)
        && let (Some(items_schema), Some(arr)) = (obj.get("items"), value.as_array())
    {
        for item in arr {
            if !validate_against_schema_inner(item, items_schema, root, depth + 1) {
                return false;
            }
        }
    }

    true
}

/// The obligation to call in `required` and forced modes, which is all a
/// template that renders the definitions itself still needs to hear; `auto`
/// needs nothing.
fn tool_choice_instruction(config: &ToolConfig) -> Option<String> {
    match &config.choice_mode {
        ToolChoiceMode::Required => {
            Some("You must call one or more tools. Do not respond with plain text.".to_string())
        }
        ToolChoiceMode::ForcedFunction { name } => Some(format!(
            "You must call exactly one tool named \"{name}\". Do not respond with plain text."
        )),
        ToolChoiceMode::None | ToolChoiceMode::Auto => None,
    }
}

/// Builds the system-prompt block that teaches a model without native tool
/// support the JSON calling convention, including the choice-mode and
/// parallel-call instructions.
fn tools_system_instruction(tools: &[ToolDefinition], config: &ToolConfig) -> String {
    let tools_json = serde_json::to_string_pretty(tools).unwrap_or_default();
    let choice_instruction = tool_choice_instruction(config).unwrap_or_else(|| {
        match config.choice_mode {
            ToolChoiceMode::None => "Do not call any tool. Respond normally.",
            _ => "If no tool is needed, respond normally. Otherwise, emit tool calls only.",
        }
        .to_string()
    });
    let parallel_instruction = if config.parallel_tool_calls {
        "You may call multiple tools by adding more objects to the array."
    } else {
        "Call at most one tool."
    };
    format!(
        "You have access to the following tools:\n\
         <tools>\n{tools_json}\n</tools>\n\n\
         If you need to call a tool, respond ONLY with a valid JSON object \
         (no other text, no markdown) in this exact format:\n\
         {{\"tool_calls\": [{{\"name\": \"function_name\", \"arguments\": {{\"param\": \"value\"}}}}]}}\n\n\
         {parallel_instruction}\n\
         {choice_instruction}"
    )
}

/// Parses one tool-call object in either the OpenAI nested shape
/// (`function.name`/`function.arguments`) or the flat `name`/`arguments`
/// shape, generating an id when absent. Llama 3's `parameters` stands for
/// `arguments`, and a `function` that is the bare name, as Llama 3.2 1B
/// writes it, is the name.
fn parse_tool_call_entry(call: &serde_json::Value) -> Option<ToolCall> {
    let call_obj = call.as_object()?;
    let name = call_obj
        .get("function")
        .and_then(|f| {
            f.get("name")
                .and_then(|v| v.as_str())
                .or_else(|| f.as_str())
        })
        .or_else(|| call_obj.get("name").and_then(|v| v.as_str()))?
        .to_string();
    let args = call_obj
        .get("function")
        .and_then(|f| f.get("arguments").or_else(|| f.get("parameters")))
        .or_else(|| call_obj.get("arguments"))
        .or_else(|| call_obj.get("parameters"))?;
    let arguments = match args {
        serde_json::Value::String(s) => s.clone(),
        other => serde_json::to_string(other).ok()?,
    };
    Some(ToolCall {
        id: call_obj
            .get("id")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .unwrap_or_else(make_tool_call_id),
        call_type: call_obj
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("function")
            .to_string(),
        function: ToolCallFunction { name, arguments },
    })
}

fn tool_exists(tools: &[ToolDefinition], name: &str) -> bool {
    tools.iter().any(|tool| tool.function.name == name)
}

/// The JSON inside one `<tool_call>` block. Qwen2.5's GGUF template shows
/// the example call with doubled braces, `{{"name": ...}}`, and the models
/// copy them, closing with two braces or three, so an object wrapped in an
/// extra opening brace is read from its own brace to its own end.
fn parse_json_block(block: &str) -> Option<serde_json::Value> {
    let text = strip_json_fences(block.trim());
    serde_json::from_str(text).ok().or_else(|| {
        let inner = text.strip_prefix('{')?.trim_start();
        if !inner.starts_with('{') {
            return None;
        }
        serde_json::Deserializer::from_str(inner)
            .into_iter::<serde_json::Value>()
            .next()?
            .ok()
    })
}

/// Parses gemma-4's call, `<|tool_call>call:NAME{key:value,...}<tool_call|>`,
/// whose string values and nested keys sit between `<|"|>` quote tokens. The
/// GGUF files keep the three markers in the text; the safetensors tokenizer
/// drops them as special, leaving `call:NAME{city:Rome}` with the quotes
/// gone, so a value that is not a JSON literal is read as a string and a
/// comma splits arguments only where a declared parameter name follows.
fn parse_gemma_calls(raw: &str, tools: &[ToolDefinition]) -> Option<Vec<ToolCall>> {
    let text = raw
        .replace("<|\"|>", "\"")
        .replace("<|tool_call>", "")
        .replace("<tool_call|>", "");
    let mut calls = Vec::new();
    let mut rest = text.as_str();
    while let Some(start) = rest.find("call:") {
        let after = &rest[start + "call:".len()..];
        let Some(open) = after.find('{') else { break };
        let name = after[..open].trim();
        let Some(close) = matching_brace(&after[open..]).map(|i| open + i) else {
            break;
        };
        if let Some(tool) = tools.iter().find(|t| t.function.name == name) {
            let names: Vec<&str> = tool
                .function
                .parameters
                .as_ref()
                .and_then(|p| p.get("properties"))
                .and_then(|p| p.as_object())
                .map(|p| p.keys().map(String::as_str).collect())
                .unwrap_or_default();
            calls.push(ToolCall {
                id: make_tool_call_id(),
                call_type: "function".to_string(),
                function: ToolCallFunction {
                    name: name.to_string(),
                    arguments: gemma_arguments(&after[open + 1..close], &names).to_string(),
                },
            });
        }
        rest = &after[close + 1..];
    }
    if calls.is_empty() { None } else { Some(calls) }
}

/// The index of the brace closing the one at `text[0]`, skipping quoted
/// strings.
fn matching_brace(text: &str) -> Option<usize> {
    let (mut depth, mut quoted, mut escaped) = (0usize, false, false);
    for (i, b) in text.bytes().enumerate() {
        match b {
            _ if escaped => escaped = false,
            b'\\' if quoted => escaped = true,
            b'"' => quoted = !quoted,
            b'{' | b'[' if !quoted => depth += 1,
            b'}' | b']' if !quoted => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Reads gemma-4's `key:value,key:value` argument list into a JSON object.
fn gemma_arguments(text: &str, names: &[&str]) -> serde_json::Value {
    let mut args = serde_json::Map::new();
    let mut rest = text.trim();
    while let Some(colon) = rest.find(':') {
        let key = rest[..colon].trim().trim_matches('"').to_string();
        let value_text = rest[colon + 1..].trim_start();
        let end = gemma_value_end(value_text, names);
        let value = &value_text[..end];
        let parsed = serde_json::from_str(value.trim())
            .unwrap_or_else(|_| serde_json::Value::String(value.trim().to_string()));
        args.insert(key, parsed);
        rest = value_text[end..].trim_start().trim_start_matches(',');
    }
    serde_json::Value::Object(args)
}

/// Where the value starting at `text[0]` ends: a quoted string at its closing
/// quote, a container at its closing bracket, anything else at the comma
/// that a declared parameter name and a colon follow, or at the end.
fn gemma_value_end(text: &str, names: &[&str]) -> usize {
    let bytes = text.as_bytes();
    match bytes.first() {
        Some(b'"') => {
            let mut escaped = false;
            for (i, &b) in bytes.iter().enumerate().skip(1) {
                match b {
                    _ if escaped => escaped = false,
                    b'\\' => escaped = true,
                    b'"' => return i + 1,
                    _ => {}
                }
            }
            text.len()
        }
        Some(b'{' | b'[') => matching_brace(text).map(|i| i + 1).unwrap_or(text.len()),
        _ => {
            let mut from = 0;
            while let Some(comma) = text[from..].find(',').map(|i| from + i) {
                let next = text[comma + 1..].trim_start();
                let starts_argument = names.iter().any(|n| {
                    next.strip_prefix(n)
                        .is_some_and(|after| after.trim_start().starts_with(':'))
                });
                if starts_argument {
                    return comma;
                }
                from = comma + 1;
            }
            text.len()
        }
    }
}

/// Parses the Mistral call `[TOOL_CALLS]name[ARGS]{...}` as it reaches the
/// text once its two special tokens are dropped: a known tool name followed
/// by a JSON object, repeated for parallel calls.
fn parse_name_brace_calls(raw: &str, tools: &[ToolDefinition]) -> Option<Vec<ToolCall>> {
    let mut calls = Vec::new();
    let mut rest = raw.trim();
    while !rest.is_empty() {
        let name = &tools
            .iter()
            .find(|t| {
                rest.strip_prefix(t.function.name.as_str())
                    .is_some_and(|after| after.trim_start().starts_with('{'))
            })?
            .function
            .name;
        let after = rest[name.len()..].trim_start();
        let mut stream = serde_json::Deserializer::from_str(after).into_iter::<serde_json::Value>();
        let args = stream.next()?.ok()?;
        if !args.is_object() {
            return None;
        }
        calls.push(ToolCall {
            id: make_tool_call_id(),
            call_type: "function".to_string(),
            function: ToolCallFunction {
                name: name.clone(),
                arguments: args.to_string(),
            },
        });
        rest = after[stream.byte_offset()..].trim_start();
    }
    if calls.is_empty() { None } else { Some(calls) }
}

/// Parse the tool calls wrapped in `<tool_call>…</tool_call>` blocks,
/// possibly preceded by free-form reasoning text and possibly repeated for
/// parallel calls. A last block the model left open, stopping right after
/// its payload instead of closing the tag, runs to the end of the text, as
/// the reference parsers read it. Two block payloads exist in the wild:
///
/// - Qwen3.5-class XML: `<function=NAME>\n<parameter=KEY>\nVALUE\n
///   </parameter>…</function>`. Parameter values are typed best-effort:
///   anything that parses as JSON (numbers, booleans, null, objects, arrays,
///   quoted strings) is kept as that value, bare text stays a string.
/// - Qwen2.5/Qwen3-class JSON: a single `{"name": …, "arguments": …}` object
///   (canonical template output), or a `{"tool_calls": […]}` object when the
///   model echoes the injected system-prompt instruction inside the tags.
fn parse_tool_call_tag_blocks(raw: &str) -> Option<Vec<ToolCall>> {
    let mut calls = Vec::new();
    let mut rest = raw;
    while let Some(start) = rest.find("<tool_call>") {
        let after = &rest[start + "<tool_call>".len()..];
        let (block, next) = match after.find("</tool_call>") {
            Some(end) => (&after[..end], &after[end + "</tool_call>".len()..]),
            None => (after, ""),
        };
        rest = next;

        let Some(fstart) = block.find("<function=") else {
            if let Some(value) = parse_json_block(block) {
                if let Some(nested) = value.get("tool_calls").and_then(|v| v.as_array()) {
                    calls.extend(nested.iter().filter_map(parse_tool_call_entry));
                } else if let Some(single) = parse_tool_call_entry(&value) {
                    calls.push(single);
                }
            }
            continue;
        };
        let fname_rest = &block[fstart + "<function=".len()..];
        let Some(fname_end) = fname_rest.find('>') else {
            continue;
        };
        let name = fname_rest[..fname_end].trim().to_string();
        let body = &fname_rest[fname_end + 1..];
        let body = body.split("</function>").next().unwrap_or(body);

        let mut args = serde_json::Map::new();
        let mut prest = body;
        while let Some(ps) = prest.find("<parameter=") {
            let pr = &prest[ps + "<parameter=".len()..];
            let Some(pe) = pr.find('>') else { break };
            let key = pr[..pe].trim().to_string();
            let vrest = &pr[pe + 1..];
            let Some(vend) = vrest.find("</parameter>") else {
                break;
            };
            // The template frames the value with one newline on each side;
            // strip exactly that framing, preserving interior whitespace.
            let val_raw = &vrest[..vend];
            let val_raw = val_raw.strip_prefix('\n').unwrap_or(val_raw);
            let val_raw = val_raw.strip_suffix('\n').unwrap_or(val_raw);
            let value = serde_json::from_str::<serde_json::Value>(val_raw.trim())
                .unwrap_or_else(|_| serde_json::Value::String(val_raw.to_string()));
            args.insert(key, value);
            prest = &vrest[vend + "</parameter>".len()..];
        }

        calls.push(ToolCall {
            id: make_tool_call_id(),
            call_type: "function".to_string(),
            function: ToolCallFunction {
                name,
                arguments: serde_json::Value::Object(args).to_string(),
            },
        });
    }
    if calls.is_empty() { None } else { Some(calls) }
}

/// Extracts tool calls from raw model output, or `None` when it is a plain
/// text response.
///
/// Tries `<tool_call>` tag blocks first ([`parse_tool_call_tag_blocks`]),
/// then the whole output as bare JSON: a `{"tool_calls": [...]}` object or a
/// single call object. Under `ForcedFunction`, and under `Required` with
/// exactly one tool, a bare JSON object is wrapped as that tool's arguments.
/// Calls naming unknown tools are dropped, serial mode truncates to one
/// call, and a forced function keeps only matching calls.
fn try_parse_tool_calls(raw: &str, config: &ToolConfig) -> Option<Vec<ToolCall>> {
    if let Some(mut result) = parse_tool_call_tag_blocks(raw)
        .or_else(|| parse_name_brace_calls(raw, &config.tools))
        .or_else(|| parse_gemma_calls(raw, &config.tools))
    {
        result.retain(|call| tool_exists(&config.tools, &call.function.name));
        if !result.is_empty() {
            if !config.parallel_tool_calls && result.len() > 1 {
                result.truncate(1);
            }
            if let ToolChoiceMode::ForcedFunction { name } = &config.choice_mode {
                result.retain(|call| call.function.name == *name);
                result.truncate(1);
            }
            if !result.is_empty() {
                return Some(result);
            }
        }
    }

    let stripped = strip_json_fences(raw.trim());
    let value: serde_json::Value = serde_json::from_str(stripped).ok()?;
    let mut result: Vec<ToolCall> = if let Some(calls) = value
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .or_else(|| value.as_array())
    {
        let mut parsed = Vec::with_capacity(calls.len());
        for call in calls {
            let parsed_call = parse_tool_call_entry(call)?;
            parsed.push(parsed_call);
        }
        parsed
    } else if let Some(single) = parse_tool_call_entry(&value) {
        vec![single]
    } else if let ToolChoiceMode::ForcedFunction { name } = &config.choice_mode {
        vec![ToolCall {
            id: make_tool_call_id(),
            call_type: "function".to_string(),
            function: ToolCallFunction {
                name: name.clone(),
                arguments: stripped.to_string(),
            },
        }]
    } else if matches!(config.choice_mode, ToolChoiceMode::Required) && config.tools.len() == 1 {
        vec![ToolCall {
            id: make_tool_call_id(),
            call_type: "function".to_string(),
            function: ToolCallFunction {
                name: config.tools[0].function.name.clone(),
                arguments: stripped.to_string(),
            },
        }]
    } else {
        return None;
    };

    result.retain(|call| {
        call.call_type == "function" && tool_exists(&config.tools, &call.function.name)
    });
    if result.is_empty() {
        return None;
    }

    if !config.parallel_tool_calls && result.len() > 1 {
        result.truncate(1);
    }

    match &config.choice_mode {
        ToolChoiceMode::ForcedFunction { name } => {
            result.retain(|call| call.function.name == *name);
            if result.len() > 1 {
                result.truncate(1);
            }
        }
        ToolChoiceMode::Required => {}
        ToolChoiceMode::Auto | ToolChoiceMode::None => {}
    }

    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}

/// Validates parsed model output against the supported JSON Schema subset:
/// `type`, `required`, `properties`, `items`, `enum`, `const`, `anyOf`,
/// `additionalProperties`, and local `$ref`.
fn validate_against_schema(value: &serde_json::Value, schema: &serde_json::Value) -> bool {
    validate_against_schema_inner(value, schema, schema, 0)
}

/// Validates one entry of the request `tools` array.
///
/// ## Errors
///
/// Returns a message naming `tools[index]` when the type is not `function`,
/// the name violates the OpenAI pattern, or the parameter schema fails
/// [`validate_schema_shape`] (strict rules apply when `strict` is set; a
/// strict tool without parameters is checked as an empty closed object).
fn validate_tool_definition(tool: &ToolDefinition, index: usize) -> Result<(), String> {
    if tool.tool_type != "function" {
        return Err(format!(
            "tools[{index}].type must be 'function'; got '{}'",
            tool.tool_type
        ));
    }
    if !is_valid_function_name(&tool.function.name) {
        return Err(format!(
            "tools[{index}].function.name must match ^[A-Za-z0-9_-]{{1,64}}$"
        ));
    }
    if let Some(parameters) = &tool.function.parameters {
        let strictness = if tool.function.strict.unwrap_or(false) {
            SchemaStrictness::Strict
        } else {
            SchemaStrictness::BestEffort
        };
        validate_schema_shape(
            parameters,
            strictness,
            &format!("tools[{index}].function.parameters"),
        )?;
    } else if tool.function.strict.unwrap_or(false) {
        let empty = serde_json::json!({
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": false
        });
        validate_schema_shape(
            &empty,
            SchemaStrictness::Strict,
            &format!("tools[{index}].function.parameters"),
        )?;
    }
    Ok(())
}

/// Extracts the function name from a tool reference in either the nested
/// (`function.name`) or flat (`name`) shape.
fn parse_tool_reference_name(tool_ref: &serde_json::Value) -> Option<String> {
    tool_ref
        .get("function")
        .and_then(|v| v.get("name"))
        .and_then(|v| v.as_str())
        .or_else(|| tool_ref.get("name").and_then(|v| v.as_str()))
        .map(str::to_string)
}

/// Validates `tools`, `tool_choice`, and `parallel_tool_calls` into a
/// [`ToolConfig`], filtering the tool list down to what the choice allows.
///
/// ## Errors
///
/// Returns an `invalid_request_error` message when a tool definition is
/// malformed, `tool_choice` has an unknown mode or shape, it references an
/// unknown function, or it leaves no usable tools.
fn build_tool_config(
    tools: Option<&[ToolDefinition]>,
    tool_choice: Option<&serde_json::Value>,
    parallel_tool_calls: Option<bool>,
) -> Result<ToolConfig, String> {
    let all_tools = tools.unwrap_or(&[]);
    for (idx, tool) in all_tools.iter().enumerate() {
        validate_tool_definition(tool, idx)?;
    }

    let default_choice = if all_tools.is_empty() {
        ToolChoiceMode::None
    } else {
        ToolChoiceMode::Auto
    };

    let mut config = ToolConfig {
        tools: all_tools.to_vec(),
        choice_mode: default_choice,
        parallel_tool_calls: parallel_tool_calls.unwrap_or(true),
    };

    let Some(raw_choice) = tool_choice else {
        return Ok(config);
    };

    match raw_choice {
        serde_json::Value::String(mode) => {
            config.choice_mode = match mode.as_str() {
                "none" => ToolChoiceMode::None,
                "auto" => ToolChoiceMode::Auto,
                "required" => ToolChoiceMode::Required,
                other => {
                    return Err(format!(
                        "tool_choice must be one of 'none', 'auto', or 'required'; got '{other}'"
                    ));
                }
            };
            if matches!(config.choice_mode, ToolChoiceMode::None) {
                config.tools.clear();
            }
        }
        serde_json::Value::Object(obj) => {
            let Some(choice_type) = obj.get("type").and_then(|v| v.as_str()) else {
                return Err(
                    "tool_choice.type is required when tool_choice is an object".to_string()
                );
            };
            match choice_type {
                "function" => {
                    let name = parse_tool_reference_name(raw_choice).ok_or_else(|| {
                        "tool_choice.function.name is required when forcing a function".to_string()
                    })?;
                    let forced_tool = all_tools
                        .iter()
                        .find(|tool| tool.function.name == name)
                        .cloned()
                        .ok_or_else(|| {
                            format!("tool_choice references unknown function '{name}'")
                        })?;
                    config.tools = vec![forced_tool];
                    config.choice_mode = ToolChoiceMode::ForcedFunction { name };
                    config.parallel_tool_calls = false;
                }
                "allowed_tools" => {
                    let allowed_cfg = obj
                        .get("allowed_tools")
                        .unwrap_or(raw_choice)
                        .as_object()
                        .ok_or_else(|| "tool_choice.allowed_tools must be an object".to_string())?;
                    let mode = allowed_cfg
                        .get("mode")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| "tool_choice.allowed_tools.mode is required".to_string())?;
                    let allowed_refs = allowed_cfg
                        .get("tools")
                        .and_then(|v| v.as_array())
                        .ok_or_else(|| {
                            "tool_choice.allowed_tools.tools must be an array".to_string()
                        })?;
                    let mut filtered = Vec::new();
                    let mut seen = HashSet::new();
                    for tool_ref in allowed_refs {
                        let name = parse_tool_reference_name(tool_ref).ok_or_else(|| {
                            "tool_choice.allowed_tools.tools[] entries must include function.name"
                                .to_string()
                        })?;
                        let Some(tool) = all_tools.iter().find(|tool| tool.function.name == name)
                        else {
                            return Err(format!(
                                "tool_choice.allowed_tools references unknown function '{name}'"
                            ));
                        };
                        if seen.insert(name.clone()) {
                            filtered.push(tool.clone());
                        }
                    }
                    config.tools = filtered;
                    config.choice_mode = match mode {
                        "auto" => ToolChoiceMode::Auto,
                        "required" => ToolChoiceMode::Required,
                        other => {
                            return Err(format!(
                                "tool_choice.allowed_tools.mode must be 'auto' or 'required'; got '{other}'"
                            ));
                        }
                    };
                }
                other => {
                    return Err(format!(
                        "tool_choice.type must be 'function' or 'allowed_tools'; got '{other}'"
                    ));
                }
            }
        }
        _ => return Err("tool_choice must be a string or object".to_string()),
    }

    if !matches!(config.choice_mode, ToolChoiceMode::None) && config.tools.is_empty() {
        return Err("tool_choice requires at least one function tool".to_string());
    }

    Ok(config)
}

/// Builds the system-prompt instruction for `response_format`: JSON-only
/// output, plus the schema, name, and description when `json_schema` is
/// requested.
fn json_system_instruction(rf: &ResponseFormat) -> String {
    match rf.format_type.as_str() {
        "json_schema" => {
            if let Some(spec) = &rf.json_schema {
                let schema_part = spec
                    .schema
                    .as_ref()
                    .and_then(|s| serde_json::to_string_pretty(s).ok())
                    .filter(|s| !s.is_empty())
                    .map(|s| {
                        format!(
                            " that conforms to the following JSON Schema:\n```json\n{}\n```",
                            s
                        )
                    })
                    .unwrap_or_default();
                let name_part = if spec.name.trim().is_empty() {
                    String::new()
                } else {
                    format!(" The schema name is \"{}\".", spec.name.trim())
                };
                let description_part = spec
                    .description
                    .as_ref()
                    .map(|d| d.trim())
                    .filter(|d| !d.is_empty())
                    .map(|d| format!(" Schema description: {}.", d))
                    .unwrap_or_default();
                let strict_part = if spec.strict.unwrap_or(false) {
                    " Follow the schema exactly and do not add unspecified fields.".to_string()
                } else {
                    String::new()
                };
                format!(
                    "You must respond with valid JSON only. Do not include any explanation, markdown, or text outside of the JSON object.{}{}{}{}",
                    schema_part, name_part, description_part, strict_part,
                )
            } else {
                "You must respond with valid JSON only. Do not include any explanation, markdown, or text outside of the JSON object.".to_string()
            }
        }
        _ => {
            "You must respond with valid JSON only. Do not include any explanation, markdown, or text outside of the JSON object.".to_string()
        }
    }
}

/// Validates sampling parameters against the OpenAI Chat Completions ranges.
///
/// ## Errors
///
/// Returns an `invalid_request_error` message when a value is out of range,
/// non-finite, or zero where at least 1 is required.
fn validate_sampling_params(body: &ChatCompletionRequest) -> Result<(), String> {
    fn check_range_inclusive(
        value: Option<f32>,
        lo: f32,
        hi: f32,
        name: &str,
    ) -> Result<(), String> {
        if let Some(v) = value
            && (!v.is_finite() || v < lo || v > hi)
        {
            return Err(format!(
                "{name} must be between {lo} and {hi} (inclusive); got {v}"
            ));
        }
        Ok(())
    }

    check_range_inclusive(body.temperature, 0.0, 2.0, "temperature")?;
    check_range_inclusive(body.top_p, 0.0, 1.0, "top_p")?;
    check_range_inclusive(body.min_p, 0.0, 1.0, "min_p")?;
    check_range_inclusive(body.frequency_penalty, -2.0, 2.0, "frequency_penalty")?;
    check_range_inclusive(body.presence_penalty, -2.0, 2.0, "presence_penalty")?;

    if let Some(r) = body.repetition_penalty
        && (!r.is_finite() || r <= 0.0)
    {
        return Err(format!(
            "repetition_penalty must be greater than 0; got {r}"
        ));
    }

    if let Some(k) = body.top_logprobs
        && k > 20
    {
        return Err(format!(
            "top_logprobs must be between 0 and 20 (inclusive); got {k}"
        ));
    }

    if let Some(m) = body.max_tokens
        && m == 0
    {
        return Err("max_tokens must be at least 1".to_string());
    }
    if let Some(m) = body.max_completion_tokens
        && m == 0
    {
        return Err("max_completion_tokens must be at least 1".to_string());
    }

    Ok(())
}

/// Validates the `response_format` parameter.
///
/// ## Errors
///
/// Returns a message when the type is neither `json_object` nor
/// `json_schema`, the schema payload is missing, or the schema fails
/// [`validate_schema_shape`].
fn validate_response_format_request(response_format: &ResponseFormat) -> Result<(), String> {
    match response_format.format_type.as_str() {
        "json_object" => Ok(()),
        "json_schema" => {
            let Some(spec) = response_format.json_schema.as_ref() else {
                return Err(
                    "response_format.json_schema is required when response_format.type is 'json_schema'"
                        .to_string(),
                );
            };
            let Some(schema) = spec.schema.as_ref() else {
                return Err(
                    "response_format.json_schema.schema is required when response_format.type is 'json_schema'"
                        .to_string(),
                );
            };
            let strictness = if spec.strict.unwrap_or(false) {
                SchemaStrictness::Strict
            } else {
                SchemaStrictness::BestEffort
            };
            validate_schema_shape(schema, strictness, "response_format.json_schema.schema")
        }
        other => Err(format!(
            "response_format.type must be 'json_object' or 'json_schema'; got '{other}'"
        )),
    }
}

/// Renders the final prompt for a conversation, via the tokenizer's Jinja
/// chat template when present, else a family-specific fallback format (turn
/// tokens, Gemma turns, Mistral `[INST]`), else plain text.
///
/// When the template rejects the conversation and it contains a system
/// message, rendering is retried without it (some templates do not support
/// the system role).
///
/// ## Errors
///
/// Returns the template engine's error when rendering fails even after the
/// system-message retry.
pub fn apply_chat_template(
    tokenizer: &Tokenizer,
    messages: &[ChatMessage],
    enable_thinking: bool,
    reasoning_effort: Option<&str>,
    tools: Option<serde_json::Value>,
) -> anyhow::Result<String> {
    let Some(template) = tokenizer.chat_template() else {
        if tokenizer.special_token_id("<|turn>").is_some()
            && tokenizer.special_token_id("<turn|>").is_some()
        {
            return Ok(chat_template::format_turn_chat(
                messages,
                tokenizer.bos_token(),
                "<|turn>",
                "<turn|>",
                true,
                enable_thinking,
            ));
        }

        if tokenizer.special_token_id("<start_of_turn>").is_some()
            && tokenizer.special_token_id("<end_of_turn>").is_some()
        {
            return Ok(chat_template::format_turn_chat(
                messages,
                tokenizer.bos_token(),
                "<start_of_turn>",
                "<end_of_turn>",
                true,
                enable_thinking,
            ));
        }

        if tokenizer.special_token_id("[INST]").is_some()
            && tokenizer.special_token_id("[/INST]").is_some()
        {
            return Ok(chat_template::format_mistral_inst_chat(
                messages,
                tokenizer.bos_token(),
                tokenizer.eos_token(),
                tokenizer.special_token_id("[SYSTEM_PROMPT]").is_some(),
            ));
        }

        return Ok(chat_template::format_plain_chat(messages));
    };

    let messages = messages_for_template(template, messages);
    let messages: &[ChatMessage] = &messages;
    let try_render = |msgs: &[ChatMessage], t: Option<serde_json::Value>| {
        chat_template::apply_chat_template(
            template,
            msgs,
            chat_template::TemplateOptions {
                bos_token: tokenizer.bos_token(),
                eos_token: tokenizer.eos_token(),
                add_generation_prompt: true,
                enable_thinking,
                reasoning_effort,
                tools: t,
            },
        )
    };

    match try_render(messages, tools.clone()) {
        Ok(prompt) => Ok(prompt),
        Err(e) => {
            let without_system: Vec<&ChatMessage> =
                messages.iter().filter(|m| m.role != "system").collect();

            if without_system.len() < messages.len() {
                let msgs_ref: Vec<ChatMessage> = without_system.into_iter().cloned().collect();
                if let Ok(prompt) = try_render(&msgs_ref, tools.clone()) {
                    tracing::warn!(
                        "system role not supported by this model template; retrying without system message"
                    );
                    return Ok(prompt);
                }
            }

            if let Some(folded) = fold_tool_turns(messages) {
                let without_system: Vec<ChatMessage> = folded
                    .iter()
                    .filter(|m| m.role != "system")
                    .cloned()
                    .collect();
                for candidate in [&folded, &without_system] {
                    if let Ok(prompt) = try_render(candidate, None) {
                        tracing::warn!(
                            "tool turns not supported by this model template; retrying with them folded into user and assistant text"
                        );
                        return Ok(prompt);
                    }
                }
            }

            tracing::error!(error = ?e, "chat template rendering failed");
            Err(e)
        }
    }
}

/// The conversation as a template that knows only user and assistant turns
/// can take it: an assistant's calls become its text, a tool result becomes
/// a user turn naming the call, and consecutive turns of one role merge so
/// the roles still alternate. `None` when there is nothing to fold. The
/// Mistral 7B v0.3 GGUF ships the plain `[INST]` template and refused the
/// `tool` role of a round trip outright.
fn fold_tool_turns(messages: &[ChatMessage]) -> Option<Vec<ChatMessage>> {
    if !messages
        .iter()
        .any(|m| m.role == "tool" || m.tool_calls.is_some())
    {
        return None;
    }
    let mut out: Vec<ChatMessage> = Vec::with_capacity(messages.len());
    for m in messages {
        let (role, text) = match m.role.as_str() {
            "tool" => (
                "user",
                format!(
                    "Result of tool call {}: {}",
                    m.tool_call_id.as_deref().unwrap_or("?"),
                    m.content.as_deref().unwrap_or("")
                ),
            ),
            "assistant" if m.tool_calls.is_some() => {
                let calls = serde_json::to_string(&m.tool_calls).unwrap_or_default();
                (
                    "assistant",
                    match m.content.as_deref() {
                        Some(c) if !c.is_empty() => format!("{c}\n{calls}"),
                        _ => calls,
                    },
                )
            }
            role => (role, m.content.clone().unwrap_or_default()),
        };
        match out.last_mut() {
            Some(last) if last.role == role => {
                let joined = format!("{}\n{}", last.content.as_deref().unwrap_or(""), text);
                last.content = Some(joined);
            }
            _ => out.push(ChatMessage {
                role: role.to_string(),
                content: Some(text),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }),
        }
    }
    Some(out)
}

/// Whether the rendered prompt ends with an opened thinking block, which
/// the templates that end in `<think>` (DeepSeek-R1 distills, whatever the
/// request asked) and gemma-4's after a tool result do, so the reply starts
/// with reasoning before any marker.
fn prompt_opens_thinking(prompt: &str) -> bool {
    let prompt = prompt.trim_end();
    prompt.ends_with("<think>") || prompt.ends_with("<|channel>thought")
}

/// Whether this model's chat template renders the tool definitions itself,
/// read off the render: a template that knows `tools` produces a different
/// prompt with them than without. Such a model gets the definitions once,
/// in the format it was trained on, instead of a second copy with the JSON
/// convention meant for templates that ignore them.
fn template_renders_tools(
    tokenizer: &Tokenizer,
    messages: &[ChatMessage],
    enable_thinking: bool,
    reasoning_effort: Option<&str>,
    tools: &serde_json::Value,
) -> bool {
    let render = |t| apply_chat_template(tokenizer, messages, enable_thinking, reasoning_effort, t);
    matches!(
        (render(Some(tools.clone())), render(None)),
        (Ok(with), Ok(without)) if with != without
    )
}

/// Converts engine logprob entries into the response `logprobs` object,
/// truncating alternatives to the requested `top_logprobs` count.
fn build_logprobs_content(entries: &[EngineLogprobEntry], req_top_n: usize) -> Logprobs {
    Logprobs {
        content: entries
            .iter()
            .map(|e| TokenLogprob {
                token: e.token_str.clone(),
                logprob: e.logprob,
                bytes: Some(e.bytes.clone()),
                top_logprobs: e
                    .top_logprobs
                    .iter()
                    .take(req_top_n)
                    .map(|(ts, lp, tb)| TopLogprobItem {
                        token: ts.clone(),
                        logprob: *lp,
                        bytes: Some(tb.clone()),
                    })
                    .collect(),
            })
            .collect(),
        refusal: None,
    }
}

/// Fully collected output of one sequence: content and reasoning text,
/// token counts, finish reason, and raw logprob entries.
struct CompletionData {
    content: String,
    reasoning_content: String,
    reasoning_tokens: usize,
    finish_reason: String,
    completion_tokens: usize,
    logprob_entries: Vec<EngineLogprobEntry>,
}

/// Runs a streaming task with an optional wall-clock timeout.
///
/// On timeout the inner future is cancelled, dropping its channel ends so
/// the engine aborts the sequence, and an error chunk plus `[DONE]` are
/// emitted on the retained watchdog sender.
async fn run_streaming_with_timeout<F>(
    inner: F,
    sse_tx_watchdog: tokio_mpsc::UnboundedSender<Result<Event, std::convert::Infallible>>,
    timeout: Option<Duration>,
) where
    F: std::future::Future<Output = ()>,
{
    match timeout {
        Some(t) => {
            tokio::select! {
                _ = inner => {}
                _ = tokio::time::sleep(t) => {
                    let err = serde_json::json!({
                        "error": {
                            "message": format!("request exceeded timeout of {}s", t.as_secs()),
                            "type": "request_timeout",
                            "param": null,
                            "code": null,
                        }
                    });
                    let _ = sse_tx_watchdog.send(Ok(
                        Event::default().data(serde_json::to_string(&err).unwrap()),
                    ));
                    let _ = sse_tx_watchdog
                        .send(Ok(Event::default().data("[DONE]")));
                }
            }
        }
        None => inner.await,
    }
}

/// Collects one sequence's events into a [`CompletionData`], subject to the
/// request timeout.
///
/// ## Errors
///
/// Returns 408 on timeout expiry (dropping the receiver, which makes the
/// engine loop abort the sequence via its closed-channel check) or 500 when
/// the engine reports an error.
async fn collect_one_completion(
    rx: tokio_mpsc::UnboundedReceiver<EngineEvent>,
    timeout: Option<Duration>,
) -> Result<CompletionData, (StatusCode, Json<serde_json::Value>)> {
    let work = collect_one_completion_inner(rx);
    match timeout {
        Some(t) => match tokio::time::timeout(t, work).await {
            Ok(res) => res,
            Err(_) => Err(error_response(
                StatusCode::REQUEST_TIMEOUT,
                format!("request exceeded timeout of {}s", t.as_secs()),
                "request_timeout",
            )),
        },
        None => work.await,
    }
}

/// Accumulates engine events for one sequence until `StreamEnd`.
///
/// ## Errors
///
/// Returns a 500 response when the engine emits [`EngineEvent::Error`].
async fn collect_one_completion_inner(
    mut rx: tokio_mpsc::UnboundedReceiver<EngineEvent>,
) -> Result<CompletionData, (StatusCode, Json<serde_json::Value>)> {
    let mut data = CompletionData {
        content: String::new(),
        reasoning_content: String::new(),
        reasoning_tokens: 0,
        finish_reason: "stop".to_string(),
        completion_tokens: 0,
        logprob_entries: Vec::new(),
    };
    while let Some(event) = rx.recv().await {
        match event {
            EngineEvent::Token {
                text,
                logprob_entries,
            } => {
                data.content.push_str(&text);
                data.logprob_entries.extend(logprob_entries);
            }
            EngineEvent::ReasoningToken(text) => {
                data.reasoning_content.push_str(&text);
                data.reasoning_tokens += 1;
            }
            EngineEvent::Finish {
                finish_reason,
                completion_tokens,
            } => {
                data.finish_reason = finish_reason;
                data.completion_tokens = completion_tokens;
            }
            EngineEvent::StreamEnd => break,
            EngineEvent::Error(msg) => {
                return Err(error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    msg,
                    "server_error",
                ));
            }
        }
    }
    Ok(data)
}

/// Collects every completion of one request concurrently, in request order.
///
/// The collectors are tasks of a `JoinSet` this future owns: dropping the
/// future, which is what happens to a handler when its client goes away and
/// to the streaming task when its SSE receiver does, aborts them and drops
/// their receivers, and a closed response channel is what makes the engine
/// loop abort the sequence. Detached collector tasks kept every sequence
/// generating to `max_tokens` after a disconnect, holding its slot and KV
/// blocks for thousands of tokens nobody would read.
///
/// ## Errors
///
/// The first completion to fail decides the error and the rest are aborted;
/// a collector that panics reports a 500.
async fn collect_all_completions(
    rxs: Vec<tokio_mpsc::UnboundedReceiver<EngineEvent>>,
    timeout: Option<Duration>,
) -> Result<Vec<CompletionData>, (StatusCode, Json<serde_json::Value>)> {
    let mut collectors = tokio::task::JoinSet::new();
    for (i, rx) in rxs.into_iter().enumerate() {
        collectors.spawn(async move { (i, collect_one_completion(rx, timeout).await) });
    }
    let mut results = Vec::with_capacity(collectors.len());
    while let Some(joined) = collectors.join_next().await {
        let (i, data) = joined.map_err(|_| {
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Task panic",
                "server_error",
            )
        })?;
        results.push((i, data?));
    }
    results.sort_by_key(|(i, _)| *i);
    Ok(results.into_iter().map(|(_, data)| data).collect())
}

/// `POST /v1/chat/completions`: the OpenAI-compatible chat completion
/// handler.
///
/// Validates the request, loads the model on demand (waiting if a load is
/// already in flight), injects tool and JSON instructions into the system
/// message, renders and encodes the prompt, then enqueues `n` sequences with
/// per-completion seeds. Non-streaming responses collect every sequence and
/// apply tool-call parsing (which takes priority over JSON mode) and schema
/// validation; streaming responses emit SSE chunks as tokens arrive, except
/// that requests with tools buffer each full completion first so tool calls
/// can be parsed before deltas are emitted. The whole request runs under an
/// `http.request` span parented to any upstream `traceparent` header and
/// cloned into each engine request, so engine spans join the caller's
/// distributed trace.
///
/// ## Errors
///
/// Returns 400 for invalid parameters, tools, or response formats; 404 when
/// the model is not found; 408 on non-streaming request timeout; 429 when
/// the request queue is full; 500 for template, tokenizer, engine, or load
/// failures; 503 when the engine channel is closed.
pub(super) async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(body): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, (StatusCode, Json<serde_json::Value>)> {
    if body.messages.is_empty() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "messages must not be empty",
            "invalid_request_error",
        ));
    }

    let model_id = body.model.as_deref().unwrap_or("").to_string();
    if model_id.is_empty() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "model field is required",
            "invalid_request_error",
        ));
    }

    if let Some(ref effort) = body.reasoning_effort
        && !matches!(effort.as_str(), "low" | "medium" | "high")
    {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            format!("reasoning_effort must be 'low', 'medium', or 'high'; got '{effort}'"),
            "invalid_request_error",
        ));
    }

    let n = body.n.unwrap_or(1);
    if n == 0 {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "n must be at least 1",
            "invalid_request_error",
        ));
    }
    if n > 128 {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "n must be at most 128",
            "invalid_request_error",
        ));
    }

    if let Err(msg) = validate_sampling_params(&body) {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            msg,
            "invalid_request_error",
        ));
    }

    if let Some(ref lb) = body.logit_bias
        && !lb.is_null()
        && !lb.is_object()
    {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "logit_bias must be a JSON object mapping token IDs to biases",
            "invalid_request_error",
        ));
    }

    if let Some(response_format) = &body.response_format
        && let Err(msg) = validate_response_format_request(response_format)
    {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            msg,
            "invalid_request_error",
        ));
    }

    let tool_config = build_tool_config(
        body.tools.as_deref(),
        body.tool_choice.as_ref(),
        body.parallel_tool_calls,
    )
    .map_err(|msg| error_response(StatusCode::BAD_REQUEST, msg, "invalid_request_error"))?;
    let has_tools = !tool_config.tools.is_empty();

    let request_id = uuid::Uuid::new_v4().to_string();
    let t_request = std::time::Instant::now();

    let http_span = tracing::info_span!(
        "http.request",
        request_id = %request_id,
        model_id = %model_id,
        stream = body.stream.unwrap_or(false),
    );
    let _ = http_span.set_parent(crate::telemetry::extract_context(&headers));

    let get_result = {
        let mut mgr = state.manager.lock().await;
        let keep_alive_override = body.keep_alive.map(Duration::from_secs);
        mgr.get_or_load(&model_id, Arc::clone(&state.manager), keep_alive_override)
    };

    let t_after_lock = t_request.elapsed();

    let handle = match get_result {
        GetResult::Ready(h) => {
            tracing::debug!(
                model_id = %model_id,
                manager_ready_ms = t_after_lock.as_secs_f64() * 1000.0,
                "model manager returned ready handle"
            );
            h
        }
        GetResult::Wait(rx) => {
            let load_result = rx.await.map_err(|_| {
                error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Model loader dropped",
                    "server_error",
                )
            })?;
            let h = load_result.map_err(|e| {
                let status = if e.contains("not found") {
                    StatusCode::NOT_FOUND
                } else {
                    StatusCode::INTERNAL_SERVER_ERROR
                };
                error_response(status, e, "server_error")
            })?;
            tracing::debug!(
                model_id = %model_id,
                load_completed_ms = t_request.elapsed().as_secs_f64() * 1000.0,
                "model load completed"
            );
            h
        }
    };

    let t_template = std::time::Instant::now();
    let enable_thinking =
        body.enable_thinking.unwrap_or(false) && handle.tokenizer.has_thinking_support();

    let reasoning_effort = body.reasoning_effort.clone();

    let json_mode = body
        .response_format
        .as_ref()
        .map(|rf| rf.format_type == "json_object" || rf.format_type == "json_schema")
        .unwrap_or(false);

    let tools_value: Option<serde_json::Value> = (!tool_config.tools.is_empty())
        .then(|| serde_json::to_value(&tool_config.tools).ok())
        .flatten();
    let native_tools = tools_value.as_ref().is_some_and(|tools| {
        template_renders_tools(
            &handle.tokenizer,
            &body.messages,
            enable_thinking,
            reasoning_effort.as_deref(),
            tools,
        )
    });

    let messages_for_prompt: std::borrow::Cow<[ChatMessage]> = {
        let mut extra_parts: Vec<String> = Vec::new();
        if has_tools && native_tools {
            extra_parts.extend(tool_choice_instruction(&tool_config));
        } else if has_tools {
            extra_parts.push(tools_system_instruction(&tool_config.tools, &tool_config));
        }
        if json_mode && let Some(rf) = &body.response_format {
            extra_parts.push(json_system_instruction(rf));
        }

        if extra_parts.is_empty() {
            std::borrow::Cow::Borrowed(&body.messages)
        } else {
            let mut msgs = body.messages.clone();
            let combined = extra_parts.join("\n\n");
            if let Some(sys) = msgs.iter_mut().find(|m| m.role == "system") {
                let existing = sys.content.take().unwrap_or_default();
                sys.content = Some(if existing.is_empty() {
                    combined
                } else {
                    format!("{existing}\n\n{combined}")
                });
            } else {
                msgs.insert(
                    0,
                    ChatMessage {
                        role: "system".to_string(),
                        content: Some(combined),
                        reasoning_content: None,
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    },
                );
            }
            std::borrow::Cow::Owned(msgs)
        }
    };

    let prompt = apply_chat_template(
        &handle.tokenizer,
        &messages_for_prompt,
        enable_thinking,
        reasoning_effort.as_deref(),
        tools_value,
    )
    .map_err(|e| {
        error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            e.to_string(),
            "template_render_failed",
        )
    })?;
    let template_ms = t_template.elapsed().as_secs_f64() * 1000.0;

    let t_encode = std::time::Instant::now();
    let prompt_tokens = handle.tokenizer.encode(&prompt).map_err(|e| {
        error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            e.to_string(),
            "server_error",
        )
    })?;
    let encode_ms = t_encode.elapsed().as_secs_f64() * 1000.0;
    let prompt_len = prompt_tokens.len();

    tracing::debug!(
        request_id = %request_id,
        model_id = %model_id,
        template_ms,
        encode_ms,
        prompt_tokens = prompt_len,
        pre_engine_total_ms = t_request.elapsed().as_secs_f64() * 1000.0,
        "prompt preparation timings"
    );

    let stop_strings: Vec<String> = match &body.stop {
        Some(StopParam::Single(s)) => vec![s.clone()],
        Some(StopParam::Multiple(strings)) => strings.clone(),
        None => Vec::new(),
    }
    .into_iter()
    .filter(|s| !s.is_empty())
    .collect();

    let logit_bias: Option<Vec<(u32, f32)>> = match &body.logit_bias {
        Some(serde_json::Value::Object(map)) if !map.is_empty() => {
            let pairs: Vec<(u32, f32)> = map
                .iter()
                .filter_map(|(k, v)| {
                    let token_id: u32 = k.parse().ok()?;
                    let bias: f32 = v.as_f64()? as f32;
                    Some((token_id, bias.clamp(-100.0, 100.0)))
                })
                .collect();
            if pairs.is_empty() { None } else { Some(pairs) }
        }
        _ => None,
    };

    let wants_logprobs = body.logprobs.unwrap_or(false);
    let req_top_n = body.top_logprobs.unwrap_or(0);
    // min 1 so the chosen token's logprob is always returned.
    let top_logprobs_k: usize = if wants_logprobs { req_top_n.max(1) } else { 0 };

    let base_sampling_params = SamplingParams {
        temperature: body.temperature.unwrap_or(0.7),
        top_k: body.top_k.unwrap_or(0),
        top_p: body.top_p.unwrap_or(1.0),
        min_p: body.min_p.unwrap_or(0.0),
        repetition_penalty: body.repetition_penalty.unwrap_or(1.0),
        repetition_window: body.repetition_window.unwrap_or(0),
        frequency_penalty: body.frequency_penalty.unwrap_or(0.0),
        presence_penalty: body.presence_penalty.unwrap_or(0.0),
        seed: body.seed,
        logit_bias,
        top_logprobs_k,
    };

    let capacity = handle.max_seq_len.min(handle.kv_capacity_tokens);
    if prompt_len >= capacity {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": format!(
                        "This model's maximum context length is {capacity} tokens. However, \
                         your messages resulted in {prompt_len} tokens. Please reduce the \
                         length of the messages."
                    ),
                    "type": "invalid_request_error",
                    "param": "messages",
                    "code": "context_length_exceeded",
                }
            })),
        ));
    }
    let remaining = capacity - prompt_len;
    let max_tokens = body
        .max_completion_tokens
        .or(body.max_tokens)
        .unwrap_or(remaining)
        .min(remaining);

    let mut completion_rxs: Vec<tokio_mpsc::UnboundedReceiver<EngineEvent>> = Vec::with_capacity(n);

    for i in 0..n {
        let (response_tx, response_rx) = tokio_mpsc::unbounded_channel();
        // Distinct seed per completion so n>1 produces different outputs.
        let seed = base_sampling_params.seed.map(|s| s.wrapping_add(i as u64));
        let sampling_params = SamplingParams {
            seed,
            ..base_sampling_params.clone()
        };

        let json_mode = body.response_format.as_ref().and_then(|rf| {
            use crate::constrain::{JsonMode, JsonSpec, SchemaArena};
            match rf.format_type.as_str() {
                "json_object" => Some(JsonSpec {
                    mode: JsonMode::Object,
                    arena: SchemaArena::unconstrained(),
                }),
                "json_schema" => {
                    let schema = rf
                        .json_schema
                        .as_ref()
                        .and_then(|spec| spec.schema.as_ref());
                    let root_is_object = schema
                        .map(|s| {
                            s["type"].as_str() == Some("object") || s.get("properties").is_some()
                        })
                        .unwrap_or(true);
                    Some(JsonSpec {
                        mode: if root_is_object {
                            JsonMode::Object
                        } else {
                            JsonMode::Value
                        },
                        arena: schema
                            .map(SchemaArena::compile)
                            .unwrap_or_else(SchemaArena::unconstrained),
                    })
                }
                _ => None,
            }
        });
        handle
            .request_tx
            .try_send(IncomingRequest {
                request_id: request_id.clone(),
                prompt_tokens: prompt_tokens.clone(),
                sampling_params,
                max_tokens,
                response_tx,
                model_id: model_id.clone(),
                enqueued_at: std::time::Instant::now(),
                enable_thinking,
                thinking_open: prompt_opens_thinking(&prompt),
                stop_strings: stop_strings.clone(),
                json_mode,
                parent_span: http_span.clone(),
            })
            .map_err(|e| match e {
                tokio::sync::mpsc::error::TrySendError::Full(_) => error_response(
                    StatusCode::TOO_MANY_REQUESTS,
                    "Server overloaded: too many queued requests. Retry later.",
                    "rate_limit_error",
                ),
                tokio::sync::mpsc::error::TrySendError::Closed(_) => error_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Engine unavailable",
                    "server_error",
                ),
            })?;
        completion_rxs.push(response_rx);
    }

    let chat_id = make_chat_id();
    let created = unix_timestamp();
    let fp = system_fingerprint(&model_id);
    let stream = body.stream.unwrap_or(false);
    let include_usage = body
        .stream_options
        .as_ref()
        .and_then(|o| o.include_usage)
        .unwrap_or(false);

    if stream {
        let (sse_tx, sse_rx) =
            tokio_mpsc::unbounded_channel::<Result<Event, std::convert::Infallible>>();

        if has_tools {
            let buffered_rxs = completion_rxs;
            let model_id_clone = model_id.clone();
            let tool_config_clone = tool_config.clone();
            let req_timeout = state.request_timeout;

            tokio::spawn(async move {
                let completions = tokio::select! {
                    collected = collect_all_completions(buffered_rxs, req_timeout) => collected,
                    _ = sse_tx.closed() => return,
                };
                let completions = match completions {
                    Ok(completions) => completions,
                    Err((_, err_json)) => {
                        let _ =
                            sse_tx
                                .send(Ok(Event::default()
                                    .data(serde_json::to_string(&err_json.0).unwrap())));
                        let _ = sse_tx.send(Ok(Event::default().data("[DONE]")));
                        return;
                    }
                };
                let total_completion_tokens: usize =
                    completions.iter().map(|data| data.completion_tokens).sum();

                for (choice_idx, data) in completions.into_iter().enumerate() {
                    let role_chunk = ChatCompletionChunk {
                        id: chat_id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model_id_clone.clone(),
                        choices: vec![ChunkChoice {
                            index: choice_idx,
                            delta: Delta {
                                role: Some("assistant".to_string()),
                                content: None,
                                reasoning_content: None,
                                tool_calls: None,
                            },
                            finish_reason: None,
                            logprobs: None,
                        }],
                        usage: None,
                        system_fingerprint: Some(fp.clone()),
                    };
                    if sse_tx
                        .send(Ok(
                            Event::default().data(serde_json::to_string(&role_chunk).unwrap())
                        ))
                        .is_err()
                    {
                        return;
                    }

                    let raw = strip_json_fences(data.content.trim());
                    if let Some(tool_calls) = try_parse_tool_calls(raw, &tool_config_clone) {
                        let header_deltas: Vec<ToolCallDelta> = tool_calls
                            .iter()
                            .enumerate()
                            .map(|(tool_idx, call)| ToolCallDelta {
                                index: tool_idx,
                                id: Some(call.id.clone()),
                                call_type: Some("function".to_string()),
                                function: ToolCallFunctionDelta {
                                    name: Some(call.function.name.clone()),
                                    arguments: String::new(),
                                },
                            })
                            .collect();
                        let header_chunk = ChatCompletionChunk {
                            id: chat_id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: model_id_clone.clone(),
                            choices: vec![ChunkChoice {
                                index: choice_idx,
                                delta: Delta {
                                    role: None,
                                    content: None,
                                    reasoning_content: None,
                                    tool_calls: Some(header_deltas),
                                },
                                finish_reason: None,
                                logprobs: None,
                            }],
                            usage: None,
                            system_fingerprint: Some(fp.clone()),
                        };
                        if sse_tx
                            .send(Ok(Event::default()
                                .data(serde_json::to_string(&header_chunk).unwrap())))
                            .is_err()
                        {
                            return;
                        }

                        for (tool_idx, call) in tool_calls.iter().enumerate() {
                            if call.function.arguments.is_empty() {
                                continue;
                            }
                            let arg_chunk = ChatCompletionChunk {
                                id: chat_id.clone(),
                                object: "chat.completion.chunk".to_string(),
                                created,
                                model: model_id_clone.clone(),
                                choices: vec![ChunkChoice {
                                    index: choice_idx,
                                    delta: Delta {
                                        role: None,
                                        content: None,
                                        reasoning_content: None,
                                        tool_calls: Some(vec![ToolCallDelta {
                                            index: tool_idx,
                                            id: None,
                                            call_type: None,
                                            function: ToolCallFunctionDelta {
                                                name: None,
                                                arguments: call.function.arguments.clone(),
                                            },
                                        }]),
                                    },
                                    finish_reason: None,
                                    logprobs: None,
                                }],
                                usage: None,
                                system_fingerprint: Some(fp.clone()),
                            };
                            if sse_tx
                                .send(Ok(Event::default()
                                    .data(serde_json::to_string(&arg_chunk).unwrap())))
                                .is_err()
                            {
                                return;
                            }
                        }

                        let finish_chunk = ChatCompletionChunk {
                            id: chat_id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: model_id_clone.clone(),
                            choices: vec![ChunkChoice {
                                index: choice_idx,
                                delta: Delta {
                                    role: None,
                                    content: None,
                                    reasoning_content: None,
                                    tool_calls: None,
                                },
                                finish_reason: Some("tool_calls".to_string()),
                                logprobs: None,
                            }],
                            usage: None,
                            system_fingerprint: Some(fp.clone()),
                        };
                        if sse_tx
                            .send(Ok(Event::default()
                                .data(serde_json::to_string(&finish_chunk).unwrap())))
                            .is_err()
                        {
                            return;
                        }
                    } else {
                        let final_content = data.content.clone();
                        if !final_content.is_empty() {
                            let chunk = ChatCompletionChunk {
                                id: chat_id.clone(),
                                object: "chat.completion.chunk".to_string(),
                                created,
                                model: model_id_clone.clone(),
                                choices: vec![ChunkChoice {
                                    index: choice_idx,
                                    delta: Delta {
                                        role: None,
                                        content: Some(final_content),
                                        reasoning_content: None,
                                        tool_calls: None,
                                    },
                                    finish_reason: None,
                                    logprobs: None,
                                }],
                                usage: None,
                                system_fingerprint: Some(fp.clone()),
                            };
                            if sse_tx
                                .send(Ok(
                                    Event::default().data(serde_json::to_string(&chunk).unwrap())
                                ))
                                .is_err()
                            {
                                return;
                            }
                        }

                        let finish_chunk = ChatCompletionChunk {
                            id: chat_id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: model_id_clone.clone(),
                            choices: vec![ChunkChoice {
                                index: choice_idx,
                                delta: Delta {
                                    role: None,
                                    content: None,
                                    reasoning_content: None,
                                    tool_calls: None,
                                },
                                finish_reason: Some(data.finish_reason.clone()),
                                logprobs: None,
                            }],
                            usage: None,
                            system_fingerprint: Some(fp.clone()),
                        };
                        if sse_tx
                            .send(Ok(Event::default()
                                .data(serde_json::to_string(&finish_chunk).unwrap())))
                            .is_err()
                        {
                            return;
                        }
                    }
                }

                if include_usage {
                    let usage_chunk = ChatCompletionChunk {
                        id: chat_id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model_id_clone.clone(),
                        choices: vec![],
                        usage: Some(Usage {
                            prompt_tokens: prompt_len,
                            completion_tokens: total_completion_tokens,
                            total_tokens: prompt_len + total_completion_tokens,
                            completion_tokens_details: None,
                        }),
                        system_fingerprint: Some(fp.clone()),
                    };
                    if sse_tx
                        .send(Ok(
                            Event::default().data(serde_json::to_string(&usage_chunk).unwrap())
                        ))
                        .is_err()
                    {
                        return;
                    }
                }

                let _ = sse_tx.send(Ok(Event::default().data("[DONE]")));
            });
        } else if n == 1 {
            let mut response_rx = completion_rxs.remove(0);
            let model_id_clone = model_id.clone();
            let req_timeout = state.request_timeout;

            tokio::spawn(async move {
                let watchdog_tx = sse_tx.clone();
                let inner = async move {
                    let role_chunk = ChatCompletionChunk {
                        id: chat_id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model_id_clone.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: Delta {
                                role: Some("assistant".to_string()),
                                content: None,
                                reasoning_content: None,
                                tool_calls: None,
                            },
                            finish_reason: None,
                            logprobs: None,
                        }],
                        usage: None,
                        system_fingerprint: Some(fp.clone()),
                    };
                    if sse_tx
                        .send(Ok(
                            Event::default().data(serde_json::to_string(&role_chunk).unwrap())
                        ))
                        .is_err()
                    {
                        return;
                    }

                    while let Some(event) = response_rx.recv().await {
                        match event {
                            EngineEvent::Token {
                                text,
                                logprob_entries,
                            } => {
                                if text.is_empty() && logprob_entries.is_empty() {
                                    continue;
                                }
                                let chunk_logprobs =
                                    if wants_logprobs && !logprob_entries.is_empty() {
                                        Some(build_logprobs_content(&logprob_entries, req_top_n))
                                    } else if wants_logprobs {
                                        Some(Logprobs {
                                            content: vec![],
                                            refusal: None,
                                        })
                                    } else {
                                        None
                                    };
                                let chunk = ChatCompletionChunk {
                                    id: chat_id.clone(),
                                    object: "chat.completion.chunk".to_string(),
                                    created,
                                    model: model_id_clone.clone(),
                                    choices: vec![ChunkChoice {
                                        index: 0,
                                        delta: Delta {
                                            role: None,
                                            content: Some(text),
                                            reasoning_content: None,
                                            tool_calls: None,
                                        },
                                        finish_reason: None,
                                        logprobs: chunk_logprobs,
                                    }],
                                    usage: None,
                                    system_fingerprint: Some(fp.clone()),
                                };
                                if sse_tx
                                    .send(Ok(Event::default()
                                        .data(serde_json::to_string(&chunk).unwrap())))
                                    .is_err()
                                {
                                    break;
                                }
                            }
                            EngineEvent::ReasoningToken(text) => {
                                let chunk = ChatCompletionChunk {
                                    id: chat_id.clone(),
                                    object: "chat.completion.chunk".to_string(),
                                    created,
                                    model: model_id_clone.clone(),
                                    choices: vec![ChunkChoice {
                                        index: 0,
                                        delta: Delta {
                                            role: None,
                                            content: None,
                                            reasoning_content: Some(text),
                                            tool_calls: None,
                                        },
                                        finish_reason: None,
                                        logprobs: None,
                                    }],
                                    usage: None,
                                    system_fingerprint: Some(fp.clone()),
                                };
                                if sse_tx
                                    .send(Ok(Event::default()
                                        .data(serde_json::to_string(&chunk).unwrap())))
                                    .is_err()
                                {
                                    break;
                                }
                            }
                            EngineEvent::Finish {
                                finish_reason,
                                completion_tokens,
                            } => {
                                let chunk = ChatCompletionChunk {
                                    id: chat_id.clone(),
                                    object: "chat.completion.chunk".to_string(),
                                    created,
                                    model: model_id_clone.clone(),
                                    choices: vec![ChunkChoice {
                                        index: 0,
                                        delta: Delta {
                                            role: None,
                                            content: None,
                                            reasoning_content: None,
                                            tool_calls: None,
                                        },
                                        finish_reason: Some(finish_reason),
                                        logprobs: None,
                                    }],
                                    usage: None,
                                    system_fingerprint: Some(fp.clone()),
                                };
                                if sse_tx
                                    .send(Ok(Event::default()
                                        .data(serde_json::to_string(&chunk).unwrap())))
                                    .is_err()
                                {
                                    break;
                                }

                                if include_usage {
                                    let usage_chunk = ChatCompletionChunk {
                                        id: chat_id.clone(),
                                        object: "chat.completion.chunk".to_string(),
                                        created,
                                        model: model_id_clone.clone(),
                                        choices: vec![],
                                        usage: Some(Usage {
                                            prompt_tokens: prompt_len,
                                            completion_tokens,
                                            total_tokens: prompt_len + completion_tokens,
                                            completion_tokens_details: None,
                                        }),
                                        system_fingerprint: Some(fp.clone()),
                                    };
                                    if sse_tx
                                        .send(Ok(Event::default()
                                            .data(serde_json::to_string(&usage_chunk).unwrap())))
                                        .is_err()
                                    {
                                        break;
                                    }
                                }
                            }
                            EngineEvent::StreamEnd => {
                                if sse_tx.send(Ok(Event::default().data("[DONE]"))).is_err() {
                                    break;
                                }
                                break;
                            }
                            EngineEvent::Error(msg) => {
                                let err = serde_json::json!({
                                    "error": { "message": msg, "type": "server_error", "param": null, "code": null }
                                });
                                let _ =
                                    sse_tx
                                        .send(Ok(Event::default()
                                            .data(serde_json::to_string(&err).unwrap())));
                                let _ = sse_tx.send(Ok(Event::default().data("[DONE]")));
                                break;
                            }
                        }
                    }
                };
                run_streaming_with_timeout(inner, watchdog_tx, req_timeout).await;
            });
        } else {
            let (merged_tx, merged_rx) = tokio_mpsc::unbounded_channel::<(usize, EngineEvent)>();
            for (i, rx) in completion_rxs.into_iter().enumerate() {
                let tx = merged_tx.clone();
                tokio::spawn(async move {
                    let mut rx = rx;
                    while let Some(ev) = rx.recv().await {
                        if tx.send((i, ev)).is_err() {
                            break;
                        }
                    }
                });
            }
            drop(merged_tx);

            let model_id_clone = model_id.clone();
            let req_timeout = state.request_timeout;

            tokio::spawn(async move {
                let watchdog_tx = sse_tx.clone();
                let inner = async move {
                    for i in 0..n {
                        let role_chunk = ChatCompletionChunk {
                            id: chat_id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: model_id_clone.clone(),
                            choices: vec![ChunkChoice {
                                index: i,
                                delta: Delta {
                                    role: Some("assistant".to_string()),
                                    content: None,
                                    reasoning_content: None,
                                    tool_calls: None,
                                },
                                finish_reason: None,
                                logprobs: None,
                            }],
                            usage: None,
                            system_fingerprint: Some(fp.clone()),
                        };
                        if sse_tx
                            .send(Ok(
                                Event::default().data(serde_json::to_string(&role_chunk).unwrap())
                            ))
                            .is_err()
                        {
                            return;
                        }
                    }

                    let mut rx = merged_rx;
                    let mut stream_ends: usize = 0;
                    let mut total_completion_tokens: usize = 0;

                    while let Some((idx, event)) = rx.recv().await {
                        match event {
                            EngineEvent::Token {
                                text,
                                logprob_entries,
                            } => {
                                if text.is_empty() && logprob_entries.is_empty() {
                                    continue;
                                }
                                let chunk_logprobs =
                                    if wants_logprobs && !logprob_entries.is_empty() {
                                        Some(build_logprobs_content(&logprob_entries, req_top_n))
                                    } else if wants_logprobs {
                                        Some(Logprobs {
                                            content: vec![],
                                            refusal: None,
                                        })
                                    } else {
                                        None
                                    };
                                let chunk = ChatCompletionChunk {
                                    id: chat_id.clone(),
                                    object: "chat.completion.chunk".to_string(),
                                    created,
                                    model: model_id_clone.clone(),
                                    choices: vec![ChunkChoice {
                                        index: idx,
                                        delta: Delta {
                                            role: None,
                                            content: Some(text),
                                            reasoning_content: None,
                                            tool_calls: None,
                                        },
                                        finish_reason: None,
                                        logprobs: chunk_logprobs,
                                    }],
                                    usage: None,
                                    system_fingerprint: Some(fp.clone()),
                                };
                                if sse_tx
                                    .send(Ok(Event::default()
                                        .data(serde_json::to_string(&chunk).unwrap())))
                                    .is_err()
                                {
                                    break;
                                }
                            }
                            EngineEvent::ReasoningToken(text) => {
                                let chunk = ChatCompletionChunk {
                                    id: chat_id.clone(),
                                    object: "chat.completion.chunk".to_string(),
                                    created,
                                    model: model_id_clone.clone(),
                                    choices: vec![ChunkChoice {
                                        index: idx,
                                        delta: Delta {
                                            role: None,
                                            content: None,
                                            reasoning_content: Some(text),
                                            tool_calls: None,
                                        },
                                        finish_reason: None,
                                        logprobs: None,
                                    }],
                                    usage: None,
                                    system_fingerprint: Some(fp.clone()),
                                };
                                if sse_tx
                                    .send(Ok(Event::default()
                                        .data(serde_json::to_string(&chunk).unwrap())))
                                    .is_err()
                                {
                                    break;
                                }
                            }
                            EngineEvent::Finish {
                                finish_reason,
                                completion_tokens,
                            } => {
                                total_completion_tokens += completion_tokens;
                                let chunk = ChatCompletionChunk {
                                    id: chat_id.clone(),
                                    object: "chat.completion.chunk".to_string(),
                                    created,
                                    model: model_id_clone.clone(),
                                    choices: vec![ChunkChoice {
                                        index: idx,
                                        delta: Delta {
                                            role: None,
                                            content: None,
                                            reasoning_content: None,
                                            tool_calls: None,
                                        },
                                        finish_reason: Some(finish_reason),
                                        logprobs: None,
                                    }],
                                    usage: None,
                                    system_fingerprint: Some(fp.clone()),
                                };
                                if sse_tx
                                    .send(Ok(Event::default()
                                        .data(serde_json::to_string(&chunk).unwrap())))
                                    .is_err()
                                {
                                    break;
                                }
                            }
                            EngineEvent::StreamEnd => {
                                stream_ends += 1;
                                if stream_ends == n {
                                    if include_usage {
                                        let usage_chunk = ChatCompletionChunk {
                                            id: chat_id.clone(),
                                            object: "chat.completion.chunk".to_string(),
                                            created,
                                            model: model_id_clone.clone(),
                                            choices: vec![],
                                            usage: Some(Usage {
                                                prompt_tokens: prompt_len,
                                                completion_tokens: total_completion_tokens,
                                                total_tokens: prompt_len + total_completion_tokens,
                                                completion_tokens_details: None,
                                            }),
                                            system_fingerprint: Some(fp.clone()),
                                        };
                                        if sse_tx
                                            .send(Ok(Event::default().data(
                                                serde_json::to_string(&usage_chunk).unwrap(),
                                            )))
                                            .is_err()
                                        {
                                            break;
                                        }
                                    }
                                    if sse_tx.send(Ok(Event::default().data("[DONE]"))).is_err() {
                                        break;
                                    }
                                    break;
                                }
                            }
                            EngineEvent::Error(msg) => {
                                let err = serde_json::json!({
                                    "error": { "message": msg, "type": "server_error", "param": null, "code": null }
                                });
                                let _ =
                                    sse_tx
                                        .send(Ok(Event::default()
                                            .data(serde_json::to_string(&err).unwrap())));
                                let _ = sse_tx.send(Ok(Event::default().data("[DONE]")));
                                break;
                            }
                        }
                    }
                };
                run_streaming_with_timeout(inner, watchdog_tx, req_timeout).await;
            });
        }

        let sse_stream = UnboundedReceiverStream::new(sse_rx);
        Ok(Sse::new(sse_stream).into_response())
    } else {
        let completions = collect_all_completions(completion_rxs, state.request_timeout).await?;

        let mut all_choices: Vec<Choice> = Vec::with_capacity(n);
        let mut total_completion_tokens: usize = 0;
        let mut total_reasoning_tokens: usize = 0;

        for (i, data) in completions.into_iter().enumerate() {
            total_completion_tokens += data.completion_tokens;
            total_reasoning_tokens += data.reasoning_tokens;

            let reasoning_opt = if data.reasoning_content.is_empty() {
                None
            } else {
                Some(data.reasoning_content.clone())
            };

            let logprobs = if wants_logprobs {
                Some(build_logprobs_content(&data.logprob_entries, req_top_n))
            } else {
                None
            };

            let (response_msg, finish_reason) = if has_tools {
                let raw = strip_json_fences(data.content.trim());
                if let Some(tool_calls) = try_parse_tool_calls(raw, &tool_config) {
                    (
                        ResponseMessage {
                            role: "assistant".to_string(),
                            content: None,
                            reasoning_content: reasoning_opt,
                            tool_calls: Some(tool_calls),
                        },
                        "tool_calls".to_string(),
                    )
                } else {
                    (
                        ResponseMessage {
                            role: "assistant".to_string(),
                            content: Some(data.content.clone()),
                            reasoning_content: reasoning_opt,
                            tool_calls: None,
                        },
                        data.finish_reason.clone(),
                    )
                }
            } else if json_mode {
                let raw = strip_json_fences(data.content.trim()).to_string();
                let json_schema_spec = body
                    .response_format
                    .as_ref()
                    .and_then(|rf| rf.json_schema.as_ref());
                let schema_fail = json_schema_spec
                    .and_then(|js| js.schema.as_ref())
                    .map(
                        |spec| match serde_json::from_str::<serde_json::Value>(&raw) {
                            Ok(val) => !validate_against_schema(&val, spec),
                            Err(_) => true,
                        },
                    )
                    .unwrap_or(false);
                let (response_content, finish_reason) = if schema_fail {
                    let is_strict = json_schema_spec.and_then(|js| js.strict).unwrap_or(false);
                    if is_strict {
                        tracing::warn!(
                            "model output did not match response_format.json_schema (strict=true)"
                        );
                        (None, "content_filter".to_string())
                    } else {
                        tracing::warn!(
                            "model output did not match response_format.json_schema; returning raw content"
                        );
                        (Some(raw), data.finish_reason.clone())
                    }
                } else {
                    (Some(raw), data.finish_reason.clone())
                };
                (
                    ResponseMessage {
                        role: "assistant".to_string(),
                        content: response_content,
                        reasoning_content: reasoning_opt,
                        tool_calls: None,
                    },
                    finish_reason,
                )
            } else {
                (
                    ResponseMessage {
                        role: "assistant".to_string(),
                        content: Some(data.content.clone()),
                        reasoning_content: reasoning_opt,
                        tool_calls: None,
                    },
                    data.finish_reason.clone(),
                )
            };

            all_choices.push(Choice {
                index: i,
                message: response_msg,
                finish_reason,
                logprobs,
            });
        }

        let completion_tokens_details = if total_reasoning_tokens > 0 {
            Some(CompletionTokensDetails {
                reasoning_tokens: total_reasoning_tokens,
            })
        } else {
            None
        };

        let response = ChatCompletionResponse {
            id: chat_id,
            object: "chat.completion".to_string(),
            created,
            model: model_id,
            choices: all_choices,
            usage: Usage {
                prompt_tokens: prompt_len,
                completion_tokens: total_completion_tokens,
                total_tokens: prompt_len + total_completion_tokens,
                completion_tokens_details,
            },
            system_fingerprint: Some(fp),
        };

        Ok(Json(response).into_response())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Contract: dropping the collection future closes every completion
    /// channel, which is the only way a client disconnect reaches the engine
    /// loop; detached collectors left them open until generation ended.
    #[tokio::test]
    async fn dropping_the_collection_closes_every_completion_channel() {
        let (tx_a, rx_a) = tokio_mpsc::unbounded_channel::<EngineEvent>();
        let (tx_b, rx_b) = tokio_mpsc::unbounded_channel::<EngineEvent>();
        let handler = tokio::spawn(collect_all_completions(vec![rx_a, rx_b], None));
        tokio::task::yield_now().await;
        assert!(
            !tx_a.is_closed() && !tx_b.is_closed(),
            "collectors are waiting"
        );

        handler.abort();
        for _ in 0..100 {
            if tx_a.is_closed() && tx_b.is_closed() {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(
            tx_a.is_closed() && tx_b.is_closed(),
            "every completion channel must close with the handler"
        );
    }

    fn sample_tools() -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                tool_type: "function".to_string(),
                function: super::super::types::FunctionDefinition {
                    name: "get_weather".to_string(),
                    description: None,
                    parameters: Some(json!({
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"],
                        "additionalProperties": false
                    })),
                    strict: Some(true),
                },
            },
            ToolDefinition {
                tool_type: "function".to_string(),
                function: super::super::types::FunctionDefinition {
                    name: "search_docs".to_string(),
                    description: None,
                    parameters: Some(json!({
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"],
                        "additionalProperties": false
                    })),
                    strict: Some(true),
                },
            },
        ]
    }

    #[test]
    fn build_tool_config_supports_forced_function_choice() {
        let tools = sample_tools();
        let config = build_tool_config(
            Some(&tools),
            Some(&json!({
                "type": "function",
                "function": { "name": "get_weather" }
            })),
            Some(true),
        )
        .expect("forced function should parse");

        assert_eq!(config.tools.len(), 1);
        assert_eq!(config.tools[0].function.name, "get_weather");
        assert!(!config.parallel_tool_calls);
        assert_eq!(
            config.choice_mode,
            ToolChoiceMode::ForcedFunction {
                name: "get_weather".to_string()
            }
        );
    }

    #[test]
    fn build_tool_config_supports_allowed_tools_wrapper() {
        let tools = sample_tools();
        let config = build_tool_config(
            Some(&tools),
            Some(&json!({
                "type": "allowed_tools",
                "allowed_tools": {
                    "mode": "required",
                    "tools": [
                        { "type": "function", "function": { "name": "search_docs" } }
                    ]
                }
            })),
            Some(false),
        )
        .expect("allowed_tools should parse");

        assert_eq!(config.tools.len(), 1);
        assert_eq!(config.tools[0].function.name, "search_docs");
        assert_eq!(config.choice_mode, ToolChoiceMode::Required);
        assert!(!config.parallel_tool_calls);
    }

    #[test]
    fn try_parse_tool_calls_accepts_nested_function_shape() {
        let config = ToolConfig {
            tools: sample_tools(),
            choice_mode: ToolChoiceMode::Auto,
            parallel_tool_calls: true,
        };
        let parsed = try_parse_tool_calls(
            r#"{"tool_calls":[{"id":"call_123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"Paris\"}"}}]}"#,
            &config,
        )
        .expect("tool call should parse");

        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].id, "call_123");
        assert_eq!(parsed[0].function.name, "get_weather");
        assert_eq!(parsed[0].function.arguments, "{\"location\":\"Paris\"}");
    }

    /// Contract: the XML-ish format emitted by Qwen3.5-class templates
    /// (`<tool_call><function=NAME><parameter=K>V</parameter>…`) parses into
    /// proper tool calls: multiline values preserved, JSON-typed scalars
    /// coerced, free-form reasoning before the block tolerated, parallel
    /// calls supported, unknown functions filtered out.
    #[test]
    fn try_parse_tool_calls_accepts_qwen35_function_xml() {
        let config = ToolConfig {
            tools: sample_tools(),
            choice_mode: ToolChoiceMode::Auto,
            parallel_tool_calls: true,
        };
        let raw = "I should check the weather first.\n\n<tool_call>\n<function=get_weather>\n<parameter=location>\nRome\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=note>\nline one\nline two\n</parameter>\n</function>\n</tool_call>";
        let parsed = try_parse_tool_calls(raw, &config).expect("xml tool call should parse");

        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].function.name, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&parsed[0].function.arguments).unwrap();
        assert_eq!(args["location"], "Rome");
        assert_eq!(args["days"], 3, "bare integers must coerce to numbers");
        assert_eq!(args["note"], "line one\nline two");

        // Parallel calls: two blocks give two calls; unknown function dropped.
        let raw2 = "<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=not_a_real_tool>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>";
        let parsed2 = try_parse_tool_calls(raw2, &config).expect("should parse");
        assert_eq!(parsed2.len(), 1, "unknown function must be filtered");
        assert_eq!(parsed2[0].function.name, "get_weather");
    }

    /// Contract: the JSON payloads that Qwen2.5/Qwen3-class templates wrap in
    /// Contract: a call whose arguments sit under `parameters`, which is how
    /// Llama 3 renders its tool calls, parses like one that says `arguments`.
    /// Contract: a bare JSON array of calls, which is what Granite's
    /// `<|tool_call|>[...]` leaves once the special token is dropped, parses
    /// like the `tool_calls` wrapper.
    #[test]
    fn a_bare_array_of_calls_parses() {
        let config = build_tool_config(Some(&sample_tools()), None, None).expect("config");
        let raw = r#"[{"name": "get_weather", "arguments": {"location": "Rome"}}]"#;
        let calls = try_parse_tool_calls(raw, &config).expect("parsed");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, r#"{"location":"Rome"}"#);
    }

    fn tokenizer_with_template(template: &str) -> (Tokenizer, tempfile::TempDir) {
        use tokenizers::models::wordlevel::WordLevel;
        let tmp = tempfile::tempdir().expect("tempdir");
        let model = WordLevel::builder()
            .vocab([("[UNK]".to_string(), 0u32)].into_iter().collect())
            .unk_token("[UNK]".to_string())
            .build()
            .expect("wordlevel");
        tokenizers::Tokenizer::new(model)
            .save(tmp.path().join("tokenizer.json"), false)
            .expect("save tokenizer");
        std::fs::write(
            tmp.path().join("tokenizer_config.json"),
            serde_json::to_string(&json!({"chat_template": template})).expect("json"),
        )
        .expect("write config");
        let tokenizer =
            Tokenizer::from_dir(tmp.path().to_str().expect("utf-8 path")).expect("load");
        (tokenizer, tmp)
    }

    /// Contract: a template that renders `tools` itself is recognised from
    /// its output, and one that ignores them is not, so the definitions
    /// reach the model exactly once either way.
    #[test]
    fn a_template_is_known_to_render_tools_by_its_output() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some("weather".to_string()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];
        let tools = serde_json::to_value(sample_tools()).expect("tools");
        let native = "{% if tools %}<tools>{{ tools | tojson }}</tools>{% endif %}\
                      {% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}";
        let (tokenizer, _tmp) = tokenizer_with_template(native);
        assert!(template_renders_tools(
            &tokenizer, &messages, false, None, &tools
        ));

        let blind = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}";
        let (tokenizer, _tmp) = tokenizer_with_template(blind);
        assert!(!template_renders_tools(
            &tokenizer, &messages, false, None, &tools
        ));
    }

    /// Contract: the doubled braces that Qwen2.5's GGUF template teaches by
    /// example do not hide the call.
    #[test]
    fn doubled_braces_around_a_tool_call_are_unwrapped() {
        let config = build_tool_config(Some(&sample_tools()), None, None).expect("config");
        for closing in ["}}}", "}}"] {
            let raw = format!(
                "<tool_call>\n{{{{\"name\": \"get_weather\", \"arguments\": {{\"location\": \"Rome\"{closing}\n</tool_call>"
            );
            let calls = try_parse_tool_calls(&raw, &config).expect("parsed");
            assert_eq!(calls.len(), 1, "{raw}");
            assert_eq!(calls[0].function.name, "get_weather");
            assert_eq!(calls[0].function.arguments, r#"{"location":"Rome"}"#);
        }
        assert!(parse_json_block("{{\"name\": 1").is_none(), "unfinished");
    }

    /// Contract: a Mistral call reaches the parser as `name{json}` once its
    /// `[TOOL_CALLS]` and `[ARGS]` tokens are dropped, also when parallel,
    /// while prose that merely starts with a tool name is not a call.
    #[test]
    fn a_mistral_call_without_its_special_tokens_parses() {
        let config = build_tool_config(Some(&sample_tools()), None, None).expect("config");
        let calls =
            try_parse_tool_calls(r#"get_weather{"location": "Rome"}"#, &config).expect("parsed");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, r#"{"location":"Rome"}"#);

        let two = r#"get_weather{"location": "Rome"} search_docs{"query": "rain"}"#;
        let calls = try_parse_tool_calls(two, &config).expect("parsed");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].function.name, "search_docs");
        assert_eq!(calls[1].function.arguments, r#"{"query":"rain"}"#);

        assert!(try_parse_tool_calls("get_weather is a tool I could use.", &config).is_none());
        assert!(
            try_parse_tool_calls(r#"get_weather{"location": "Rome"} and more"#, &config).is_none()
        );
    }

    /// Contract: gemma-4's call parses from the GGUF text, markers and quote
    /// tokens included, and from the safetensors text that dropped them as
    /// special, where a comma inside a string value does not split it.
    #[test]
    fn a_gemma_call_parses_with_and_without_its_markers() {
        let config = build_tool_config(Some(&sample_tools()), None, None).expect("config");
        let gguf = "<|tool_call>call:get_weather{location:<|\"|>Rome<|\"|>}<tool_call|>";
        let calls = try_parse_tool_calls(gguf, &config).expect("parsed");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, r#"{"location":"Rome"}"#);

        let mut tools = sample_tools();
        tools.push(ToolDefinition {
            tool_type: "function".to_string(),
            function: super::super::types::FunctionDefinition {
                name: "get_forecast".to_string(),
                description: None,
                parameters: Some(json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}
                })),
                strict: None,
            },
        });
        let config = build_tool_config(Some(&tools), None, None).expect("config");
        let bare = "call:get_forecast{days:3,city:Rome, Italy}";
        let calls = try_parse_tool_calls(bare, &config).expect("parsed");
        assert_eq!(
            calls[0].function.arguments,
            r#"{"days":3,"city":"Rome, Italy"}"#
        );

        let two = "call:get_weather{location:Rome}call:search_docs{query:x}";
        assert_eq!(try_parse_tool_calls(two, &config).expect("parsed").len(), 2);
        assert!(try_parse_tool_calls("call: me later", &config).is_none());
    }

    /// Contract: a tool definition renders into the prompt with only the
    /// fields the request gave, as transformers and llama.cpp render it; a
    /// `"strict": null` the model never saw in training cost four tokens and
    /// turned Qwen3-1.7B's "2+2" into "6".
    #[test]
    fn absent_tool_fields_are_not_rendered_as_null() {
        let value = serde_json::to_value(sample_tools()).expect("tools");
        let text = value.to_string();
        assert!(!text.contains("null"), "{text}");
        assert!(text.contains(r#""strict":true"#), "a given field stays");
    }

    /// Contract: a template that refuses tool-call ids of any length but nine,
    /// as Mistral's do, renders a round trip whose ids came from the client in
    /// another shape, with the call and its result still matching; the ids the
    /// server makes already have that shape.
    #[test]
    fn mistral_s_nine_character_ids_are_satisfied_for_any_client_id() {
        let template = "{% for m in messages %}{% if m.tool_calls %}{% for c in m.tool_calls %}\
            {% if not c.id is defined or c.id|length != 9 %}{{ raise_exception(\"Tool call IDs should be alphanumeric strings with length 9!\") }}{% endif %}\
            call:{{ c.id }} {% endfor %}{% endif %}{% if m.role == 'tool' %}result:{{ m.tool_call_id }} {% endif %}{% endfor %}";
        let messages: Vec<ChatMessage> = serde_json::from_value(json!([
            {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"}
        ]))
        .expect("messages");
        let (tokenizer, _tmp) = tokenizer_with_template(template);
        let rendered =
            apply_chat_template(&tokenizer, &messages, false, None, None).expect("render");
        let id = rendered
            .split("call:")
            .nth(1)
            .unwrap()
            .split(' ')
            .next()
            .unwrap();
        assert_eq!(id.len(), 9, "{rendered}");
        assert!(rendered.contains(&format!("result:{id} ")), "{rendered}");
        assert_eq!(make_tool_call_id().len(), 9);
        assert_eq!(
            nine_char_id("abcdefghi"),
            "abcdefghi",
            "a fitting id is kept"
        );
    }

    /// Contract: a template that knows only alternating user and assistant
    /// turns still renders a tool round trip, with the call and its result
    /// folded into text and the roles alternating.
    #[test]
    fn a_tool_round_trip_renders_on_a_template_without_tool_turns() {
        let template = "{% for m in messages %}{% if (m.role == 'user') != (loop.index0 % 2 == 0) %}\
            {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}\
            {% if m.role == 'user' %}[INST] {{ m.content }} [/INST]{% else %}{{ m.content }}</s>{% endif %}{% endfor %}";
        let messages: Vec<ChatMessage> = serde_json::from_value(json!([
            {"role": "user", "content": "weather in Rome?"},
            {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Rome\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "{\"temp_c\": 21}"}
        ]))
        .expect("messages");
        let (tokenizer, _tmp) = tokenizer_with_template(template);
        let rendered =
            apply_chat_template(&tokenizer, &messages, false, None, None).expect("render");
        assert!(rendered.contains("get_weather"), "{rendered}");
        assert!(
            rendered.contains("[INST] Result of tool call c1: {\"temp_c\": 21} [/INST]"),
            "{rendered}"
        );
        assert!(fold_tool_turns(&messages[..1]).is_none(), "nothing to fold");
        let mut with_system = vec![ChatMessage {
            role: "system".to_string(),
            content: Some("tools follow".to_string()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];
        with_system.extend(messages.iter().cloned());
        assert!(
            apply_chat_template(&tokenizer, &with_system, false, None, None).is_ok(),
            "the system turn the tool instruction adds goes too"
        );
    }

    #[test]
    fn a_llama_style_call_with_parameters_parses() {
        let config = ToolConfig {
            tools: sample_tools(),
            choice_mode: ToolChoiceMode::Auto,
            parallel_tool_calls: true,
        };
        let raw =
            r#"{"tool_calls": [{"name": "get_weather", "parameters": {"location": "Rome"}}]}"#;
        let calls = try_parse_tool_calls(raw, &config).expect("one call");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, r#"{"location":"Rome"}"#);
        let raw = r#"{"name": "get_weather", "parameters": {"location": "Rome"}}"#;
        assert_eq!(
            try_parse_tool_calls(raw, &config).expect("one call").len(),
            1
        );
        let bare =
            r#"{"type": "function", "function": "get_weather", "parameters": {"city": "Rome"}}"#;
        let calls = try_parse_tool_calls(bare, &config).expect("a bare function name");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, r#"{"city":"Rome"}"#);
    }

    /// Contract: a final `<tool_call>` block the model never closed still
    /// parses, in both payload shapes, as the reference parsers read it.
    ///
    /// Qwen2.5 stops right after the JSON payload often enough; while the
    /// tags were stripped from GGUF output the bare JSON happened to parse,
    /// and with the tags kept a closing tag must not be required.
    #[test]
    fn an_unclosed_final_tool_call_block_still_parses() {
        let raw = "<tool_call>\n{\n  \"tool_calls\": [\n    {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}\n  ]\n}";
        let calls = parse_tool_call_tag_blocks(raw).expect("one call");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, "{\"city\":\"Rome\"}");

        let raw = "<tool_call>\n{\"name\": \"a\", \"arguments\": {}}\n</tool_call>\n<tool_call>\n<function=b>\n<parameter=x>\n1\n</parameter>\n</function>";
        let calls = parse_tool_call_tag_blocks(raw).expect("two calls");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].function.name, "b");
        assert_eq!(calls[1].function.arguments, "{\"x\":1}");

        assert!(parse_tool_call_tag_blocks("<tool_call>\n").is_none());
    }

    /// `<tool_call>` tags parse into proper tool calls: both the canonical
    /// single `{"name": …, "arguments": …}` object and the
    /// `{"tool_calls": […]}` shape models produce when echoing the injected
    /// system-prompt instruction. Parallel blocks supported, unknown
    /// functions filtered, reasoning text before the block tolerated.
    #[test]
    fn try_parse_tool_calls_accepts_json_inside_tool_call_tags() {
        let config = ToolConfig {
            tools: sample_tools(),
            choice_mode: ToolChoiceMode::Auto,
            parallel_tool_calls: true,
        };

        // Canonical Qwen: single {"name", "arguments"} object per block.
        let raw = "Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}\n</tool_call>";
        let parsed = try_parse_tool_calls(raw, &config).expect("canonical block should parse");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].function.name, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&parsed[0].function.arguments).unwrap();
        assert_eq!(args["location"], "Paris");

        // Instruction-echo shape observed E2E on Qwen2.5-1.5B.
        let raw2 = "<tool_call>\n{\"tool_calls\": [{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Rome\"}}]}\n</tool_call>";
        let parsed2 = try_parse_tool_calls(raw2, &config).expect("echo shape should parse");
        assert_eq!(parsed2.len(), 1);
        assert_eq!(parsed2[0].function.name, "get_weather");

        // Two blocks become parallel calls; unknown function filtered.
        let raw3 = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"not_a_real_tool\", \"arguments\": {}}\n</tool_call>";
        let parsed3 = try_parse_tool_calls(raw3, &config).expect("should parse");
        assert_eq!(parsed3.len(), 1, "unknown function must be filtered");

        // parallel_tool_calls=false truncates to the first call.
        let config_serial = ToolConfig {
            tools: sample_tools(),
            choice_mode: ToolChoiceMode::Auto,
            parallel_tool_calls: false,
        };
        let raw4 = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Rome\"}}\n</tool_call>";
        let parsed4 = try_parse_tool_calls(raw4, &config_serial).expect("should parse");
        assert_eq!(parsed4.len(), 1, "serial mode must truncate to one call");
    }

    #[test]
    fn try_parse_tool_calls_wraps_direct_json_for_forced_function() {
        let config = ToolConfig {
            tools: vec![sample_tools()[0].clone()],
            choice_mode: ToolChoiceMode::ForcedFunction {
                name: "get_weather".to_string(),
            },
            parallel_tool_calls: false,
        };
        let parsed = try_parse_tool_calls(r#"{"location":"Paris"}"#, &config)
            .expect("forced function fallback should parse");

        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].function.name, "get_weather");
        assert_eq!(parsed[0].function.arguments, r#"{"location":"Paris"}"#);
    }

    #[test]
    fn try_parse_tool_calls_truncates_when_parallel_disabled() {
        let config = ToolConfig {
            tools: sample_tools(),
            choice_mode: ToolChoiceMode::Auto,
            parallel_tool_calls: false,
        };
        let parsed = try_parse_tool_calls(
            r#"{"tool_calls":[{"name":"get_weather","arguments":{"location":"Paris"}},{"name":"search_docs","arguments":{"query":"weather"}}]}"#,
            &config,
        )
        .expect("tool calls should parse");

        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].function.name, "get_weather");
    }

    #[test]
    fn validate_schema_shape_rejects_non_strict_object_schema() {
        let err = validate_schema_shape(
            &json!({
                "type": "object",
                "properties": {
                    "location": { "type": "string" }
                },
                "required": ["location"]
            }),
            SchemaStrictness::Strict,
            "schema",
        )
        .expect_err("strict schema should require additionalProperties false");

        assert!(err.contains("additionalProperties"));
    }

    #[test]
    fn validate_against_schema_supports_refs_anyof_and_null_union() {
        let schema = json!({
            "type": "object",
            "properties": {
                "result": {
                    "anyOf": [
                        { "$ref": "#/$defs/weather" },
                        { "type": "null" }
                    ]
                }
            },
            "required": ["result"],
            "additionalProperties": false,
            "$defs": {
                "weather": {
                    "type": "object",
                    "properties": {
                        "location": { "type": "string" },
                        "temperature_c": { "type": ["number", "null"] }
                    },
                    "required": ["location", "temperature_c"],
                    "additionalProperties": false
                }
            }
        });

        assert!(validate_against_schema(
            &json!({"result":{"location":"Paris","temperature_c":21.5}}),
            &schema
        ));
        assert!(validate_against_schema(&json!({"result":null}), &schema));
        assert!(!validate_against_schema(
            &json!({"result":{"location":"Paris","extra":true}}),
            &schema
        ));
    }

    #[test]
    fn validate_response_format_request_requires_json_schema_schema() {
        let err = validate_response_format_request(&ResponseFormat {
            format_type: "json_schema".to_string(),
            json_schema: Some(super::super::types::JsonSchemaSpec {
                name: "answer".to_string(),
                schema: None,
                strict: Some(true),
                description: None,
            }),
        })
        .expect_err("json_schema should require schema");

        assert!(err.contains("json_schema.schema"));
    }
}
