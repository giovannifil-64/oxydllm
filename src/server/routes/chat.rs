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

#[derive(Serialize, Clone)]
struct TopLogprobItem {
    token: String,
    logprob: f32,
    bytes: Option<Vec<u8>>,
}

#[derive(Serialize, Clone)]
struct TokenLogprob {
    token: String,
    logprob: f32,
    bytes: Option<Vec<u8>>,
    top_logprobs: Vec<TopLogprobItem>,
}

#[derive(Serialize, Clone)]
struct Logprobs {
    content: Vec<TokenLogprob>,
    refusal: Option<String>,
}

// Separate from ChatMessage so `content` serializes as explicit `null` when tool_calls present.
#[derive(Serialize)]
struct ResponseMessage {
    role: String,
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
}

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

#[derive(Serialize)]
struct Choice {
    index: usize,
    message: ResponseMessage,
    finish_reason: String,
    logprobs: Option<Logprobs>,
}

#[derive(Serialize)]
struct CompletionTokensDetails {
    reasoning_tokens: usize,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    completion_tokens_details: Option<CompletionTokensDetails>,
}

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

#[derive(Serialize)]
struct ChunkChoice {
    index: usize,
    delta: Delta,
    finish_reason: Option<String>,
    logprobs: Option<Logprobs>,
}

#[derive(Serialize)]
struct ToolCallFunctionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    arguments: String,
}

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

#[derive(Clone, Debug, PartialEq, Eq)]
enum ToolChoiceMode {
    None,
    Auto,
    Required,
    ForcedFunction { name: String },
}

#[derive(Clone, Debug)]
struct ToolConfig {
    tools: Vec<ToolDefinition>,
    choice_mode: ToolChoiceMode,
    parallel_tool_calls: bool,
}

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

fn system_fingerprint(model_id: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    model_id.hash(&mut h);
    env!("CARGO_PKG_VERSION").hash(&mut h);
    format!("fp_{:012x}", h.finish() & 0xFFFF_FFFF_FFFF)
}

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

fn make_tool_call_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("call_{:012x}", seq)
}

fn is_valid_function_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
}

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

fn is_array_schema(schema_obj: &serde_json::Map<String, serde_json::Value>) -> bool {
    match schema_obj.get("type") {
        Some(serde_json::Value::String(t)) => t == "array",
        Some(serde_json::Value::Array(arr)) => arr.iter().any(|v| v.as_str() == Some("array")),
        _ => schema_obj.contains_key("items"),
    }
}

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

fn validate_schema_shape(
    schema: &serde_json::Value,
    strictness: SchemaStrictness,
    path: &str,
) -> Result<(), String> {
    validate_schema_shape_inner(schema, schema, path, strictness, 0)
}

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

fn tools_system_instruction(tools: &[ToolDefinition], config: &ToolConfig) -> String {
    let tools_json = serde_json::to_string_pretty(tools).unwrap_or_default();
    let choice_instruction = match &config.choice_mode {
        ToolChoiceMode::None => "Do not call any tool. Respond normally.".to_string(),
        ToolChoiceMode::Auto => {
            "If no tool is needed, respond normally. Otherwise, emit tool calls only.".to_string()
        }
        ToolChoiceMode::Required => {
            "You must call one or more tools. Do not respond with plain text.".to_string()
        }
        ToolChoiceMode::ForcedFunction { name } => {
            format!(
                "You must call exactly one tool named \"{name}\". Do not respond with plain text."
            )
        }
    };
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

fn parse_tool_call_entry(call: &serde_json::Value) -> Option<ToolCall> {
    let call_obj = call.as_object()?;
    let name = call_obj
        .get("function")
        .and_then(|f| f.get("name"))
        .and_then(|v| v.as_str())
        .or_else(|| call_obj.get("name").and_then(|v| v.as_str()))?
        .to_string();
    let args = call_obj
        .get("function")
        .and_then(|f| f.get("arguments"))
        .or_else(|| call_obj.get("arguments"))?;
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

fn try_parse_tool_calls(raw: &str, config: &ToolConfig) -> Option<Vec<ToolCall>> {
    let stripped = strip_json_fences(raw.trim());
    let value: serde_json::Value = serde_json::from_str(stripped).ok()?;
    let mut result: Vec<ToolCall> = if let Some(calls) =
        value.get("tool_calls").and_then(|v| v.as_array())
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

fn validate_against_schema(value: &serde_json::Value, schema: &serde_json::Value) -> bool {
    validate_against_schema_inner(value, schema, schema, 0)
}

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

fn parse_tool_reference_name(tool_ref: &serde_json::Value) -> Option<String> {
    tool_ref
        .get("function")
        .and_then(|v| v.get("name"))
        .and_then(|v| v.as_str())
        .or_else(|| tool_ref.get("name").and_then(|v| v.as_str()))
        .map(str::to_string)
}

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

// Mirrors OpenAI Chat Completions ranges; out-of-range becomes invalid_request_error.
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
                if let Ok(prompt) = try_render(&msgs_ref, tools) {
                    tracing::warn!(
                        "system role not supported by this model template; retrying without system message"
                    );
                    return Ok(prompt);
                }
            }

            tracing::error!(error = ?e, "chat template rendering failed");
            Err(e)
        }
    }
}

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

struct CompletionData {
    content: String,
    reasoning_content: String,
    reasoning_tokens: usize,
    finish_reason: String,
    completion_tokens: usize,
    logprob_entries: Vec<EngineLogprobEntry>,
}

// On timeout: cancel inner (drops sse_tx/rx, engine aborts the sequence) and
// emit error chunk + [DONE] on the retained sse_tx clone.
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

// On timeout expiry: returns 408 and drops the receiver, which triggers the
// engine loop to abort the sequence via tracker.tx.is_closed().
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

pub(super) async fn chat_completions(
    State(state): State<Arc<AppState>>,
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

    let messages_for_prompt: std::borrow::Cow<[ChatMessage]> = {
        let needs_json_instr = json_mode;
        let needs_tool_instr = has_tools;

        if needs_json_instr || needs_tool_instr {
            let mut msgs = body.messages.clone();

            let mut extra_parts: Vec<String> = Vec::new();
            if needs_tool_instr {
                extra_parts.push(tools_system_instruction(&tool_config.tools, &tool_config));
            }
            if needs_json_instr && let Some(rf) = &body.response_format {
                extra_parts.push(json_system_instruction(rf));
            }

            if !extra_parts.is_empty() {
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
            }

            std::borrow::Cow::Owned(msgs)
        } else {
            std::borrow::Cow::Borrowed(&body.messages)
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

    // Single-token stops only.
    let extra_stop_token_ids: Vec<u32> = match &body.stop {
        Some(StopParam::Single(s)) => {
            let ids = handle.tokenizer.encode(s).unwrap_or_default();
            if ids.len() == 1 { ids } else { Vec::new() }
        }
        Some(StopParam::Multiple(strings)) => strings
            .iter()
            .filter_map(|s| {
                let ids = handle.tokenizer.encode(s).unwrap_or_default();
                if ids.len() == 1 { Some(ids[0]) } else { None }
            })
            .collect(),
        None => Vec::new(),
    };

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

    let remaining = handle.max_seq_len.saturating_sub(prompt_len);
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
                extra_stop_token_ids: extra_stop_token_ids.clone(),
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
                let mut handles: Vec<_> = buffered_rxs
                    .into_iter()
                    .map(|rx| Some(tokio::spawn(collect_one_completion(rx, req_timeout))))
                    .collect();

                let mut completions = Vec::with_capacity(handles.len());
                let mut total_completion_tokens = 0usize;
                for idx in 0..handles.len() {
                    let handle = handles[idx].take().expect("handle consumed once");
                    let data = match handle.await {
                        Ok(Ok(data)) => data,
                        Ok(Err((_, err_json))) => {
                            for h in handles[idx + 1..].iter_mut().filter_map(|h| h.take()) {
                                h.abort();
                            }
                            let _ = sse_tx
                                .send(Ok(Event::default()
                                    .data(serde_json::to_string(&err_json.0).unwrap())));
                            let _ = sse_tx.send(Ok(Event::default().data("[DONE]")));
                            return;
                        }
                        Err(_) => {
                            for h in handles[idx + 1..].iter_mut().filter_map(|h| h.take()) {
                                h.abort();
                            }
                            let _ = sse_tx.send(Ok(Event::default().data("[DONE]")));
                            return;
                        }
                    };
                    total_completion_tokens += data.completion_tokens;
                    completions.push((idx, data));
                }

                for (choice_idx, data) in completions {
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
                        let final_content = data.content.trim().to_string();
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
        let req_timeout = state.request_timeout;
        let mut handles = Vec::with_capacity(n);
        for rx in completion_rxs {
            handles.push(tokio::spawn(collect_one_completion(rx, req_timeout)));
        }

        let mut all_choices: Vec<Choice> = Vec::with_capacity(n);
        let mut total_completion_tokens: usize = 0;
        let mut total_reasoning_tokens: usize = 0;

        for (i, handle) in handles.into_iter().enumerate() {
            let data = handle.await.map_err(|_| {
                error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Task panic",
                    "server_error",
                )
            })??;

            total_completion_tokens += data.completion_tokens;
            total_reasoning_tokens += data.reasoning_tokens;

            let reasoning_opt = if data.reasoning_content.is_empty() {
                None
            } else {
                Some(data.reasoning_content.trim().to_string())
            };

            let logprobs = if wants_logprobs {
                Some(build_logprobs_content(&data.logprob_entries, req_top_n))
            } else {
                None
            };

            // Tool call detection takes priority over JSON mode.
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
                            content: Some(data.content.trim().to_string()),
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
                        content: Some(data.content.trim().to_string()),
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
