use anyhow::{Context, Result};
use minijinja::{Environment, Value};
use serde::Serialize;

use crate::server::ChatMessage;

const MAX_STRFTIME_FMT_LEN: usize = 128;

#[derive(Default)]
pub struct TemplateOptions<'a> {
    pub bos_token: Option<&'a str>,
    pub eos_token: Option<&'a str>,
    pub add_generation_prompt: bool,
    pub enable_thinking: bool,
    /// Harmony models (gpt-oss): low | medium | high. Omitted from the Jinja
    /// context when None so the template's own default applies.
    pub reasoning_effort: Option<&'a str>,
    pub tools: Option<serde_json::Value>,
}

pub fn apply_chat_template(
    template: &str,
    messages: &[ChatMessage],
    opts: TemplateOptions<'_>,
) -> Result<String> {
    let template = preprocess_template(template);

    let mut env = Environment::new();

    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_filter("tojson", tojson_filter);
    env.add_function("raise_exception", raise_exception);
    env.add_function("strftime_now", strftime_now);
    env.add_function("namespace", minijinja::functions::namespace);
    env.add_template("chat", &template)
        .context("Failed to compile chat template")?;

    let tmpl = env.get_template("chat").unwrap();

    let ctx = TemplateContext {
        messages: messages
            .iter()
            .map(|m| TemplateMessage {
                role: &m.role,
                content: m.content.as_deref().unwrap_or(""),
                tool_calls: m
                    .tool_calls
                    .as_ref()
                    .and_then(|tc| serde_json::to_value(tc).ok())
                    .map(parse_arguments_for_template),
                tool_call_id: m.tool_call_id.as_deref(),
                name: m.name.as_deref(),
            })
            .collect(),
        bos_token: opts.bos_token.unwrap_or(""),
        eos_token: opts.eos_token.unwrap_or(""),
        add_generation_prompt: opts.add_generation_prompt,
        enable_thinking: opts.enable_thinking,
        reasoning_effort: opts.reasoning_effort,
        tools: opts.tools,
    };

    let rendered = tmpl
        .render(&ctx)
        .context("Failed to render chat template")?;

    Ok(rendered)
}

/// HF chat templates are written against the transformers convention where
/// `tool_call.function.arguments` is a mapping (Qwen3.5 iterates it with
/// `|items`; Qwen3 serializes it with `|tojson`). The OpenAI wire format we
/// receive stores arguments as a JSON *string*: parse it for the template
/// context so both conventions render correctly. Unparseable strings are
/// passed through unchanged.
fn parse_arguments_for_template(mut tool_calls: serde_json::Value) -> serde_json::Value {
    if let Some(calls) = tool_calls.as_array_mut() {
        for call in calls {
            if let Some(args) = call.pointer_mut("/function/arguments")
                && let Some(s) = args.as_str()
                && let Ok(parsed) = serde_json::from_str::<serde_json::Value>(s)
            {
                *args = parsed;
            }
        }
    }
    tool_calls
}

pub fn format_plain_chat(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        let content = msg.content.as_deref().unwrap_or("");
        match msg.role.as_str() {
            "system" => {
                prompt.push_str("System: ");
                prompt.push_str(content);
                prompt.push('\n');
            }
            "assistant" => {
                if let Some(tc) = &msg.tool_calls {
                    prompt.push_str("Assistant (tool call): ");
                    if let Ok(s) = serde_json::to_string(tc) {
                        prompt.push_str(&s);
                    }
                    prompt.push('\n');
                } else {
                    prompt.push_str("Assistant: ");
                    prompt.push_str(content);
                    prompt.push('\n');
                }
            }
            "tool" => {
                let id = msg.tool_call_id.as_deref().unwrap_or("");
                prompt.push_str(&format!("Tool result [{}]: {}\n", id, content));
            }
            _ => {
                prompt.push_str("User: ");
                prompt.push_str(content);
                prompt.push('\n');
            }
        }
    }
    prompt.push_str("Assistant: ");
    prompt
}

pub fn format_turn_chat(
    messages: &[ChatMessage],
    bos_token: Option<&str>,
    start_turn_token: &str,
    end_turn_token: &str,
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> String {
    let mut prompt = String::new();
    if let Some(bos) = bos_token {
        prompt.push_str(bos);
    }

    let mut start_idx = 0usize;
    let first_is_system = messages
        .first()
        .map(|m| m.role == "system" || m.role == "developer")
        .unwrap_or(false);

    if enable_thinking || first_is_system {
        prompt.push_str(start_turn_token);
        prompt.push_str("system\n");
        if enable_thinking {
            prompt.push_str("<|think|>");
        }
        if first_is_system {
            let content = messages[0].content.as_deref().unwrap_or("").trim();
            prompt.push_str(content);
            start_idx = 1;
        }
        prompt.push_str(end_turn_token);
        prompt.push('\n');
    }

    for message in &messages[start_idx..] {
        let role = match message.role.as_str() {
            "assistant" => "model",
            "tool" => "tool",
            other => other,
        };

        prompt.push_str(start_turn_token);
        prompt.push_str(role);
        prompt.push('\n');

        if message.role == "assistant" {
            if let Some(tc) = &message.tool_calls {
                if let Ok(s) = serde_json::to_string(tc) {
                    prompt.push_str(&s);
                }
            } else {
                prompt.push_str(message.content.as_deref().unwrap_or("").trim());
            }
        } else if message.role == "tool" {
            let id = message.tool_call_id.as_deref().unwrap_or("");
            let content = message.content.as_deref().unwrap_or("");
            prompt.push_str(&format!("[{}]: {}", id, content));
        } else {
            prompt.push_str(message.content.as_deref().unwrap_or("").trim());
        }

        prompt.push_str(end_turn_token);
        prompt.push('\n');
    }

    if add_generation_prompt {
        prompt.push_str(start_turn_token);
        prompt.push_str("model\n");
    }

    prompt
}

pub fn format_mistral_inst_chat(
    messages: &[ChatMessage],
    bos_token: Option<&str>,
    eos_token: Option<&str>,
    has_system_prompt_token: bool,
) -> String {
    let mut prompt = String::new();
    if let Some(bos) = bos_token {
        prompt.push_str(bos);
    }

    let mut pending_system: Option<String> = None;
    for message in messages {
        let content = message.content.as_deref().unwrap_or("").trim();
        match message.role.as_str() {
            "system" | "developer" => {
                if has_system_prompt_token {
                    prompt.push_str("[SYSTEM_PROMPT] ");
                    prompt.push_str(content);
                    prompt.push_str("[/SYSTEM_PROMPT]");
                } else {
                    pending_system = Some(content.to_string());
                }
            }
            "assistant" => {
                if let Some(tc) = &message.tool_calls {
                    prompt.push_str("[TOOL_CALLS] ");
                    if let Ok(s) = serde_json::to_string(tc) {
                        prompt.push_str(&s);
                    }
                } else {
                    prompt.push(' ');
                    prompt.push_str(content);
                }
                if let Some(eos) = eos_token {
                    prompt.push_str(eos);
                }
            }
            "tool" => {
                prompt.push_str("[TOOL_RESULTS] ");
                prompt.push_str(content);
                prompt.push_str("[/TOOL_RESULTS]");
            }
            _ => {
                prompt.push_str("[INST] ");
                if let Some(sys) = pending_system.take() {
                    prompt.push_str(&sys);
                    prompt.push_str("\n\n");
                }
                prompt.push_str(content);
                prompt.push_str("[/INST]");
            }
        }
    }
    prompt
}

fn preprocess_template(template: &str) -> String {
    let mut t = template.to_string();
    t = t.replace("[::-1]", "|reverse");
    t
}

#[derive(Serialize)]
struct TemplateContext<'a> {
    messages: Vec<TemplateMessage<'a>>,
    bos_token: &'a str,
    eos_token: &'a str,
    add_generation_prompt: bool,
    enable_thinking: bool,
    /// Omitted when None so the template's own default ("medium") applies.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct TemplateMessage<'a> {
    role: &'a str,
    content: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<&'a str>,
}

fn tojson_filter(
    value: Value,
    kwargs: minijinja::value::Kwargs,
) -> std::result::Result<Value, minijinja::Error> {
    let indent: Option<u32> = kwargs.get("indent")?;
    kwargs.assert_all_used()?;
    let json_string = serde_json::to_string(&value).map_err(|e| {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            format!("tojson serialization failed: {e}"),
        )
    })?;
    let result = if let Some(n) = indent {
        let parsed: serde_json::Value = serde_json::from_str(&json_string).map_err(|e| {
            minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                format!("tojson re-parse failed: {e}"),
            )
        })?;
        let indent_str = " ".repeat(n as usize);
        let fmt = serde_json::ser::PrettyFormatter::with_indent(indent_str.as_bytes());
        let mut buf = Vec::new();
        let mut ser = serde_json::Serializer::with_formatter(&mut buf, fmt);
        serde::Serialize::serialize(&parsed, &mut ser).map_err(|e| {
            minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                format!("tojson pretty-print failed: {e}"),
            )
        })?;
        String::from_utf8(buf).unwrap_or(json_string)
    } else {
        json_string
    };
    Ok(Value::from(result))
}

fn raise_exception(msg: String) -> Result<Value, minijinja::Error> {
    Err(minijinja::Error::new(
        minijinja::ErrorKind::InvalidOperation,
        msg,
    ))
}

fn strftime_now(fmt: String) -> std::result::Result<String, minijinja::Error> {
    if fmt.len() > MAX_STRFTIME_FMT_LEN {
        return Err(minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            format!(
                "strftime_now format is too long (max {} chars)",
                MAX_STRFTIME_FMT_LEN
            ),
        ));
    }
    if fmt.contains('\n') || fmt.contains('\r') {
        return Err(minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            "strftime_now format must be a single line",
        ));
    }
    Ok(chrono::Local::now().format(&fmt).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_chat_template_renders_basic_template() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some("hello".to_string()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];

        let rendered = apply_chat_template(
            "{% for m in messages %}{{ m.role }}: {{ m.content }}{% endfor %}",
            &messages,
            TemplateOptions::default(),
        )
        .expect("template should render");

        assert_eq!(rendered, "user: hello");
    }

    /// Contract: tool-call `arguments` reach the template as a MAPPING, not
    /// the OpenAI wire string: Qwen3.5's template iterates them with
    /// `|items` (a string there fails the whole render with a 500), and
    /// Qwen3-style `|tojson` would double-encode a string.
    #[test]
    fn tool_call_arguments_are_parsed_to_a_mapping_for_the_template() {
        let msg: ChatMessage = serde_json::from_value(serde_json::json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"city\": \"Rome\", \"days\": 3}"
                }
            }]
        }))
        .expect("valid assistant tool-call message");
        let messages = vec![msg];

        // Mirrors Qwen3.5's `for name, value in tool_call.arguments|items`.
        let rendered = apply_chat_template(
            "{% for m in messages %}{% for tc in m.tool_calls %}{{ tc.function.name }}({% for k, v in tc.function.arguments|items %}{{ k }}={{ v }};{% endfor %}){% endfor %}{% endfor %}",
            &messages,
            TemplateOptions::default(),
        )
        .expect("template with |items over arguments must render");

        assert_eq!(rendered, "get_weather(city=Rome;days=3;)");
    }

    // Contract: reasoning_effort reaches the template when set, and is ABSENT
    // (not null) when None: the harmony template's `is not defined` default
    // ("medium") must fire. This mirrors gpt-oss's chat_template.jinja lines
    // 203-206.
    #[test]
    fn reasoning_effort_renders_and_defaults_via_is_not_defined() {
        let template = "{%- if reasoning_effort is not defined %}\
                        {%- set reasoning_effort = \"medium\" %}\
                        {%- endif %}Reasoning: {{ reasoning_effort }}";
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some("hi".to_string()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];

        let low = apply_chat_template(
            template,
            &messages,
            TemplateOptions {
                reasoning_effort: Some("low"),
                ..Default::default()
            },
        )
        .expect("render with effort");
        assert_eq!(low, "Reasoning: low");

        let default = apply_chat_template(template, &messages, TemplateOptions::default())
            .expect("render without effort");
        assert_eq!(default, "Reasoning: medium");
    }

    // Renders the real downloaded gpt-oss harmony template (skipped when the
    // checkpoint isn't present): the rendered prompt must carry the requested
    // reasoning effort, and default to medium when the field is absent.
    #[test]
    fn gpt_oss_real_template_renders_reasoning_effort() {
        let home = std::env::var("HOME").unwrap_or_default();
        let path = format!("{home}/.oxydllm/models/openai/gpt-oss-20b/chat_template.jinja");
        let Ok(template) = std::fs::read_to_string(&path) else {
            return;
        };
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some("hello".to_string()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];

        let low = apply_chat_template(
            &template,
            &messages,
            TemplateOptions {
                add_generation_prompt: true,
                reasoning_effort: Some("low"),
                ..Default::default()
            },
        )
        .expect("harmony template should render");
        assert!(low.contains("Reasoning: low"), "missing 'Reasoning: low'");

        let default = apply_chat_template(
            &template,
            &messages,
            TemplateOptions {
                add_generation_prompt: true,
                ..Default::default()
            },
        )
        .expect("harmony template should render");
        assert!(
            default.contains("Reasoning: medium"),
            "missing default 'Reasoning: medium'"
        );
    }

    #[test]
    fn strftime_now_rejects_overlong_format() {
        let fmt = "x".repeat(MAX_STRFTIME_FMT_LEN + 1);
        let err = strftime_now(fmt).expect_err("overlong format should fail");
        let msg = err.to_string();
        assert!(msg.contains("strftime_now format is too long"));
    }

    #[test]
    fn strftime_now_rejects_newline_format() {
        let err =
            strftime_now("%Y-%m-%d\n%H:%M".to_string()).expect_err("multiline format should fail");
        let msg = err.to_string();
        assert!(msg.contains("strftime_now format must be a single line"));
    }

    #[test]
    fn apply_chat_template_fails_on_invalid_strftime_format() {
        let fmt = "x".repeat(MAX_STRFTIME_FMT_LEN + 1);
        let template = format!("{{{{ strftime_now(\"{}\") }}}}", fmt);
        let result = apply_chat_template(&template, &[], TemplateOptions::default());
        assert!(result.is_err());
    }
}
