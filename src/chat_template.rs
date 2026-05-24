use anyhow::{Context, Result};
use minijinja::{Environment, Value};
use serde::Serialize;

use crate::server::ChatMessage;

const MAX_STRFTIME_FMT_LEN: usize = 128;

pub fn apply_chat_template(
    template: &str,
    messages: &[ChatMessage],
    bos_token: Option<&str>,
    eos_token: Option<&str>,
    add_generation_prompt: bool,
    enable_thinking: bool,
    tools: Option<serde_json::Value>,
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
                    .and_then(|tc| serde_json::to_value(tc).ok()),
                tool_call_id: m.tool_call_id.as_deref(),
                name: m.name.as_deref(),
            })
            .collect(),
        bos_token: bos_token.unwrap_or(""),
        eos_token: eos_token.unwrap_or(""),
        add_generation_prompt,
        enable_thinking,
        tools,
    };

    let rendered = tmpl
        .render(&ctx)
        .context("Failed to render chat template")?;

    Ok(rendered)
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
                    prompt.push_str("[SYSTEM_PROMPT]");
                    prompt.push_str(content);
                    prompt.push_str("[/SYSTEM_PROMPT]");
                } else {
                    pending_system = Some(content.to_string());
                }
            }
            "assistant" => {
                if let Some(tc) = &message.tool_calls {
                    prompt.push_str("[TOOL_CALLS]");
                    if let Ok(s) = serde_json::to_string(tc) {
                        prompt.push_str(&s);
                    }
                } else {
                    prompt.push_str(content);
                }
                if let Some(eos) = eos_token {
                    prompt.push_str(eos);
                }
            }
            "tool" => {
                prompt.push_str("[TOOL_RESULTS]");
                prompt.push_str(content);
                prompt.push_str("[/TOOL_RESULTS]");
            }
            _ => {
                prompt.push_str("[INST]");
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
            None,
            None,
            false,
            false,
            None,
        )
        .expect("template should render");

        assert_eq!(rendered, "user: hello");
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
        let result = apply_chat_template(&template, &[], None, None, false, false, None);
        assert!(result.is_err());
    }
}
