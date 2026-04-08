use anyhow::{Context, Result};
use minijinja::{Environment, Value};
use serde::Serialize;

use crate::server::ChatMessage;

pub fn apply_chat_template(
    template: &str,
    messages: &[ChatMessage],
    bos_token: Option<&str>,
    eos_token: Option<&str>,
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> Result<String> {
    let template = preprocess_template(template);

    let mut env = Environment::new();

    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
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
                content: &m.content,
            })
            .collect(),
        bos_token: bos_token.unwrap_or(""),
        eos_token: eos_token.unwrap_or(""),
        add_generation_prompt,
        enable_thinking,
    };

    let rendered = tmpl
        .render(&ctx)
        .context("Failed to render chat template")?;

    Ok(rendered)
}

pub fn format_plain_chat(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                prompt.push_str("System: ");
                prompt.push_str(&msg.content);
                prompt.push('\n');
            }
            "assistant" => {
                prompt.push_str("Assistant: ");
                prompt.push_str(&msg.content);
                prompt.push('\n');
            }
            _ => {
                prompt.push_str("User: ");
                prompt.push_str(&msg.content);
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
            prompt.push_str(messages[0].content.trim());
            start_idx = 1;
        }
        prompt.push_str(end_turn_token);
        prompt.push('\n');
    }

    for message in &messages[start_idx..] {
        let role = if message.role == "assistant" {
            "model"
        } else {
            message.role.as_str()
        };

        prompt.push_str(start_turn_token);
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(message.content.trim());
        prompt.push_str(end_turn_token);
        prompt.push('\n');
    }

    if add_generation_prompt {
        prompt.push_str(start_turn_token);
        prompt.push_str("model\n");
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
}

#[derive(Serialize)]
struct TemplateMessage<'a> {
    role: &'a str,
    content: &'a str,
}

fn raise_exception(msg: String) -> Result<Value, minijinja::Error> {
    Err(minijinja::Error::new(
        minijinja::ErrorKind::InvalidOperation,
        msg,
    ))
}

fn strftime_now(fmt: String) -> String {
    chrono::Local::now().format(&fmt).to_string()
}
