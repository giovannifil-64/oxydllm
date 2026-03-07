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
        // Disable Qwen3-style thinking mode by default. Models like Qwen3 add
        // a <think>…</think> block to every response when this is true, which
        // results in the raw reasoning tokens being sent to the caller.
        enable_thinking: false,
    };

    let rendered = tmpl
        .render(&ctx)
        .context("Failed to render chat template")?;

    Ok(rendered)
}

/// Remove `<think>…</think>` blocks from a model response.
///
/// Qwen3 and similar reasoning models can emit an internal chain-of-thought
/// block even when `enable_thinking` is set to `false` in the chat template
/// (e.g. when the template ignores that variable, or when the model was
/// fine-tuned to always think).  This helper strips those blocks so that
/// callers always receive the final answer only.
pub fn strip_thinking_content(content: &str) -> String {
    let mut result = String::new();
    let mut remainder = content;
    while let Some(start) = remainder.find("<think>") {
        result.push_str(&remainder[..start]);
        remainder = &remainder[start + "<think>".len()..];
        if let Some(end) = remainder.find("</think>") {
            remainder = &remainder[end + "</think>".len()..];
        } else {
            // Unclosed <think> block – discard everything that follows.
            remainder = "";
        }
    }
    result.push_str(remainder);
    // Only strip leading whitespace that was adjacent to a removed <think> block,
    // not trailing whitespace which may be intentional in the model's response.
    result.trim_start_matches('\n').to_string()
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

/// Stateful filter for streaming token responses that removes `<think>…</think>` blocks.
///
/// Feed each decoded token text through [`ThinkingStreamFilter::push`]; it returns the
/// portion of the text that should be forwarded to the client (empty when the token
/// falls entirely inside a thinking block).
pub struct ThinkingStreamFilter {
    /// Text buffered while scanning for an opening or closing tag boundary.
    buffer: String,
    /// Whether we are currently inside a `<think>…</think>` block.
    in_think: bool,
}

impl ThinkingStreamFilter {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            in_think: false,
        }
    }

    /// Accept the next decoded token and return whatever should be emitted to the client.
    /// Returns an empty string when the token is inside a thinking block.
    pub fn push(&mut self, text: &str) -> String {
        self.buffer.push_str(text);
        self.drain_ready()
    }

    /// Flush any remaining buffered text at end-of-generation.
    ///
    /// Text that is still inside an unclosed `<think>` block is discarded.
    pub fn finish(&mut self) -> String {
        if self.in_think {
            self.buffer.clear();
            String::new()
        } else {
            std::mem::take(&mut self.buffer)
        }
    }

    /// Drain text that can be safely emitted without risking cutting a tag in half.
    fn drain_ready(&mut self) -> String {
        const OPEN: &str = "<think>";
        const CLOSE: &str = "</think>";
        let mut output = String::new();

        loop {
            if self.in_think {
                match self.buffer.find(CLOSE) {
                    Some(pos) => {
                        // Discard everything up to and including </think>.
                        self.buffer.drain(..pos + CLOSE.len());
                        self.in_think = false;
                        // Continue scanning for a possible second <think> block.
                    }
                    None => {
                        // Keep buffering – the closing tag may straddle the next chunk.
                        // Only retain the last (CLOSE.len() - 1) chars.
                        let keep = CLOSE.len().saturating_sub(1);
                        if self.buffer.len() > keep {
                            self.buffer.drain(..self.buffer.len() - keep);
                        }
                        break;
                    }
                }
            } else {
                match self.buffer.find(OPEN) {
                    Some(pos) => {
                        // Emit text before <think>, then enter thinking mode.
                        output.push_str(&self.buffer[..pos]);
                        self.buffer.drain(..pos + OPEN.len());
                        self.in_think = true;
                        // Continue scanning for </think>.
                    }
                    None => {
                        // Emit everything except the last (OPEN.len() - 1) chars
                        // in case the opening tag straddles the next chunk.
                        let keep = OPEN.len().saturating_sub(1);
                        if self.buffer.len() > keep {
                            let safe = self.buffer.len() - keep;
                            output.push_str(&self.buffer[..safe]);
                            self.buffer.drain(..safe);
                        }
                        break;
                    }
                }
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── strip_thinking_content ──────────────────────────────────────────────

    #[test]
    fn strip_thinking_no_block() {
        assert_eq!(strip_thinking_content("Hello world"), "Hello world");
    }

    #[test]
    fn strip_thinking_full_block() {
        let input = "<think>some internal reasoning</think>The answer is 42.";
        assert_eq!(strip_thinking_content(input), "The answer is 42.");
    }

    #[test]
    fn strip_thinking_block_with_newlines() {
        let input = "<think>\nI need to think.\nStep 1: ...\n</think>\n\nFinal answer.";
        assert_eq!(strip_thinking_content(input), "Final answer.");
    }

    #[test]
    fn strip_thinking_multiple_blocks() {
        let input = "<think>first</think>middle<think>second</think>end";
        assert_eq!(strip_thinking_content(input), "middleend");
    }

    #[test]
    fn strip_thinking_unclosed_block() {
        // An unclosed <think> block should discard everything after the opening tag.
        let input = "before<think>reasoning without closing tag";
        assert_eq!(strip_thinking_content(input), "before");
    }

    #[test]
    fn strip_thinking_only_block() {
        let input = "<think>only thinking, no answer</think>";
        assert_eq!(strip_thinking_content(input), "");
    }

    // ── ThinkingStreamFilter ────────────────────────────────────────────────

    #[test]
    fn stream_filter_passthrough() {
        let mut f = ThinkingStreamFilter::new();
        // Text with no thinking blocks should pass through unchanged (finish() flushes buffer).
        let o1 = f.push("Hello, world!");
        let o2 = f.finish();
        assert_eq!(o1 + &o2, "Hello, world!");
    }

    #[test]
    fn stream_filter_full_block_single_chunk() {
        let mut f = ThinkingStreamFilter::new();
        let o1 = f.push("<think>reasoning</think>Answer");
        let o2 = f.finish();
        assert_eq!(o1 + &o2, "Answer");
    }

    #[test]
    fn stream_filter_block_split_across_chunks() {
        let mut f = ThinkingStreamFilter::new();
        // Opening tag arrives split across two chunks.
        let o1 = f.push("<thi");
        let o2 = f.push("nk>internal</think>Result");
        let o3 = f.finish();
        // Nothing should be emitted until we know whether <think> is coming.
        assert_eq!(o1 + &o2 + &o3, "Result");
    }

    #[test]
    fn stream_filter_finish_flushes_normal_text() {
        let mut f = ThinkingStreamFilter::new();
        // Short input – not enough to emit with the keep-back heuristic.
        f.push("Hi");
        // finish() should return the buffered text.
        assert_eq!(f.finish(), "Hi");
    }

    #[test]
    fn stream_filter_unclosed_think_discarded_on_finish() {
        let mut f = ThinkingStreamFilter::new();
        f.push("<think>incomplete reasoning");
        assert_eq!(f.finish(), "");
    }

    #[test]
    fn stream_filter_multiple_chunks_normal() {
        let mut f = ThinkingStreamFilter::new();
        let o1 = f.push("Part 1. ");
        let o2 = f.push("<think>skip</think>");
        let o3 = f.push("Part 2.");
        let o4 = f.finish();
        assert_eq!(o1 + &o2 + &o3 + &o4, "Part 1. Part 2.");
    }
}
