//! Incremental decoding of a growing token sequence into text deltas.
//!
//! The server's streaming loop and the terminal chat both turn the tokens of
//! a sequence into text as they arrive, and they share this decoder so the
//! same tokens show the same text in both: one channel's state lives in an
//! [`IncrementalDecoder`], fed the whole sequence so far on every token.

use crate::tokenizer::Tokenizer;

/// Clamps `idx` down to the nearest UTF-8 character boundary in `s`.
pub fn clamp_to_char_boundary(s: &str, idx: usize) -> usize {
    let mut i = idx.min(s.len());
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Emits the not-yet-sent suffix of `full`, advancing `decoded_len`.
///
/// A trailing `U+FFFD` is held back so multi-byte UTF-8 split across tokens
/// is buffered until the continuation token arrives.
fn emit_suffix(full: &str, decoded_len: &mut usize) -> Option<String> {
    let start = clamp_to_char_boundary(full, *decoded_len);
    let new_text = &full[start..];
    let emit = new_text.trim_end_matches('\u{FFFD}');
    if !emit.is_empty() {
        *decoded_len = start + emit.len();
        Some(emit.to_string())
    } else {
        *decoded_len = start;
        None
    }
}

/// Number of trailing tokens re-decoded every step; anything older is frozen
/// into the stable prefix because the decoder can no longer change its text.
const DECODE_STABLE_LOOKBACK: usize = 4;

/// Decodes `ids` as a mid-sequence fragment by prefixing a neutral token and
/// stripping its text, preserving the leading space that a `Strip`/BPE
/// decoder would otherwise drop from a standalone decode.
fn decode_with_neutral_prefix(
    tokenizer: &Tokenizer,
    ids: &[u32],
    neutral_id: u32,
    neutral_text: &str,
) -> String {
    if ids.is_empty() {
        return String::new();
    }
    let mut prefixed = Vec::with_capacity(ids.len() + 1);
    prefixed.push(neutral_id);
    prefixed.extend_from_slice(ids);
    let with_prefix = tokenizer.decode(&prefixed).unwrap_or_default();
    with_prefix
        .strip_prefix(neutral_text)
        .map(|s| s.to_string())
        .unwrap_or_else(|| tokenizer.decode(ids).unwrap_or_default())
}

/// Tokens the stable boundary may back off to keep a split character whole:
/// a UTF-8 character is at most four bytes, so at most three of its tokens
/// can precede the boundary.
const SPLIT_CHAR_BACKOFF: usize = 3;

/// Freezes tokens older than [`DECODE_STABLE_LOOKBACK`] into `stable_text`,
/// appending their decoded delta so each step re-decodes only the tail.
///
/// The boundary never lands inside a character: frozen text is final, and a
/// multi-byte character split across tokens decodes to U+FFFD on either side
/// of a boundary through it, which also changes the frozen length under the
/// text already emitted. When the delta would end in U+FFFD the boundary
/// backs off, at most [`SPLIT_CHAR_BACKOFF`] tokens, to the nearest end that
/// decodes cleanly; bytes that never form a character are frozen as U+FFFD
/// once the back-off is exhausted, so a run of them cannot stall the window.
fn advance_stable_window(
    tokenizer: &Tokenizer,
    all_ids: &[u32],
    stable_end: &mut usize,
    stable_text: &mut String,
    neutral_id: u32,
    neutral_text: &str,
) {
    let target = all_ids.len().saturating_sub(DECODE_STABLE_LOOKBACK);
    if target <= *stable_end {
        return;
    }
    let decode_delta = |end: usize| {
        if *stable_end == 0 {
            tokenizer.decode(&all_ids[..end]).unwrap_or_default()
        } else {
            decode_with_neutral_prefix(
                tokenizer,
                &all_ids[*stable_end..end],
                neutral_id,
                neutral_text,
            )
        }
    };
    let mut end = target;
    let mut delta = decode_delta(end);
    while delta.ends_with('\u{FFFD}') && end > *stable_end + 1 && end + SPLIT_CHAR_BACKOFF > target
    {
        end -= 1;
        delta = decode_delta(end);
    }
    if delta.ends_with('\u{FFFD}') {
        if end + SPLIT_CHAR_BACKOFF > target {
            return;
        }
        end = target;
        delta = decode_delta(end);
    }
    if *stable_end == 0 {
        *stable_text = delta;
    } else {
        stable_text.push_str(&delta);
    }
    *stable_end = end;
}

/// A neutral token a mid-sequence fragment is decoded behind, so a `Strip`
/// or BPE decoder keeps the leading space it would drop from a standalone
/// decode; its own text is stripped from the result.
pub struct NeutralPrefix {
    id: u32,
    text: String,
}

impl NeutralPrefix {
    pub fn new(tokenizer: &Tokenizer) -> Self {
        let id = tokenizer
            .encode("a")
            .ok()
            .and_then(|ids| ids.into_iter().next())
            .unwrap_or(1);
        let text = tokenizer.decode(&[id]).unwrap_or_default();
        Self { id, text }
    }
}

/// The decoder state of one text channel: how much of the decoded text has
/// been handed out, and the frozen prefix the tail is decoded behind.
///
/// Decoding one token at a time is wrong for tokenizers with a `Strip`
/// decoder (each token loses its leading space), and re-decoding the whole
/// sequence every token is quadratic. The stable prefix (everything older
/// than the lookback window) is decoded once and cached; each step re-decodes
/// only the tail window and emits what lies past `decoded_len`.
#[derive(Default)]
pub struct IncrementalDecoder {
    decoded_len: usize,
    stable_end: usize,
    stable_text: String,
}

impl IncrementalDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Decodes `all_ids`, the channel's whole sequence so far, and returns
    /// the text not yet handed out, if any; a trailing incomplete character
    /// is held back until the token completing it arrives.
    pub fn push(
        &mut self,
        tokenizer: &Tokenizer,
        all_ids: &[u32],
        neutral: &NeutralPrefix,
    ) -> Option<String> {
        advance_stable_window(
            tokenizer,
            all_ids,
            &mut self.stable_end,
            &mut self.stable_text,
            neutral.id,
            &neutral.text,
        );
        let window_ids = &all_ids[self.stable_end..];
        let window_text = if self.stable_end == 0 {
            tokenizer.decode(window_ids).unwrap_or_default()
        } else {
            decode_with_neutral_prefix(tokenizer, window_ids, neutral.id, &neutral.text)
        };
        let full = format!("{}{}", self.stable_text, window_text);
        emit_suffix(&full, &mut self.decoded_len)
    }

    /// Hands out whatever a final decode of `all_ids` holds past what was
    /// emitted, the held-back incomplete character included, once the
    /// sequence is complete.
    pub fn finish(&mut self, tokenizer: &Tokenizer, all_ids: &[u32]) -> Option<String> {
        if all_ids.is_empty() {
            return None;
        }
        let full = tokenizer.decode(all_ids).unwrap_or_default();
        let start = clamp_to_char_boundary(&full, self.decoded_len);
        self.decoded_len = full.len();
        (start < full.len()).then(|| full[start..].to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_stream(steps: &[&str]) -> String {
        let mut decoded_len = 0;
        let mut out = String::new();
        for full in steps {
            if let Some(s) = emit_suffix(full, &mut decoded_len) {
                out.push_str(&s);
            }
        }
        out
    }

    #[test]
    fn cumulative_emission_matches_final_canonical_text() {
        let steps = [
            "Sono",
            "Sono un",
            "Sono un assistente",
            "Sono un assistente virtuale",
        ];
        assert_eq!(run_stream(&steps), *steps.last().unwrap());
    }

    #[test]
    fn space_only_token_followed_by_g_prefixed_token_does_not_double_space() {
        // Regression: Ġ-only space token followed by a Ġ-prefixed token must
        // not emit a double space.
        let steps = [
            "Sono un assistente",
            "Sono un assistente ",
            "Sono un assistente virtuale",
        ];
        assert_eq!(run_stream(&steps), "Sono un assistente virtuale");
    }

    #[test]
    fn no_character_loss_across_partial_utf8_token_boundary() {
        let steps = ["caf\u{FFFD}", "café", "café 🎉"];
        assert_eq!(run_stream(&steps), "café 🎉");
    }

    /// A byte-level BPE tokenizer with no merges, so every byte is one token
    /// and any multi-byte character is split across tokens, which is how a
    /// real byte-level or byte-fallback vocabulary emits rare characters.
    fn byte_per_token_tokenizer() -> (Tokenizer, tempfile::TempDir) {
        use tokenizers::models::bpe::{BPE, Vocab};
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;

        let mut alphabet: Vec<char> = ByteLevel::alphabet().into_iter().collect();
        alphabet.sort_unstable();
        let vocab: Vocab = alphabet
            .iter()
            .enumerate()
            .map(|(i, c)| (c.to_string(), i as u32))
            .collect();
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, Vec::new())
            .build()
            .unwrap_or_else(|e| panic!("build bpe: {e}"));
        let mut inner = tokenizers::Tokenizer::new(bpe);
        inner.with_pre_tokenizer(Some(ByteLevel::new(false, false, false)));
        inner.with_decoder(Some(ByteLevel::new(false, false, false)));
        let tmp = tempfile::tempdir().expect("tempdir");
        inner
            .save(tmp.path().join("tokenizer.json"), false)
            .unwrap_or_else(|e| panic!("save tokenizer: {e}"));
        std::fs::write(tmp.path().join("tokenizer_config.json"), "{}").expect("config");
        let tok =
            Tokenizer::from_dir(tmp.path().to_str().expect("utf-8 path")).expect("load tokenizer");
        (tok, tmp)
    }

    /// Streams `ids` one token at a time through the incremental decoder the
    /// engine loop uses and returns everything it emitted.
    fn stream_incrementally(tok: &Tokenizer, ids: &[u32]) -> String {
        let neutral = NeutralPrefix::new(tok);
        let mut decoder = IncrementalDecoder::new();
        let mut out = String::new();
        for n in 1..=ids.len() {
            if let Some(text) = decoder.push(tok, &ids[..n], &neutral) {
                out.push_str(&text);
            }
        }
        if let Some(text) = decoder.finish(tok, ids) {
            out.push_str(&text);
        }
        out
    }

    /// Contract: the streamed text equals the decode of the whole sequence
    /// whatever token boundaries fall inside a multi-byte character.
    ///
    /// The stable window froze decoded text one token at a time, so a
    /// character split across tokens was frozen as U+FFFD, and the frozen
    /// text then differed in length from what had already been emitted: the
    /// suffix came out duplicated or truncated around every such character.
    /// Byte-level BPE vocabularies split emoji and CJK this way routinely.
    #[test]
    fn split_characters_stream_as_the_final_decode() {
        let (tok, _tmp) = byte_per_token_tokenizer();
        for text in [
            "caf\u{e9} au lait, s'il vous pla\u{ee}t",
            "price: 12\u{20ac} or 10\u{a3}, ok?",
            "\u{1f389}\u{1f389} party \u{1f600} time \u{1f389}",
            "\u{4eca}\u{65e5}\u{306f}\u{4e16}\u{754c} hello \u{4e16}\u{754c}",
            "mixed \u{e9}\u{1f600}\u{4e16} end",
        ] {
            let ids = tok.encode(text).unwrap();
            assert!(
                ids.len() > text.chars().count(),
                "characters must be split across tokens"
            );
            let streamed = stream_incrementally(&tok, &ids);
            assert_eq!(streamed, tok.decode(&ids).unwrap(), "streaming {text:?}");
            assert_eq!(streamed, text);
        }
    }

    /// Contract: bytes that never form a character are rendered once, as
    /// U+FFFD, and do not stall the stream or shift what follows them.
    #[test]
    fn invalid_bytes_do_not_stall_or_duplicate_the_stream() {
        let (tok, _tmp) = byte_per_token_tokenizer();
        let good = tok.encode("ab").unwrap();
        let lone_continuation = tok.encode("\u{e9}").unwrap()[1];
        let tail = tok.encode("cdefghij").unwrap();
        let ids: Vec<u32> = good
            .iter()
            .chain([lone_continuation].iter())
            .chain(tail.iter())
            .copied()
            .collect();
        let streamed = stream_incrementally(&tok, &ids);
        let full = tok.decode(&ids).unwrap();
        assert_eq!(full, "ab\u{fffd}cdefghij");
        assert_eq!(streamed, full);
    }

    #[test]
    fn empty_step_emits_nothing() {
        let steps = ["hello", "hello", "hello world"];
        assert_eq!(run_stream(&steps), "hello world");
    }

    #[test]
    fn idempotent_when_decoded_len_matches_full_len() {
        let mut decoded_len = 5;
        let out = emit_suffix("hello", &mut decoded_len);
        assert!(out.is_none());
        assert_eq!(decoded_len, 5);
    }

    #[test]
    fn clamps_decoded_len_past_end_of_full() {
        let mut decoded_len = 100;
        let out = emit_suffix("short", &mut decoded_len);
        assert!(out.is_none());
        assert_eq!(decoded_len, 5);
    }

    #[test]
    fn clamps_decoded_len_to_char_boundary_inside_multibyte() {
        // decoded_len starts inside é (2-byte UTF-8); must back off to boundary.
        let mut decoded_len = 4;
        let out = emit_suffix("café!", &mut decoded_len);
        assert_eq!(out.as_deref(), Some("é!"));
        assert_eq!(decoded_len, 6);
    }
}
