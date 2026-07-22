//! Grammar-constrained decoding: JSON guaranteed valid by construction.
//!
//! When a request asks for JSON output, the sampler masks every token whose
//! bytes would take an incremental JSON automaton into an invalid state, and
//! allows the stop tokens only once the automaton accepts a complete value.
//! The automaton state is [`Copy`] and a few dozen bytes (nesting kept as a
//! bitstack), so probing all ~150k vocabulary tokens per step stays in the
//! microsecond class after the one-time [`TokenByteTable`] build.
//!
//! Scope: JSON syntax (parseability). Schema guidance (keys, types, enums)
//! layers on top separately; schema validation of the finished output remains
//! in the chat handler either way.

use std::collections::HashMap;
use std::sync::Arc;

/// Maximum object/array nesting the constraint permits. At the cap, opening
/// brackets are masked out, forcing the model to close instead of recursing
/// forever; the bitstack backing is a `u128`.
const MAX_DEPTH: u8 = 64;

/// What the top-level JSON value may be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JsonMode {
    /// The top-level value must be an object (OpenAI `json_object` semantics).
    Object,
    /// Any JSON value (used for `json_schema` roots that are not objects).
    Value,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum NumS {
    /// After a leading minus sign: a digit must follow.
    Minus,
    /// After a leading zero: only `.`, `e`, or a delimiter may follow.
    Zero,
    /// In the integer part (last byte was a nonzero-led digit).
    Int,
    /// Right after the decimal point: a digit must follow.
    Dot,
    /// In the fractional part.
    Frac,
    /// Right after `e`/`E` (optionally a sign next): a digit must follow.
    Exp,
    /// After `e`/`E` and a sign: a digit must follow.
    ExpSign,
    /// In the exponent digits.
    ExpDigits,
}

impl NumS {
    fn complete(self) -> bool {
        matches!(self, Self::Zero | Self::Int | Self::Frac | Self::ExpDigits)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum S {
    /// Expecting a value (whitespace allowed).
    Value,
    /// After `{`: expecting a key string or `}`.
    ObjFirst,
    /// After `[`: expecting a value or `]`.
    ArrFirst,
    /// After `,` inside an object: expecting a key string.
    ObjComma,
    /// Inside a string; `key` strings expect `:` after closing.
    Str { esc: bool, uni: u8, key: bool },
    /// After a key string closed: expecting `:`.
    Colon,
    /// Inside a number.
    Num(NumS),
    /// Inside `true`(0) / `false`(1) / `null`(2), at byte `pos`.
    Lit { lit: u8, pos: u8 },
    /// After a value inside an object: expecting `,` or `}`.
    ObjNext,
    /// After a value inside an array: expecting `,` or `]`.
    ArrNext,
    /// Top-level value complete: only trailing whitespace.
    Done,
}

/// The incremental JSON syntax automaton. `Copy` by design: the per-step mask
/// probes every vocabulary token from the same start state.
#[derive(Clone, Copy, Debug)]
pub struct JsonMachine {
    /// One bit per nesting level: 0 = object, 1 = array.
    stack: u128,
    depth: u8,
    state: S,
}

const LITERALS: [&[u8]; 3] = [b"true", b"false", b"null"];

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

impl JsonMachine {
    pub fn new(mode: JsonMode) -> Self {
        Self {
            stack: 0,
            depth: 0,
            state: match mode {
                // A leading `{` is forced by only accepting object-start
                // bytes in ObjTop; reuse Value with a post-filter instead of
                // a dedicated state: see `step`.
                JsonMode::Object => S::Value,
                JsonMode::Value => S::Value,
            },
        }
    }

    fn object_only(mode: JsonMode) -> bool {
        mode == JsonMode::Object
    }

    /// Whether the machine has consumed one complete top-level value.
    pub fn accepting(&self) -> bool {
        self.state == S::Done || (self.depth == 0 && self.number_completes_top())
    }

    fn number_completes_top(&self) -> bool {
        matches!(self.state, S::Num(n) if n.complete())
    }

    fn push(&mut self, is_array: bool) -> bool {
        if self.depth >= MAX_DEPTH {
            return false;
        }
        if is_array {
            self.stack |= 1u128 << self.depth;
        } else {
            self.stack &= !(1u128 << self.depth);
        }
        self.depth += 1;
        true
    }

    /// The state that follows a completed value at the current depth.
    fn after_value(&self) -> S {
        if self.depth == 0 {
            S::Done
        } else if (self.stack >> (self.depth - 1)) & 1 == 1 {
            S::ArrNext
        } else {
            S::ObjNext
        }
    }

    fn pop(&mut self) -> Option<S> {
        if self.depth == 0 {
            return None;
        }
        self.depth -= 1;
        Some(self.after_value())
    }

    fn start_value(&mut self, b: u8, top_object_only: bool) -> bool {
        if top_object_only && self.depth == 0 && b != b'{' && !is_ws(b) {
            return false;
        }
        match b {
            _ if is_ws(b) => true,
            b'{' => {
                if !self.push(false) {
                    return false;
                }
                self.state = S::ObjFirst;
                true
            }
            b'[' => {
                if !self.push(true) {
                    return false;
                }
                self.state = S::ArrFirst;
                true
            }
            b'"' => {
                self.state = S::Str {
                    esc: false,
                    uni: 0,
                    key: false,
                };
                true
            }
            b'-' => {
                self.state = S::Num(NumS::Minus);
                true
            }
            b'0' => {
                self.state = S::Num(NumS::Zero);
                true
            }
            b'1'..=b'9' => {
                self.state = S::Num(NumS::Int);
                true
            }
            b't' => {
                self.state = S::Lit { lit: 0, pos: 1 };
                true
            }
            b'f' => {
                self.state = S::Lit { lit: 1, pos: 1 };
                true
            }
            b'n' => {
                self.state = S::Lit { lit: 2, pos: 1 };
                true
            }
            _ => false,
        }
    }

    /// Advances by one byte; `false` means the byte is illegal here.
    pub fn step(&mut self, b: u8, mode: JsonMode) -> bool {
        let top_only = Self::object_only(mode);
        match self.state {
            S::Value => self.start_value(b, top_only),
            S::ObjFirst => match b {
                _ if is_ws(b) => true,
                b'"' => {
                    self.state = S::Str {
                        esc: false,
                        uni: 0,
                        key: true,
                    };
                    true
                }
                b'}' => match self.pop() {
                    Some(next) => {
                        self.state = next;
                        true
                    }
                    None => false,
                },
                _ => false,
            },
            S::ArrFirst => {
                if b == b']' {
                    return match self.pop() {
                        Some(next) => {
                            self.state = next;
                            true
                        }
                        None => false,
                    };
                }
                self.state = S::Value;
                self.start_value(b, top_only)
            }
            S::ObjComma => match b {
                _ if is_ws(b) => true,
                b'"' => {
                    self.state = S::Str {
                        esc: false,
                        uni: 0,
                        key: true,
                    };
                    true
                }
                _ => false,
            },
            S::Str { esc, uni, key } => {
                if uni > 0 {
                    if b.is_ascii_hexdigit() {
                        self.state = S::Str {
                            esc: false,
                            uni: uni - 1,
                            key,
                        };
                        true
                    } else {
                        false
                    }
                } else if esc {
                    match b {
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                            self.state = S::Str {
                                esc: false,
                                uni: 0,
                                key,
                            };
                            true
                        }
                        b'u' => {
                            self.state = S::Str {
                                esc: false,
                                uni: 4,
                                key,
                            };
                            true
                        }
                        _ => false,
                    }
                } else {
                    match b {
                        b'"' => {
                            self.state = if key { S::Colon } else { self.after_value() };
                            true
                        }
                        b'\\' => {
                            self.state = S::Str {
                                esc: true,
                                uni: 0,
                                key,
                            };
                            true
                        }
                        0x00..=0x1F => false,
                        _ => true,
                    }
                }
            }
            S::Colon => match b {
                _ if is_ws(b) => true,
                b':' => {
                    self.state = S::Value;
                    true
                }
                _ => false,
            },
            S::Num(n) => {
                let next = match (n, b) {
                    (NumS::Minus, b'0') => Some(NumS::Zero),
                    (NumS::Minus, b'1'..=b'9') => Some(NumS::Int),
                    (NumS::Zero, b'.') => Some(NumS::Dot),
                    (NumS::Zero, b'e' | b'E') => Some(NumS::Exp),
                    (NumS::Int, b'0'..=b'9') => Some(NumS::Int),
                    (NumS::Int, b'.') => Some(NumS::Dot),
                    (NumS::Int, b'e' | b'E') => Some(NumS::Exp),
                    (NumS::Dot | NumS::Frac, b'0'..=b'9') => Some(NumS::Frac),
                    (NumS::Frac, b'e' | b'E') => Some(NumS::Exp),
                    (NumS::Exp, b'+' | b'-') => Some(NumS::ExpSign),
                    (NumS::Exp | NumS::ExpSign | NumS::ExpDigits, b'0'..=b'9') => {
                        Some(NumS::ExpDigits)
                    }
                    _ => None,
                };
                if let Some(n2) = next {
                    self.state = S::Num(n2);
                    return true;
                }
                if n.complete() {
                    self.state = self.after_value();
                    return self.step(b, mode);
                }
                false
            }
            S::Lit { lit, pos } => {
                let word = LITERALS[lit as usize];
                if (pos as usize) < word.len() && b == word[pos as usize] {
                    if pos as usize + 1 == word.len() {
                        self.state = self.after_value();
                    } else {
                        self.state = S::Lit { lit, pos: pos + 1 };
                    }
                    true
                } else {
                    false
                }
            }
            S::ObjNext => match b {
                _ if is_ws(b) => true,
                b',' => {
                    self.state = S::ObjComma;
                    true
                }
                b'}' => match self.pop() {
                    Some(next) => {
                        self.state = next;
                        true
                    }
                    None => false,
                },
                _ => false,
            },
            S::ArrNext => match b {
                _ if is_ws(b) => true,
                b',' => {
                    self.state = S::Value;
                    true
                }
                b']' => match self.pop() {
                    Some(next) => {
                        self.state = next;
                        true
                    }
                    None => false,
                },
                _ => false,
            },
            S::Done => is_ws(b),
        }
    }
}

/// Every vocabulary token's produced bytes, or `None` for tokens that must
/// never appear inside constrained output (added/special tokens).
pub struct TokenByteTable {
    bytes: Vec<Option<Vec<u8>>>,
    stop_ids: Vec<u32>,
}

impl TokenByteTable {
    /// Builds the table from a byte-level BPE tokenizer via the GPT-2
    /// unicode-to-byte alphabet. Returns `None` when the vocabulary does not
    /// look byte-level (e.g. SentencePiece), in which case constrained
    /// decoding is unavailable and callers fall back to post-hoc validation.
    pub fn build(tokenizer: &crate::tokenizer::Tokenizer) -> Option<Arc<Self>> {
        let char_to_byte = gpt2_char_to_byte();
        let vocab_size = tokenizer.vocab_size();
        let special: std::collections::HashSet<u32> =
            tokenizer.special_token_id_values().into_iter().collect();
        let mut bytes: Vec<Option<Vec<u8>>> = Vec::with_capacity(vocab_size);
        let mut mapped = 0usize;
        for id in 0..vocab_size as u32 {
            if special.contains(&id) {
                bytes.push(None);
                continue;
            }
            let Some(tok) = tokenizer.id_to_token(id) else {
                bytes.push(None);
                continue;
            };
            let mut out = Vec::with_capacity(tok.len());
            let mut ok = true;
            for ch in tok.chars() {
                match char_to_byte.get(&ch) {
                    Some(&b) => out.push(b),
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                mapped += 1;
                bytes.push(Some(out));
            } else {
                bytes.push(None);
            }
        }
        if (mapped as f64) < vocab_size as f64 * 0.9 {
            return None;
        }
        Some(Arc::new(Self {
            bytes,
            stop_ids: tokenizer.stop_token_ids(),
        }))
    }

    #[cfg(test)]
    fn from_raw(bytes: Vec<Option<Vec<u8>>>, stop_ids: Vec<u32>) -> Arc<Self> {
        Arc::new(Self { bytes, stop_ids })
    }
}

/// The per-sequence constraint: automaton state plus the shared byte table.
pub struct JsonConstraint {
    table: Arc<TokenByteTable>,
    machine: JsonMachine,
    mode: JsonMode,
}

impl JsonConstraint {
    pub fn new(table: Arc<TokenByteTable>, mode: JsonMode) -> Self {
        Self {
            table,
            machine: JsonMachine::new(mode),
            mode,
        }
    }

    /// Marks every token whose bytes are illegal from the current state as
    /// `false`; stop tokens are `true` only when the machine accepts.
    pub fn allowed(&self, vocab_size: usize) -> Vec<bool> {
        let accepting = self.machine.accepting();
        let mut allowed = vec![false; vocab_size];
        for (id, entry) in self.table.bytes.iter().enumerate().take(vocab_size) {
            let Some(tok_bytes) = entry else { continue };
            if tok_bytes.is_empty() {
                continue;
            }
            let mut m = self.machine;
            if tok_bytes.iter().all(|&b| m.step(b, self.mode)) {
                allowed[id] = true;
            }
        }
        if accepting {
            for &id in &self.table.stop_ids {
                if (id as usize) < vocab_size {
                    allowed[id as usize] = true;
                }
            }
        }
        allowed
    }

    /// Advances the automaton with the chosen token. Stop tokens (legal only
    /// in the accepting state) leave the machine unchanged.
    pub fn advance(&mut self, token: u32) {
        if self.table.stop_ids.contains(&token) {
            return;
        }
        if let Some(Some(tok_bytes)) = self.table.bytes.get(token as usize) {
            for &b in tok_bytes {
                if !self.machine.step(b, self.mode) {
                    return;
                }
            }
        }
    }
}

/// The GPT-2 byte-level BPE alphabet: printable-unicode char back to the raw
/// byte it encodes.
fn gpt2_char_to_byte() -> HashMap<char, u8> {
    let mut byte_to_char: Vec<(u8, char)> = Vec::with_capacity(256);
    let printable = (b'!'..=b'~')
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect::<Vec<u8>>();
    for &b in &printable {
        byte_to_char.push((b, char::from_u32(b as u32).unwrap()));
    }
    let mut n = 0u32;
    for b in 0..=255u8 {
        if !printable.contains(&b) {
            byte_to_char.push((b, char::from_u32(256 + n).unwrap()));
            n += 1;
        }
    }
    byte_to_char.into_iter().map(|(b, c)| (c, b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(machine: &mut JsonMachine, s: &str, mode: JsonMode) -> bool {
        s.bytes().all(|b| machine.step(b, mode))
    }

    /// Contract: every syntactically valid JSON document walks the automaton
    /// byte by byte into the accepting state, in both modes.
    #[test]
    fn valid_json_accepts() {
        let docs = [
            r#"{}"#,
            r#"{ "a": 1 }"#,
            r#"{"a":{"b":[1,2.5,-3e+7,0.1E-2]},"c":"x \"y\" é \\ /","d":[[],{}],"e":true,"f":false,"g":null}"#,
            "{\n  \"pretty\": [1, 2, 3]\n}",
        ];
        for doc in docs {
            for mode in [JsonMode::Object, JsonMode::Value] {
                let mut m = JsonMachine::new(mode);
                assert!(run(&mut m, doc, mode), "rejected valid {doc:?} in {mode:?}");
                assert!(m.accepting(), "not accepting after {doc:?} in {mode:?}");
            }
        }
        for doc in [r#""bare string""#, "42", "-0.5e3", "true", "[1,2]"] {
            let mode = JsonMode::Value;
            let mut m = JsonMachine::new(mode);
            assert!(run(&mut m, doc, mode), "rejected valid {doc:?}");
            assert!(m.accepting(), "not accepting after {doc:?}");
        }
    }

    /// Contract: malformed JSON is rejected at the first illegal byte, and
    /// non-object roots are rejected in Object mode.
    #[test]
    fn invalid_json_rejects() {
        let bad = [
            r#"{"a":}"#,
            r#"{"a":1,}"#,
            r#"{,}"#,
            r#"{"a" 1}"#,
            r#"[1,]"#,
            r#"{"a":01}"#,
            r#"{"a":1..2}"#,
            r#"{"a":+1}"#,
            r#"{"a":tru}"#,
            r#"{"a":"unterminated \x"}"#,
            "{\"a\":\"ctrl\x01\"}",
            r#"}{"#,
        ];
        for doc in bad {
            let mut m = JsonMachine::new(JsonMode::Object);
            let ok = run(&mut m, doc, JsonMode::Object) && m.accepting();
            assert!(!ok, "accepted invalid {doc:?}");
        }
        let mut m = JsonMachine::new(JsonMode::Object);
        assert!(
            !run(&mut m, "[1]", JsonMode::Object),
            "object mode must reject array root"
        );
    }

    /// Contract: after a complete document, only whitespace may follow.
    #[test]
    fn trailing_garbage_rejects() {
        let mode = JsonMode::Object;
        let mut m = JsonMachine::new(mode);
        assert!(run(&mut m, r#"{"a":1}"#, mode));
        assert!(m.accepting());
        assert!(m.step(b' ', mode));
        assert!(!m.step(b'{', mode));
    }

    /// Contract: nesting beyond [`MAX_DEPTH`] is refused so the constraint
    /// cannot be driven into unbounded recursion.
    #[test]
    fn depth_cap_holds() {
        let mode = JsonMode::Value;
        let mut m = JsonMachine::new(mode);
        for i in 0..MAX_DEPTH {
            assert!(m.step(b'[', mode), "open {i} rejected");
        }
        assert!(!m.step(b'[', mode), "depth cap not enforced");
        assert!(m.step(b'1', mode));
        for i in 0..MAX_DEPTH {
            assert!(m.step(b']', mode), "close {i} rejected");
        }
        assert!(m.accepting());
    }

    /// Contract: the mask allows exactly the tokens whose bytes fit the
    /// current state, and stop tokens only once accepting.
    #[test]
    fn mask_and_advance_agree() {
        let table = TokenByteTable::from_raw(
            vec![
                Some(b"{".to_vec()),      // 0
                Some(b"\"k\":".to_vec()), // 1
                Some(b"1".to_vec()),      // 2
                Some(b"}".to_vec()),      // 3
                Some(b"hello".to_vec()),  // 4
                None,                     // 5 special
            ],
            vec![5],
        );
        let mut c = JsonConstraint::new(table, JsonMode::Object);

        let a = c.allowed(6);
        assert_eq!(a, vec![true, false, false, false, false, false]);
        c.advance(0);
        let a = c.allowed(6);
        assert_eq!(a, vec![false, true, false, true, false, false]);
        c.advance(1);
        assert_eq!(c.allowed(6), vec![true, false, true, false, false, false]);
        c.advance(2);
        assert_eq!(
            c.allowed(6),
            vec![false, false, true, true, false, false],
            "after '1' another digit continues the number and '}}' closes"
        );
        c.advance(3);
        let a = c.allowed(6);
        assert_eq!(a, vec![false, false, false, false, false, true]);
    }
}
