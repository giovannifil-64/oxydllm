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

/// Structural events one byte can trigger, as a bitset: a single byte may
/// both finish a scalar and close a container (`5}` finishes the number and
/// closes the object).
pub type Events = u16;
pub const EV_VALUE_START: Events = 1 << 0;
pub const EV_SCALAR_END: Events = 1 << 1;
pub const EV_OBJ_OPEN: Events = 1 << 2;
pub const EV_OBJ_CLOSE: Events = 1 << 3;
pub const EV_ARR_OPEN: Events = 1 << 4;
pub const EV_ARR_CLOSE: Events = 1 << 5;
pub const EV_KEY_START: Events = 1 << 6;
pub const EV_KEY_END: Events = 1 << 7;
pub const EV_OBJ_COMMA: Events = 1 << 8;

impl JsonMachine {
    fn in_number(&self) -> bool {
        matches!(self.state, S::Num(_))
    }
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
        self.step_ev(b, mode).is_some()
    }

    /// Advances by one byte, reporting the structural [`Events`] it caused;
    /// `None` means the byte is illegal here.
    pub fn step_ev(&mut self, b: u8, mode: JsonMode) -> Option<Events> {
        let before_state = self.state;
        let before_depth = self.depth;
        if !self.step_inner(b, mode) {
            return None;
        }
        let mut ev: Events = 0;
        let value_started =
            matches!(before_state, S::Value | S::ArrFirst) && !is_ws(b) && b != b']';
        if value_started {
            ev |= EV_VALUE_START;
        }
        match b {
            b'{' if value_started => ev |= EV_OBJ_OPEN,
            b'[' if value_started => ev |= EV_ARR_OPEN,
            b'"' => {
                if matches!(before_state, S::ObjFirst | S::ObjComma) {
                    ev |= EV_KEY_START;
                } else if let S::Str {
                    esc: false,
                    uni: 0,
                    key,
                    ..
                } = before_state
                {
                    if key {
                        ev |= EV_KEY_END;
                    } else {
                        ev |= EV_SCALAR_END;
                    }
                }
            }
            b',' => {
                if matches!(before_state, S::ObjNext) {
                    ev |= EV_OBJ_COMMA;
                }
            }
            b'}' => {
                if matches!(before_state, S::ObjFirst | S::ObjNext | S::Num(_)) {
                    if matches!(before_state, S::Num(_)) {
                        ev |= EV_SCALAR_END;
                    }
                    ev |= EV_OBJ_CLOSE;
                }
            }
            b']' => {
                if matches!(before_state, S::ArrFirst | S::ArrNext | S::Num(_)) {
                    if matches!(before_state, S::Num(_)) {
                        ev |= EV_SCALAR_END;
                    }
                    ev |= EV_ARR_CLOSE;
                }
            }
            _ => {}
        }
        // A literal or number that just completed into a delimiter, or a
        // literal that hit its final letter.
        if matches!(before_state, S::Lit { .. }) && !matches!(self.state, S::Lit { .. }) {
            ev |= EV_SCALAR_END;
        }
        if matches!(before_state, S::Num(_))
            && !matches!(self.state, S::Num(_))
            && !matches!(b, b'}' | b']')
        {
            ev |= EV_SCALAR_END;
        }
        if before_depth < self.depth {
            // already flagged via byte match above
        }
        Some(ev)
    }

    fn step_inner(&mut self, b: u8, mode: JsonMode) -> bool {
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

/// A request's JSON constraint specification: the top-level shape plus the
/// compiled schema guidance (unconstrained for plain `json_object`).
#[derive(Clone)]
pub struct JsonSpec {
    pub mode: JsonMode,
    pub arena: Arc<SchemaArena>,
}

/// A node id into a [`SchemaArena`]; id 0 is always the unconstrained `Any`.
pub type NodeId = u16;
const ANY: NodeId = 0;
const NO_PROP: u16 = u16::MAX;
/// Schema guidance tracks at most this much nesting; deeper levels fall back
/// to syntax-only (sound: never stricter than the schema requires... looser).
const GUIDED_DEPTH: usize = 8;

/// A compiled schema node. Anything the compiler does not understand becomes
/// [`SchemaNode::Any`], which keeps the mask sound: unsupported subtrees are
/// guided by syntax only and judged by post-hoc validation.
pub enum SchemaNode {
    Any,
    Object {
        /// Property names as raw bytes with their value nodes.
        props: Vec<(Vec<u8>, NodeId)>,
        /// Bitmask over `props` of the required ones.
        required: u64,
        /// Whether unknown keys are allowed.
        additional: bool,
    },
    Array {
        items: NodeId,
    },
    Str,
    Number {
        integer: bool,
    },
    Boolean,
    Null,
    /// Scalar enum: the exact canonical JSON serializations allowed.
    Enum {
        literals: Vec<Vec<u8>>,
    },
}

/// The compiled schema, arena-allocated so states are plain indices.
pub struct SchemaArena {
    nodes: Vec<SchemaNode>,
}

impl SchemaArena {
    /// An arena whose root constrains nothing (plain JSON-syntax guidance).
    pub fn unconstrained() -> Arc<Self> {
        Arc::new(Self {
            nodes: vec![SchemaNode::Any],
        })
    }

    /// The root node of the compiled schema.
    pub fn root(&self) -> NodeId {
        (self.nodes.len() - 1) as NodeId
    }

    fn node(&self, id: NodeId) -> &SchemaNode {
        &self.nodes[id as usize]
    }

    /// Compiles a JSON Schema into guidance nodes; the root is
    /// [`root`](Self::root). Unsupported constructs compile to `Any`.
    pub fn compile(schema: &serde_json::Value) -> Arc<Self> {
        let mut arena = Self {
            nodes: vec![SchemaNode::Any],
        };
        let root = arena.compile_node(schema, schema, 0);
        if root != arena.nodes.len() as NodeId - 1 {
            // Root resolved to an existing node (e.g. Any): append an alias
            // is unnecessary because root() must be the last: push a copy.
            let clone_target = root;
            let node = match arena.node(clone_target) {
                SchemaNode::Any => SchemaNode::Any,
                _ => unreachable!("only Any is shared"),
            };
            arena.nodes.push(node);
        }
        Arc::new(arena)
    }

    fn compile_node(
        &mut self,
        schema: &serde_json::Value,
        root: &serde_json::Value,
        depth: u8,
    ) -> NodeId {
        if depth > 32 {
            return ANY;
        }
        let Some(obj) = schema.as_object() else {
            return ANY;
        };
        if let Some(r) = obj.get("$ref").and_then(|v| v.as_str()) {
            for prefix in ["#/$defs/", "#/definitions/"] {
                if let Some(name) = r.strip_prefix(prefix) {
                    let key = if prefix.contains("definitions") {
                        "definitions"
                    } else {
                        "$defs"
                    };
                    if let Some(target) = root.get(key).and_then(|d| d.get(name)) {
                        return self.compile_node(target, root, depth + 1);
                    }
                }
            }
            return ANY;
        }
        if let Some(values) = obj.get("enum").and_then(|v| v.as_array()) {
            let mut literals = Vec::with_capacity(values.len());
            if values.is_empty() || values.len() > 64 {
                return ANY;
            }
            for v in values {
                if v.is_object() || v.is_array() {
                    return ANY;
                }
                match serde_json::to_vec(v) {
                    Ok(bytes) => literals.push(bytes),
                    Err(_) => return ANY,
                }
            }
            self.nodes.push(SchemaNode::Enum { literals });
            return (self.nodes.len() - 1) as NodeId;
        }
        if obj.contains_key("anyOf") || obj.contains_key("oneOf") || obj.contains_key("allOf") {
            return ANY;
        }
        let ty = obj.get("type").and_then(|v| v.as_str());
        match ty {
            Some("object") | None if obj.contains_key("properties") || ty == Some("object") => {
                let mut props: Vec<(Vec<u8>, NodeId)> = Vec::new();
                if let Some(properties) = obj.get("properties").and_then(|v| v.as_object()) {
                    if properties.len() > 64 {
                        return ANY;
                    }
                    for (name, sub) in properties {
                        let node = self.compile_node(sub, root, depth + 1);
                        props.push((name.clone().into_bytes(), node));
                    }
                }
                let mut required = 0u64;
                if let Some(req) = obj.get("required").and_then(|v| v.as_array()) {
                    for r in req {
                        let Some(name) = r.as_str() else { return ANY };
                        match props.iter().position(|(p, _)| p == name.as_bytes()) {
                            Some(i) => required |= 1 << i,
                            None => return ANY,
                        }
                    }
                }
                let additional = obj
                    .get("additionalProperties")
                    .map(|v| v != &serde_json::Value::Bool(false))
                    .unwrap_or(true);
                self.nodes.push(SchemaNode::Object {
                    props,
                    required,
                    additional,
                });
                (self.nodes.len() - 1) as NodeId
            }
            Some("array") => {
                let items = obj
                    .get("items")
                    .map(|it| self.compile_node(it, root, depth + 1))
                    .unwrap_or(ANY);
                self.nodes.push(SchemaNode::Array { items });
                (self.nodes.len() - 1) as NodeId
            }
            Some("string") => {
                self.nodes.push(SchemaNode::Str);
                (self.nodes.len() - 1) as NodeId
            }
            Some("number") => {
                self.nodes.push(SchemaNode::Number { integer: false });
                (self.nodes.len() - 1) as NodeId
            }
            Some("integer") => {
                self.nodes.push(SchemaNode::Number { integer: true });
                (self.nodes.len() - 1) as NodeId
            }
            Some("boolean") => {
                self.nodes.push(SchemaNode::Boolean);
                (self.nodes.len() - 1) as NodeId
            }
            Some("null") => {
                self.nodes.push(SchemaNode::Null);
                (self.nodes.len() - 1) as NodeId
            }
            _ => ANY,
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

/// One guided nesting level: the governing schema node, which known
/// properties have been emitted, and the property whose value is pending.
#[derive(Clone, Copy)]
struct Level {
    node: NodeId,
    emitted: u64,
    cur_prop: u16,
}

/// The constraint on the scalar currently being emitted.
#[derive(Clone, Copy)]
enum ValCtx {
    Free,
    /// Matching enum literals byte-exactly; `alive` is a bitmask over the
    /// literals of the enum `node`.
    EnumScan {
        node: NodeId,
        alive: u64,
        pos: u16,
    },
    /// Matching an object key against the level's property names.
    KeyScan {
        alive: u64,
        pos: u16,
        any: bool,
    },
    /// Inside a number under an `integer` schema: `.` and exponents masked.
    IntNumber,
}

/// The schema-guidance half of the walker: `Copy`, so the per-token probe
/// costs one memcpy. Nesting deeper than [`GUIDED_DEPTH`] is counted in
/// `overflow` and guided by syntax only.
#[derive(Clone, Copy)]
struct GuideState {
    levels: [Level; GUIDED_DEPTH],
    depth: u8,
    overflow: u8,
    val: ValCtx,
}

impl GuideState {
    fn new() -> Self {
        Self {
            levels: [Level {
                node: ANY,
                emitted: 0,
                cur_prop: NO_PROP,
            }; GUIDED_DEPTH],
            depth: 0,
            overflow: 0,
            val: ValCtx::Free,
        }
    }

    /// The schema node governing the next value at the current position.
    fn expected(&self, arena: &SchemaArena, root: NodeId) -> NodeId {
        if self.overflow > 0 {
            return ANY;
        }
        if self.depth == 0 {
            return root;
        }
        let level = &self.levels[self.depth as usize - 1];
        match arena.node(level.node) {
            SchemaNode::Array { items } => *items,
            SchemaNode::Object { props, .. } => {
                if level.cur_prop == NO_PROP {
                    ANY
                } else {
                    props[level.cur_prop as usize].1
                }
            }
            _ => ANY,
        }
    }

    /// Processes one accepted byte and its machine events; `false` rejects
    /// the byte on schema grounds.
    fn on_byte(
        &mut self,
        arena: &SchemaArena,
        root: NodeId,
        b: u8,
        ev: Events,
        was_num: bool,
    ) -> bool {
        // A number ending on a delimiter: the delimiter byte is not part of
        // the scalar, so settle the scalar constraint first.
        if ev & EV_SCALAR_END != 0 && was_num && !self.finish_scalar(arena, false, b) {
            return false;
        }

        if ev & EV_OBJ_COMMA != 0 && !self.has_available_key(arena) {
            return false;
        }
        if ev & EV_VALUE_START != 0 {
            let node = self.expected(arena, root);
            if !self.enter_value(arena, node, b) {
                return false;
            }
        } else if ev & EV_KEY_START != 0 {
            if !self.has_available_key(arena) {
                return false;
            }
            self.start_key(arena);
        } else {
            // Content byte of the current scalar or key.
            match &mut self.val {
                ValCtx::EnumScan { node, alive, pos } => {
                    if ev & (EV_KEY_END | EV_SCALAR_END | EV_OBJ_CLOSE | EV_ARR_CLOSE) == 0 {
                        if let SchemaNode::Enum { literals } = arena.node(*node) {
                            let mut next = 0u64;
                            for (i, lit) in literals.iter().enumerate() {
                                if *alive & (1 << i) != 0 && lit.get(*pos as usize) == Some(&b) {
                                    next |= 1 << i;
                                }
                            }
                            *alive = next;
                        }
                        if *alive == 0 {
                            return false;
                        }
                        *pos += 1;
                    }
                }
                ValCtx::KeyScan { alive, pos, any } => {
                    if ev & EV_KEY_END == 0 {
                        let level = &self.levels[self.depth as usize - 1];
                        if let SchemaNode::Object { props, .. } = arena.node(level.node) {
                            let mut next = 0u64;
                            for (i, (name, _)) in props.iter().enumerate() {
                                if *alive & (1 << i) != 0 && name.get(*pos as usize) == Some(&b) {
                                    next |= 1 << i;
                                }
                            }
                            *alive = next;
                        }
                        if *alive == 0 && !*any {
                            return false;
                        }
                        *pos += 1;
                    }
                }
                ValCtx::IntNumber => {
                    if matches!(b, b'.' | b'e' | b'E') {
                        return false;
                    }
                }
                ValCtx::Free => {}
            }
        }

        if ev & EV_SCALAR_END != 0 && !was_num && !self.finish_scalar(arena, true, b) {
            return false;
        }
        if ev & EV_KEY_END != 0 && !self.finish_key(arena) {
            return false;
        }
        if ev & (EV_OBJ_CLOSE | EV_ARR_CLOSE) != 0 && !self.close_container(arena) {
            return false;
        }
        true
    }

    fn enter_value(&mut self, arena: &SchemaArena, node: NodeId, b: u8) -> bool {
        self.val = ValCtx::Free;
        let opens = matches!(b, b'{' | b'[');
        let type_ok = match arena.node(node) {
            SchemaNode::Any => true,
            SchemaNode::Object { .. } => b == b'{',
            SchemaNode::Array { .. } => b == b'[',
            SchemaNode::Str => b == b'"',
            SchemaNode::Number { integer } => {
                if matches!(b, b'-' | b'0'..=b'9') {
                    if *integer {
                        self.val = ValCtx::IntNumber;
                    }
                    true
                } else {
                    false
                }
            }
            SchemaNode::Boolean => matches!(b, b't' | b'f'),
            SchemaNode::Null => b == b'n',
            SchemaNode::Enum { literals } => {
                let mut alive = 0u64;
                for (i, lit) in literals.iter().enumerate() {
                    if lit.first() == Some(&b) {
                        alive |= 1 << i;
                    }
                }
                if alive == 0 {
                    return false;
                }
                self.val = ValCtx::EnumScan {
                    node,
                    alive,
                    pos: 1,
                };
                true
            }
        };
        if !type_ok {
            return false;
        }
        if opens {
            if self.overflow > 0 || self.depth as usize >= GUIDED_DEPTH {
                self.overflow = self.overflow.saturating_add(1);
            } else {
                let level_node = match arena.node(node) {
                    SchemaNode::Object { .. } | SchemaNode::Array { .. } => node,
                    _ => ANY,
                };
                self.levels[self.depth as usize] = Level {
                    node: level_node,
                    emitted: 0,
                    cur_prop: NO_PROP,
                };
                self.depth += 1;
            }
        }
        true
    }

    /// Whether the current object can accept another key: any unknown key
    /// when `additionalProperties`, otherwise at least one property not yet
    /// emitted.
    fn has_available_key(&self, arena: &SchemaArena) -> bool {
        if self.overflow > 0 || self.depth == 0 {
            return true;
        }
        let level = &self.levels[self.depth as usize - 1];
        match arena.node(level.node) {
            SchemaNode::Object {
                props, additional, ..
            } => *additional || (0..props.len()).any(|i| level.emitted & (1 << i) == 0),
            _ => true,
        }
    }

    fn start_key(&mut self, arena: &SchemaArena) {
        if self.overflow > 0 || self.depth == 0 {
            self.val = ValCtx::Free;
            return;
        }
        let level = &self.levels[self.depth as usize - 1];
        match arena.node(level.node) {
            SchemaNode::Object {
                props, additional, ..
            } => {
                let mut alive = 0u64;
                for i in 0..props.len() {
                    if level.emitted & (1 << i) == 0 {
                        alive |= 1 << i;
                    }
                }
                self.val = ValCtx::KeyScan {
                    alive,
                    pos: 0,
                    any: *additional,
                };
            }
            _ => self.val = ValCtx::Free,
        }
    }

    fn finish_key(&mut self, arena: &SchemaArena) -> bool {
        let ValCtx::KeyScan { alive, pos, any } = self.val else {
            self.val = ValCtx::Free;
            return true;
        };
        self.val = ValCtx::Free;
        let level = &mut self.levels[self.depth as usize - 1];
        if let SchemaNode::Object { props, .. } = arena.node(level.node) {
            let exact = (0..props.len())
                .find(|&i| alive & (1 << i) != 0 && props[i].0.len() == pos as usize);
            match exact {
                Some(i) => {
                    level.emitted |= 1 << i;
                    level.cur_prop = i as u16;
                    true
                }
                None => {
                    level.cur_prop = NO_PROP;
                    any
                }
            }
        } else {
            true
        }
    }

    fn finish_scalar(&mut self, arena: &SchemaArena, byte_included: bool, b: u8) -> bool {
        let ok = match self.val {
            ValCtx::EnumScan { node, alive, pos } => {
                if let SchemaNode::Enum { literals } = arena.node(node) {
                    literals.iter().enumerate().any(|(i, lit)| {
                        alive & (1 << i) != 0
                            && if byte_included {
                                lit.get(pos as usize) == Some(&b) && lit.len() == pos as usize + 1
                            } else {
                                lit.len() == pos as usize
                            }
                    })
                } else {
                    true
                }
            }
            _ => true,
        };
        self.val = ValCtx::Free;
        if self.depth > 0 && self.overflow == 0 {
            self.levels[self.depth as usize - 1].cur_prop = NO_PROP;
        }
        ok
    }

    fn close_container(&mut self, arena: &SchemaArena) -> bool {
        if self.overflow > 0 {
            self.overflow -= 1;
            return true;
        }
        if self.depth == 0 {
            return true;
        }
        let level = &self.levels[self.depth as usize - 1];
        let ok = match arena.node(level.node) {
            SchemaNode::Object { required, .. } => level.emitted & *required == *required,
            _ => true,
        };
        self.depth -= 1;
        self.val = ValCtx::Free;
        if self.depth > 0 {
            self.levels[self.depth as usize - 1].cur_prop = NO_PROP;
        }
        ok
    }
}

/// The per-sequence constraint: automaton state, schema guidance, and the
/// shared byte table.
pub struct JsonConstraint {
    table: Arc<TokenByteTable>,
    arena: Arc<SchemaArena>,
    root: NodeId,
    machine: JsonMachine,
    guide: GuideState,
    mode: JsonMode,
}

impl JsonConstraint {
    pub fn new(table: Arc<TokenByteTable>, mode: JsonMode, arena: Arc<SchemaArena>) -> Self {
        let root = arena.root();
        Self {
            table,
            arena,
            root,
            machine: JsonMachine::new(mode),
            guide: GuideState::new(),
            mode,
        }
    }

    fn walk(&self, machine: &mut JsonMachine, guide: &mut GuideState, bytes: &[u8]) -> bool {
        for &b in bytes {
            let was_num = machine.in_number();
            let Some(ev) = machine.step_ev(b, self.mode) else {
                return false;
            };
            if !guide.on_byte(&self.arena, self.root, b, ev, was_num) {
                return false;
            }
        }
        true
    }

    /// Marks every token whose bytes are illegal (syntactically or against
    /// the schema guidance) as `false`; stop tokens are `true` only when the
    /// machine accepts a complete value.
    pub fn allowed(&self, vocab_size: usize) -> Vec<bool> {
        let accepting = self.machine.accepting();
        let mut allowed = vec![false; vocab_size];
        for (id, entry) in self.table.bytes.iter().enumerate().take(vocab_size) {
            let Some(tok_bytes) = entry else { continue };
            if tok_bytes.is_empty() {
                continue;
            }
            let mut m = self.machine;
            let mut g = self.guide;
            if self.walk(&mut m, &mut g, tok_bytes) {
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

    /// Advances with the chosen token. Stop tokens (legal only when
    /// accepting) leave the state unchanged.
    pub fn advance(&mut self, token: u32) {
        if self.table.stop_ids.contains(&token) {
            return;
        }
        let Some(Some(tok_bytes)) = self.table.bytes.get(token as usize) else {
            return;
        };
        let bytes = tok_bytes.clone();
        let mut m = self.machine;
        let mut g = self.guide;
        if self.walk(&mut m, &mut g, &bytes) {
            self.machine = m;
            self.guide = g;
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

    fn person_schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "color": {"enum": ["red", "green", "blue"]},
                "tags": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["name", "age"],
            "additionalProperties": false
        })
    }

    fn guided(schema: &serde_json::Value) -> JsonConstraint {
        let table = TokenByteTable::from_raw(
            (0x20u8..0x7F)
                .map(|b| Some(vec![b]))
                .chain([Some(b"\n".to_vec()), None])
                .collect(),
            vec![96],
        );
        JsonConstraint::new(table, JsonMode::Object, SchemaArena::compile(schema))
    }

    fn walk_str(c: &mut JsonConstraint, doc: &str) -> bool {
        let mut m = c.machine;
        let mut g = c.guide;
        let ok = c.walk(&mut m, &mut g, doc.as_bytes());
        if ok {
            c.machine = m;
            c.guide = g;
        }
        ok
    }

    /// Contract: schema guidance accepts schema-valid documents and rejects,
    /// at the first offending byte, unknown keys, missing-required closes,
    /// wrong value types, non-integer numbers under integer, non-member enum
    /// values, and wrongly typed array items.
    #[test]
    fn schema_guidance_accepts_and_rejects() {
        let schema = person_schema();
        let mut c = guided(&schema);
        assert!(walk_str(
            &mut c,
            r#"{"name": "Ada", "age": 36, "color": "green", "tags": ["x", "y"]}"#
        ));
        assert!(c.machine.accepting());

        for (doc, why) in [
            (
                r#"{"nickname"#,
                "unknown key with additionalProperties false",
            ),
            (r#"{"name": "Ada"}"#, "close before required 'age'"),
            (r#"{"name": 3"#, "number where string required"),
            (r#"{"name": "A", "age": 3.5"#, "non-integer under integer"),
            (
                r#"{"name": "A", "age": 3, "color": "purple"#,
                "non-member enum",
            ),
            (
                r#"{"name": "A", "age": 3, "tags": [1"#,
                "wrong array item type",
            ),
            (r#"{"name": "A", "name"#, "duplicate known key"),
        ] {
            let mut c = guided(&schema);
            assert!(!walk_str(&mut c, doc), "accepted invalid ({why}): {doc:?}");
        }
    }

    /// Contract: unsupported constructs (anyOf) fall back to syntax-only
    /// guidance, and $ref into $defs resolves.
    #[test]
    fn schema_fallback_and_ref() {
        let any_of = serde_json::json!({
            "type": "object",
            "properties": {"x": {"anyOf": [{"type": "string"}, {"type": "number"}]}},
            "required": ["x"]
        });
        let mut c = guided(&any_of);
        assert!(
            walk_str(&mut c, r#"{"x": [1, 2]}"#),
            "anyOf subtree must be unguided"
        );

        let with_ref = serde_json::json!({
            "type": "object",
            "properties": {"p": {"$ref": "#/$defs/point"}},
            "required": ["p"],
            "$defs": {"point": {"type": "object", "properties": {"x": {"type": "integer"}},
                                  "required": ["x"], "additionalProperties": false}}
        });
        let mut c = guided(&with_ref);
        assert!(walk_str(&mut c, r#"{"p": {"x": 7}}"#));
        let mut c = guided(&with_ref);
        assert!(
            !walk_str(&mut c, r#"{"p": {"y"#),
            "ref'd schema must guide keys"
        );
    }

    /// Contract (soundness + strictness): random mask-guided generation with
    /// single-byte tokens is never stuck (some token or stop is always
    /// allowed) and every accepted document parses and satisfies the schema.
    #[test]
    fn fuzz_guided_walks_always_valid() {
        let schema = person_schema();
        for seed in 0u64..40 {
            let mut c = guided(&schema);
            let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let mut out: Vec<u8> = Vec::new();
            let mut finished = false;
            for _ in 0..400 {
                let allowed = c.allowed(97);
                let choices: Vec<usize> = allowed
                    .iter()
                    .enumerate()
                    .filter(|&(_, &a)| a)
                    .map(|(i, _)| i)
                    .collect();
                assert!(
                    !choices.is_empty(),
                    "stuck at {:?}",
                    String::from_utf8_lossy(&out)
                );
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let pick = choices[(rng >> 33) as usize % choices.len()];
                if pick == 96 {
                    finished = true;
                    break;
                }
                out.extend(c.table.bytes[pick].as_ref().unwrap());
                c.advance(pick as u32);
            }
            if !finished {
                continue;
            }
            let text = String::from_utf8(out).expect("guided output is UTF-8");
            let v: serde_json::Value =
                serde_json::from_str(&text).unwrap_or_else(|e| panic!("unparseable {text:?}: {e}"));
            let obj = v
                .as_object()
                .unwrap_or_else(|| panic!("not an object: {text:?}"));
            assert!(obj["name"].is_string(), "{text:?}");
            assert!(obj["age"].is_i64() || obj["age"].is_u64(), "{text:?}");
            if let Some(color) = obj.get("color") {
                assert!(
                    ["red", "green", "blue"].contains(&color.as_str().unwrap_or("")),
                    "{text:?}"
                );
            }
            if let Some(tags) = obj.get("tags") {
                assert!(
                    tags.as_array().unwrap().iter().all(|t| t.is_string()),
                    "{text:?}"
                );
            }
            for key in obj.keys() {
                assert!(
                    ["name", "age", "color", "tags"].contains(&key.as_str()),
                    "unknown key in {text:?}"
                );
            }
        }
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
        let mut c = JsonConstraint::new(table, JsonMode::Object, SchemaArena::unconstrained());

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
