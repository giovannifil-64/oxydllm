//! Reading a GGUF file's header to decide, before downloading it, whether the
//! engine can load it at all.
//!
//! A GGUF file states its architecture and the quantization of every tensor in
//! a header at the very start, so a range request over a few tens of megabytes
//! answers in seconds what a full download answers in half an hour. That
//! matters because publishers ship "dynamic" builds that mix quantizations per
//! tensor: unsloth's `UD-*` and AtomicChat's `AD-*` variants of Qwen3.8-27B put
//! importance-quantization types next to ordinary K-quants, and a single tensor
//! in a format candle cannot decode rejects the entire file. One of those
//! builds carries exactly one such tensor out of 866.
//!
//! [`inspect_header`] parses bytes already in hand; [`probe_remote`] fetches
//! them over HTTP. Both are best-effort by design: a probe that cannot reach the
//! network, or that needs more header than it was given, reports
//! [`HeaderVerdict::Unknown`] so a download proceeds rather than being blocked
//! by a diagnostic.

use std::collections::BTreeMap;

/// Quantizations candle can decode, mirroring its own `GgmlDType::from_u32`,
/// which is crate-private. Numbers are the ggml type ids.
const SUPPORTED_TYPES: &[u32] = &[0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 30];

/// Names for the ggml type ids; ids absent here are reported numerically.
const TYPE_NAMES: &[(u32, &str)] = &[
    (0, "F32"),
    (1, "F16"),
    (2, "Q4_0"),
    (3, "Q4_1"),
    (6, "Q5_0"),
    (7, "Q5_1"),
    (8, "Q8_0"),
    (9, "Q8_1"),
    (10, "Q2_K"),
    (11, "Q3_K"),
    (12, "Q4_K"),
    (13, "Q5_K"),
    (14, "Q6_K"),
    (15, "Q8_K"),
    (30, "BF16"),
    (16, "IQ2_XXS"),
    (17, "IQ2_XS"),
    (18, "IQ3_XXS"),
    (19, "IQ1_S"),
    (20, "IQ4_NL"),
    (21, "IQ3_S"),
    (22, "IQ2_S"),
    (23, "IQ4_XS"),
    (29, "IQ1_M"),
    (34, "TQ1_0"),
    (35, "TQ2_0"),
    (39, "MXFP4"),
];

/// What a file holds, tensor by tensor, as the header states it.
///
/// A label like `Q4_K_M` names a recipe, and publishers disagree on the
/// recipe: the same label from one keeps three quarters of the tensors in a
/// type another never uses. The header is the only thing that says, so this is
/// what a download decision and a quality figure should rest on.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Composition {
    pub tensors: usize,
    pub by_type: BTreeMap<String, usize>,
    pub embedding: Option<String>,
    pub output: Option<String>,
}

impl Composition {
    /// One line: the count per type, most numerous first, then the types of
    /// the token embedding and the output projection where the file has them.
    pub fn line(&self) -> String {
        let mut types: Vec<(&String, &usize)> = self.by_type.iter().collect();
        types.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        let list = types
            .iter()
            .map(|(k, n)| format!("{k} {n}"))
            .collect::<Vec<_>>()
            .join(", ");
        let mut line = format!("{} tensors: {list}", self.tensors);
        if let Some(e) = &self.embedding {
            line.push_str(&format!("; token_embd {e}"));
        }
        if let Some(o) = &self.output {
            line.push_str(&format!("; output {o}"));
        }
        line
    }
}

/// What a header says about a file's loadability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeaderVerdict {
    /// Every tensor is in a quantization the engine decodes.
    Loadable {
        architecture: Option<String>,
        composition: Composition,
    },
    /// At least one tensor is not, with a count per offending type.
    Unreadable {
        architecture: Option<String>,
        offenders: BTreeMap<String, usize>,
        composition: Composition,
    },
    /// The header could not be read far enough to tell.
    Unknown,
}

impl HeaderVerdict {
    /// The file's composition, where the header could be read.
    pub fn composition_line(&self) -> Option<String> {
        match self {
            Self::Loadable { composition, .. } | Self::Unreadable { composition, .. } => {
                Some(composition.line())
            }
            Self::Unknown => None,
        }
    }

    /// A message naming what cannot be decoded and what to do instead, or
    /// `None` when there is nothing to warn about.
    pub fn refusal(&self) -> Option<String> {
        let Self::Unreadable { offenders, .. } = self else {
            return None;
        };
        let list = offenders
            .iter()
            .map(|(k, n)| format!("{k} ({n})"))
            .collect::<Vec<_>>()
            .join(", ");
        Some(format!(
            "this file stores tensors in quantizations the engine cannot decode: {list}. \
             These come from the importance-quantization family, which mixed-precision \
             builds use heavily; the same label from a publisher that quantizes uniformly, \
             or any variant without a UD, AD or IQ marker in its name, will usually load"
        ))
    }
}

fn type_name(id: u32) -> String {
    TYPE_NAMES
        .iter()
        .find(|(k, _)| *k == id)
        .map(|(_, n)| (*n).to_string())
        .unwrap_or_else(|| format!("type {id}"))
}

/// A cursor that refuses to read past the bytes it was given, so a truncated
/// header ends the walk instead of misreading it.
struct Cursor<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.i.checked_add(n)?;
        let s = self.b.get(self.i..end)?;
        self.i = end;
        Some(s)
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }

    fn string(&mut self) -> Option<String> {
        let n = self.u64()? as usize;
        Some(String::from_utf8_lossy(self.take(n)?).into_owned())
    }

    /// Steps over one metadata value of the given GGUF type tag.
    fn skip_value(&mut self, tag: u32) -> Option<()> {
        match tag {
            0 | 1 | 7 => self.take(1).map(|_| ()),
            2 | 3 => self.take(2).map(|_| ()),
            4..=6 => self.take(4).map(|_| ()),
            10..=12 => self.take(8).map(|_| ()),
            8 => self.string().map(|_| ()),
            9 => {
                let elem = self.u32()?;
                let n = self.u64()?;
                for _ in 0..n {
                    self.skip_value(elem)?;
                }
                Some(())
            }
            _ => None,
        }
    }
}

/// Reads architecture and tensor quantizations out of GGUF header bytes.
///
/// Returns [`HeaderVerdict::Unknown`] when `bytes` stops before the tensor
/// table ends, which is the expected outcome for a header larger than the slice
/// the caller fetched.
pub fn inspect_header(bytes: &[u8]) -> HeaderVerdict {
    let mut c = Cursor { b: bytes, i: 0 };
    if c.take(4) != Some(b"GGUF") {
        return HeaderVerdict::Unknown;
    }
    let (Some(_version), Some(n_tensors), Some(n_kv)) = (c.u32(), c.u64(), c.u64()) else {
        return HeaderVerdict::Unknown;
    };

    let mut architecture = None;
    for _ in 0..n_kv {
        let (Some(key), Some(tag)) = (c.string(), c.u32()) else {
            return HeaderVerdict::Unknown;
        };
        if key == "general.architecture" && tag == 8 {
            match c.string() {
                Some(v) => architecture = Some(v),
                None => return HeaderVerdict::Unknown,
            }
        } else if c.skip_value(tag).is_none() {
            return HeaderVerdict::Unknown;
        }
    }

    let mut offenders: BTreeMap<String, usize> = BTreeMap::new();
    let mut composition = Composition::default();
    for _ in 0..n_tensors {
        let Some(name) = c.string() else {
            return HeaderVerdict::Unknown;
        };
        let Some(n_dims) = c.u32() else {
            return HeaderVerdict::Unknown;
        };
        for _ in 0..n_dims {
            if c.u64().is_none() {
                return HeaderVerdict::Unknown;
            }
        }
        let (Some(dtype), Some(_offset)) = (c.u32(), c.u64()) else {
            return HeaderVerdict::Unknown;
        };
        let tname = type_name(dtype);
        if !SUPPORTED_TYPES.contains(&dtype) {
            *offenders.entry(tname.clone()).or_default() += 1;
        }
        composition.tensors += 1;
        *composition.by_type.entry(tname.clone()).or_default() += 1;
        if name == "token_embd.weight" {
            composition.embedding = Some(tname.clone());
        } else if name == "output.weight" {
            composition.output = Some(tname);
        }
    }

    if offenders.is_empty() {
        HeaderVerdict::Loadable {
            architecture,
            composition,
        }
    } else {
        HeaderVerdict::Unreadable {
            architecture,
            offenders,
            composition,
        }
    }
}

/// Bytes of `url` to ask for. A vocabulary of a few hundred thousand tokens and
/// its merge table dominate the header, so this is generous on purpose; the
/// server sends only what exists.
const HEADER_PROBE_BYTES: u64 = 96 * 1024 * 1024;

/// Fetches the head of a remote GGUF file and reports whether it can be loaded.
///
/// Never fails: any transport problem is [`HeaderVerdict::Unknown`], since this
/// exists to save a wasted download, not to become a new way for one to be
/// refused.
pub fn probe_remote(
    client: &reqwest::blocking::Client,
    url: &str,
    token: Option<&str>,
) -> HeaderVerdict {
    let mut req = client.get(url).header(
        reqwest::header::RANGE,
        format!("bytes=0-{}", HEADER_PROBE_BYTES - 1),
    );
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    match req.send().and_then(|r| r.bytes()) {
        Ok(b) => inspect_header(&b),
        Err(_) => HeaderVerdict::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(tensor_types: &[u32]) -> Vec<u8> {
        let named: Vec<(String, u32)> = tensor_types
            .iter()
            .enumerate()
            .map(|(i, t)| (format!("blk.{i}.weight"), *t))
            .collect();
        header_named(&named)
    }

    fn header_named(tensors: &[(String, u32)]) -> Vec<u8> {
        let tensor_types: Vec<u32> = tensors.iter().map(|(_, t)| *t).collect();
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&(tensor_types.len() as u64).to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());

        let key = b"general.architecture";
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key);
        b.extend_from_slice(&8u32.to_le_bytes());
        let val = b"qwen35";
        b.extend_from_slice(&(val.len() as u64).to_le_bytes());
        b.extend_from_slice(val);

        for (i, t) in tensor_types.iter().enumerate() {
            let name = tensors[i].0.clone();
            b.extend_from_slice(&(name.len() as u64).to_le_bytes());
            b.extend_from_slice(name.as_bytes());
            b.extend_from_slice(&1u32.to_le_bytes());
            b.extend_from_slice(&64u64.to_le_bytes());
            b.extend_from_slice(&t.to_le_bytes());
            b.extend_from_slice(&0u64.to_le_bytes());
        }
        b
    }

    /// Bytes from a seed, mixing pure noise with headers that are structurally
    /// plausible but hostile in their declared values.
    fn hostile_bytes(seed: u64) -> Vec<u8> {
        let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut next = |lo: usize, hi: usize| -> usize {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            lo + ((rng >> 33) as usize) % (hi - lo + 1)
        };
        let mut b = Vec::new();
        match seed % 3 {
            // Pure noise: nothing should get past the magic, but prove it.
            0 => {
                for _ in 0..next(0, 512) {
                    b.push(next(0, 255) as u8);
                }
            }
            // Past the magic, then noise: exercises the metadata walk.
            1 => {
                b.extend_from_slice(b"GGUF");
                for _ in 0..next(0, 512) {
                    b.push(next(0, 255) as u8);
                }
            }
            // A header whose declared counts and lengths are chosen to be
            // absurd: the point is that the work stays bounded by the bytes in
            // hand rather than by a number the file asked for.
            _ => {
                b.extend_from_slice(b"GGUF");
                b.extend_from_slice(&3u32.to_le_bytes());
                let n_tensors = [0u64, 1, 1 << 20, u64::MAX][next(0, 3)];
                let n_kv = [0u64, 1, 1 << 20, u64::MAX][next(0, 3)];
                b.extend_from_slice(&n_tensors.to_le_bytes());
                b.extend_from_slice(&n_kv.to_le_bytes());
                for _ in 0..next(0, 6) {
                    let len = [0u64, 4, 1 << 40, u64::MAX][next(0, 3)];
                    b.extend_from_slice(&len.to_le_bytes());
                    for _ in 0..next(0, 8) {
                        b.push(next(0, 255) as u8);
                    }
                    b.extend_from_slice(&(next(0, 20) as u32).to_le_bytes());
                }
                b.truncate(next(0, b.len().max(1)));
            }
        }
        b
    }

    /// Contract: any sequence of bytes produces a verdict.
    ///
    /// These bytes arrive over the network, from a range request against a file
    /// nobody in this project wrote, and they are read by a hand-rolled cursor
    /// whose every bound comes from the file itself: counts of tensors, counts
    /// of metadata entries, lengths of strings. The properties that matter are
    /// that no input panics, that the work stays proportional to the bytes in
    /// hand rather than to a number the file declared, and that a verdict of
    /// unreadable always names something.
    /// Contract: the header's composition comes back type by type, most
    /// numerous first, with the embedding and output types named, whether or
    /// not the file loads.
    #[test]
    fn a_header_reports_what_the_file_holds() {
        let tensors: Vec<(String, u32)> = vec![
            ("token_embd.weight".to_string(), 14),
            ("blk.0.attn_q.weight".to_string(), 12),
            ("blk.0.attn_k.weight".to_string(), 12),
            ("blk.0.ffn_down.weight".to_string(), 23),
            ("output.weight".to_string(), 8),
        ];
        let verdict = inspect_header(&header_named(&tensors));
        let HeaderVerdict::Unreadable {
            offenders,
            composition,
            ..
        } = &verdict
        else {
            panic!("an IQ4_XS tensor must make the file unreadable, got {verdict:?}");
        };
        assert_eq!(offenders.get("IQ4_XS"), Some(&1));
        assert_eq!(composition.tensors, 5);
        assert_eq!(
            composition.line(),
            "5 tensors: Q4_K 2, IQ4_XS 1, Q6_K 1, Q8_0 1; token_embd Q6_K; output Q8_0"
        );
        let loadable = inspect_header(&header(&[12, 12, 14]));
        assert_eq!(
            loadable.composition_line().as_deref(),
            Some("3 tensors: Q4_K 2, Q6_K 1")
        );
        assert_eq!(inspect_header(b"junk").composition_line(), None);
    }

    #[test]
    fn fuzz_inspect_header_survives_any_bytes() {
        for seed in 0u64..512 {
            let bytes = hostile_bytes(seed);
            let verdict = inspect_header(&bytes);
            match &verdict {
                HeaderVerdict::Unreadable { offenders, .. } => {
                    assert!(
                        !offenders.is_empty(),
                        "seed {seed}: unreadable names nothing"
                    );
                    assert!(verdict.refusal().is_some(), "seed {seed}: no message");
                }
                _ => assert!(
                    verdict.refusal().is_none(),
                    "seed {seed}: a readable verdict produced a refusal"
                ),
            }
        }
    }

    /// Contract: a declared count the bytes cannot back is answered from the
    /// bytes, not from the count.
    ///
    /// A header claiming four billion tensors in forty bytes must come back
    /// unknown rather than iterating four billion times or reserving room for
    /// them. The assertion is that this returns at all; a loop driven by the
    /// declared count would never reach it.
    #[test]
    fn fuzz_a_declared_count_cannot_drive_the_work() {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&u64::MAX.to_le_bytes());
        b.extend_from_slice(&u64::MAX.to_le_bytes());
        b.extend_from_slice(&[0u8; 16]);
        assert_eq!(inspect_header(&b), HeaderVerdict::Unknown);

        // The same for a string longer than any machine has memory for.
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(inspect_header(&b), HeaderVerdict::Unknown);
    }

    /// Contract: a file whose tensors are all decodable is reported loadable,
    /// with the architecture the caller may want to show.
    #[test]
    fn plain_k_quants_are_loadable() {
        let v = inspect_header(&header(&[12, 14, 0]));
        let HeaderVerdict::Loadable {
            architecture,
            composition,
        } = &v
        else {
            panic!("plain K-quants must load, got {v:?}");
        };
        assert_eq!(architecture.as_deref(), Some("qwen35"));
        assert_eq!(composition.line(), "3 tensors: F32 1, Q4_K 1, Q6_K 1");
        assert!(v.refusal().is_none());
    }

    /// Contract: one undecodable tensor among many condemns the file, because
    /// the loader cannot skip it. AtomicChat's AD-Q4_K_M carries exactly one
    /// IQ4_XS tensor out of 866 and is unusable for that reason alone.
    #[test]
    fn a_single_bad_tensor_is_enough() {
        let mut types = vec![12u32; 40];
        types.push(23);
        match inspect_header(&header(&types)) {
            HeaderVerdict::Unreadable { offenders, .. } => {
                assert_eq!(offenders.get("IQ4_XS"), Some(&1));
            }
            other => panic!("expected Unreadable, got {other:?}"),
        }
    }

    /// Contract: the refusal names every offending quantization with its count,
    /// so the reader can tell a stray tensor from a wholesale rebuild.
    #[test]
    fn refusal_names_each_offending_type() {
        let v = inspect_header(&header(&[23, 23, 21, 12]));
        let msg = v.refusal().expect("a refusal");
        assert!(msg.contains("IQ4_XS (2)"), "{msg}");
        assert!(msg.contains("IQ3_S (1)"), "{msg}");
        // The advice must not name the very label the caller asked for: in
        // these repositories "Q4_K_M" is itself the mixed-precision build.
        assert!(!msg.contains("such as Q4_K_M"), "{msg}");
    }

    /// Contract: a header cut short reports Unknown rather than guessing, so a
    /// probe that fetched too few bytes cannot block a good download.
    #[test]
    fn a_truncated_header_is_unknown() {
        let full = header(&[12, 12, 12]);
        assert_eq!(
            inspect_header(&full[..full.len() / 2]),
            HeaderVerdict::Unknown
        );
        assert_eq!(inspect_header(b"not a gguf"), HeaderVerdict::Unknown);
        assert!(HeaderVerdict::Unknown.refusal().is_none());
    }
}
