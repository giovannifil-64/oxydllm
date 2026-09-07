//! The GGUF header, read by this crate so that a tensor type candle has no
//! name for does not stop the file at the door.
//!
//! Candle's reader turns every tensor's type id into its own enum and fails on
//! the ids it lacks, which are the importance-quantized ones. The metadata and
//! the tensor table are the same bytes either way; reading them here keeps
//! candle's [`Content`] for everything that consumes metadata and hands the
//! loader the complete tensor table with the raw type ids.

use candle_core::quantized::GgmlDType;
use candle_core::quantized::gguf_file::{Content, TensorInfo, Value, VersionedMagic};
use std::collections::HashMap;

/// One tensor as the header states it: the ggml type id as written, the
/// dimensions in candle's order (outermost first), and the offset into the
/// data section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorEntry {
    pub name: String,
    pub type_id: u32,
    pub dims: Vec<usize>,
    pub offset: u64,
}

/// A parsed header: candle's view for the metadata readers, with the tensors
/// candle can name, and the whole tensor table for the loader.
pub struct GgufHeader {
    pub content: Content,
    pub tensors: Vec<TensorEntry>,
}

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

/// The name of a ggml type id, as llama.cpp spells it.
pub fn type_name(id: u32) -> String {
    TYPE_NAMES
        .iter()
        .find(|(k, _)| *k == id)
        .map(|(_, n)| (*n).to_string())
        .unwrap_or_else(|| format!("type {id}"))
}

/// The candle block type a ggml type id names, where it names one.
pub fn candle_dtype(type_id: u32) -> Option<GgmlDType> {
    Some(match type_id {
        0 => GgmlDType::F32,
        1 => GgmlDType::F16,
        2 => GgmlDType::Q4_0,
        3 => GgmlDType::Q4_1,
        6 => GgmlDType::Q5_0,
        7 => GgmlDType::Q5_1,
        8 => GgmlDType::Q8_0,
        9 => GgmlDType::Q8_1,
        10 => GgmlDType::Q2K,
        11 => GgmlDType::Q3K,
        12 => GgmlDType::Q4K,
        13 => GgmlDType::Q5K,
        14 => GgmlDType::Q6K,
        15 => GgmlDType::Q8K,
        30 => GgmlDType::BF16,
        _ => return None,
    })
}

const DEFAULT_ALIGNMENT: u64 = 32;
const MAX_DIMS: u32 = 4;
const MAX_DEPTH: usize = 64;

struct Cursor<'a> {
    b: &'a [u8],
    i: usize,
    magic: VersionedMagic,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> anyhow::Result<&'a [u8]> {
        let end = self
            .i
            .checked_add(n)
            .filter(|end| *end <= self.b.len())
            .ok_or_else(|| anyhow::anyhow!("header ends at byte {} of {}", self.i, self.b.len()))?;
        let s = &self.b[self.i..end];
        self.i = end;
        Ok(s)
    }

    fn u8(&mut self) -> anyhow::Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> anyhow::Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> anyhow::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> anyhow::Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    /// A count or length, which the first version wrote in four bytes and the
    /// later ones in eight.
    fn len(&mut self) -> anyhow::Result<usize> {
        let n = match self.magic {
            VersionedMagic::GgufV1 => u64::from(self.u32()?),
            VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => self.u64()?,
        };
        usize::try_from(n).map_err(|_| anyhow::anyhow!("length {n} does not fit"))
    }

    fn string(&mut self) -> anyhow::Result<String> {
        let n = self.len()?;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }

    fn value(&mut self, tag: u32, depth: usize) -> anyhow::Result<Value> {
        Ok(match tag {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.u8()? as i8),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.u16()? as i16),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.u8()? != 0),
            8 => Value::String(self.string()?),
            9 => {
                if depth >= MAX_DEPTH {
                    anyhow::bail!("metadata nests deeper than {MAX_DEPTH}");
                }
                let elem = self.u32()?;
                let n = self.len()?;
                let mut items = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    items.push(self.value(elem, depth + 1)?);
                }
                Value::Array(items)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            _ => anyhow::bail!("metadata value type {tag} is not GGUF"),
        })
    }
}

impl GgufHeader {
    /// Parses the header at the start of `bytes`, which may be the whole file.
    ///
    /// ## Errors
    /// Fails when the bytes are not GGUF, end before the tensor table does,
    /// or state something no GGUF file may.
    pub fn parse(bytes: &[u8]) -> anyhow::Result<Self> {
        let mut c = Cursor {
            b: bytes,
            i: 0,
            magic: VersionedMagic::GgufV3,
        };
        if c.take(4)? != b"GGUF" {
            anyhow::bail!("not a GGUF file");
        }
        c.magic = match c.u32()? {
            1 => VersionedMagic::GgufV1,
            2 => VersionedMagic::GgufV2,
            3 => VersionedMagic::GgufV3,
            v => anyhow::bail!("GGUF version {v} is not one this reader knows"),
        };
        let n_tensors = c.len()?;
        let n_kv = c.len()?;

        let mut metadata = HashMap::with_capacity(n_kv.min(1 << 16));
        for _ in 0..n_kv {
            let key = c.string()?;
            let tag = c.u32()?;
            let value = c
                .value(tag, 0)
                .map_err(|e| anyhow::anyhow!("metadata '{key}': {e}"))?;
            metadata.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(n_tensors.min(1 << 16));
        let mut tensor_infos = HashMap::with_capacity(n_tensors.min(1 << 16));
        for _ in 0..n_tensors {
            let name = c.string()?;
            let n_dims = c.u32()?;
            if n_dims > MAX_DIMS {
                anyhow::bail!("tensor '{name}' has {n_dims} dimensions");
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(c.len()?);
            }
            dims.reverse();
            let type_id = c.u32()?;
            let offset = c.u64()?;
            if let Some(ggml_dtype) = candle_dtype(type_id) {
                tensor_infos.insert(
                    name.clone(),
                    TensorInfo {
                        ggml_dtype,
                        shape: candle_core::Shape::from(dims.clone()),
                        offset,
                    },
                );
            }
            tensors.push(TensorEntry {
                name,
                type_id,
                dims,
                offset,
            });
        }

        let alignment = match metadata.get("general.alignment") {
            Some(Value::U8(v)) => u64::from(*v),
            Some(Value::U16(v)) => u64::from(*v),
            Some(Value::U32(v)) => u64::from(*v),
            Some(Value::I8(v)) if *v >= 0 => *v as u64,
            Some(Value::I16(v)) if *v >= 0 => *v as u64,
            Some(Value::I32(v)) if *v >= 0 => *v as u64,
            _ => DEFAULT_ALIGNMENT,
        };
        if alignment == 0 {
            anyhow::bail!("general.alignment is zero");
        }
        let tensor_data_offset = (c.i as u64).div_ceil(alignment) * alignment;
        Ok(Self {
            content: Content {
                magic: c.magic,
                metadata,
                tensor_infos,
                tensor_data_offset,
            },
            tensors,
        })
    }

    /// Reads the header of the file at `path`.
    ///
    /// ## Errors
    /// Fails when the file cannot be opened or mapped, or as [`Self::parse`].
    pub fn read(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)
            .map_err(|e| anyhow::anyhow!("cannot open {}: {e}", path.display()))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| anyhow::anyhow!("cannot map {}: {e}", path.display()))?;
        Self::parse(&mmap[..]).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Writes a GGUF v3 file with `kv` metadata entries and `tensors` in the
    /// table, each given as `(name, dims as written, type id)`, followed by
    /// `data` at the aligned data offset.
    pub(crate) fn gguf_bytes(
        kv: &[(&str, Value)],
        tensors: &[(&str, &[u64], u32)],
        data: &[u8],
    ) -> Vec<u8> {
        fn put_str(out: &mut Vec<u8>, s: &str) {
            out.extend_from_slice(&(s.len() as u64).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        fn tag(v: &Value) -> u32 {
            match v {
                Value::U8(_) => 0,
                Value::I8(_) => 1,
                Value::U16(_) => 2,
                Value::I16(_) => 3,
                Value::U32(_) => 4,
                Value::I32(_) => 5,
                Value::F32(_) => 6,
                Value::Bool(_) => 7,
                Value::String(_) => 8,
                Value::Array(_) => 9,
                Value::U64(_) => 10,
                Value::I64(_) => 11,
                Value::F64(_) => 12,
            }
        }
        fn put_value(out: &mut Vec<u8>, v: &Value) {
            match v {
                Value::U8(x) => out.push(*x),
                Value::I8(x) => out.push(*x as u8),
                Value::U16(x) => out.extend_from_slice(&x.to_le_bytes()),
                Value::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
                Value::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
                Value::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
                Value::F32(x) => out.extend_from_slice(&x.to_le_bytes()),
                Value::Bool(x) => out.push(u8::from(*x)),
                Value::String(s) => put_str(out, s),
                Value::Array(items) => {
                    let elem = items.first().map_or(4u32, tag);
                    out.extend_from_slice(&elem.to_le_bytes());
                    out.extend_from_slice(&(items.len() as u64).to_le_bytes());
                    for it in items {
                        put_value(out, it);
                    }
                }
                Value::U64(x) => out.extend_from_slice(&x.to_le_bytes()),
                Value::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
                Value::F64(x) => out.extend_from_slice(&x.to_le_bytes()),
            }
        }
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(kv.len() as u64).to_le_bytes());
        for (k, v) in kv {
            put_str(&mut out, k);
            out.extend_from_slice(&tag(v).to_le_bytes());
            put_value(&mut out, v);
        }
        let mut offset = 0u64;
        for (name, dims, type_id) in tensors {
            put_str(&mut out, name);
            out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in *dims {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&type_id.to_le_bytes());
            out.extend_from_slice(&offset.to_le_bytes());
            let elems: u64 = dims.iter().product();
            let bytes = match type_id {
                0 => elems * 4,
                12 => elems / 256 * 144,
                23 => elems / 256 * 136,
                20 => elems / 32 * 18,
                21 => elems / 256 * 110,
                other => panic!("no size for type {other} in this fixture"),
            };
            offset += bytes.div_ceil(32) * 32;
        }
        let pad = out.len().div_ceil(32) * 32 - out.len();
        out.extend(std::iter::repeat_n(0u8, pad));
        out.extend_from_slice(data);
        out
    }

    /// Contract: on a file made only of types candle names, this reader and
    /// candle's agree on every metadata value, every tensor, and the data
    /// offset, so nothing that read through candle before reads differently
    /// now.
    #[test]
    fn two_readers_agree_on_a_file_candle_can_read() {
        let kv = [
            ("general.architecture", Value::String("llama".into())),
            ("llama.block_count", Value::U32(2)),
            ("general.alignment", Value::U32(64)),
            (
                "tokenizer.ggml.scores",
                Value::Array(vec![Value::F32(-1.5), Value::F32(2.0)]),
            ),
            (
                "tokenizer.ggml.tokens",
                Value::Array(vec![
                    Value::String("<s>".into()),
                    Value::String("ciao".into()),
                ]),
            ),
            ("a.bool", Value::Bool(true)),
            ("a.i64", Value::I64(-7)),
            ("a.f64", Value::F64(0.25)),
        ];
        let tensors: [(&str, &[u64], u32); 2] = [
            ("token_embd.weight", &[256, 4], 12),
            ("output_norm.weight", &[8], 0),
        ];
        let bytes = gguf_bytes(&kv, &tensors, &[0u8; 4096]);

        let ours = GgufHeader::parse(&bytes).unwrap();
        let theirs = Content::read(&mut std::io::Cursor::new(&bytes)).unwrap();

        assert_eq!(ours.content.tensor_data_offset, theirs.tensor_data_offset);
        assert_eq!(ours.content.metadata.len(), theirs.metadata.len());
        for (k, v) in &theirs.metadata {
            assert_eq!(
                format!("{:?}", ours.content.metadata[k]),
                format!("{v:?}"),
                "{k}"
            );
        }
        assert_eq!(ours.content.tensor_infos.len(), theirs.tensor_infos.len());
        for (k, v) in &theirs.tensor_infos {
            let o = &ours.content.tensor_infos[k];
            assert_eq!(
                (o.ggml_dtype, o.shape.dims(), o.offset),
                (v.ggml_dtype, v.shape.dims(), v.offset),
                "{k}"
            );
        }
        assert_eq!(ours.tensors.len(), 2);
        assert_eq!(ours.tensors[0].dims, [4, 256]);
    }

    /// Contract: a tensor in a type candle cannot name is kept in the table
    /// with its id and left out of candle's view, and the file still reads,
    /// where candle's reader refuses it.
    #[test]
    fn a_type_candle_cannot_name_stays_in_the_table() {
        let kv = [("general.architecture", Value::String("qwen3".into()))];
        let tensors: [(&str, &[u64], u32); 3] = [
            ("blk.0.ffn_down.weight", &[256, 2], 23),
            ("blk.0.ffn_up.weight", &[32, 2], 20),
            ("output.weight", &[256, 2], 12),
        ];
        let bytes = gguf_bytes(&kv, &tensors, &[0u8; 1024]);

        assert!(Content::read(&mut std::io::Cursor::new(&bytes)).is_err());
        let ours = GgufHeader::parse(&bytes).unwrap();
        assert_eq!(ours.tensors.len(), 3);
        assert_eq!(ours.content.tensor_infos.len(), 1);
        assert_eq!(ours.tensors[0].type_id, 23);
        assert_eq!(ours.tensors[0].dims, [2, 256]);
        assert_eq!(
            ours.tensors[1].offset, 288,
            "two IQ4_XS blocks, padded to the alignment"
        );
        assert_eq!(
            ours.content.metadata["general.architecture"]
                .to_string()
                .unwrap(),
            "qwen3"
        );
    }

    /// Contract: a header that stops short is an error, not a partial read.
    #[test]
    fn a_truncated_header_is_refused() {
        let kv = [("general.architecture", Value::String("llama".into()))];
        let tensors: [(&str, &[u64], u32); 1] = [("output_norm.weight", &[8], 0)];
        let bytes = gguf_bytes(&kv, &tensors, &[0u8; 32]);
        let header_len = 4 + 4 + 8 + 8 + (8 + 20 + 4 + 8 + 5) + (8 + 18 + 4 + 8 + 4 + 8);
        for cut in [3usize, 12, 40, header_len - 1] {
            assert!(GgufHeader::parse(&bytes[..cut]).is_err(), "cut at {cut}");
        }
        assert!(GgufHeader::parse(&bytes[..header_len]).is_ok());
    }
}
