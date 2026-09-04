//! Prefix KV-cache reuse: share already-computed KV blocks across requests that
//! begin with a common prompt prefix.
//!
//! Prompts are hashed in fixed-size token blocks with a rolling (chain) hash, so
//! a request can reuse the cached KV blocks for its longest matching prefix
//! instead of recomputing them. [`PrefixCache`] is an LRU over those block
//! hashes; reused blocks are reference-counted in the shared
//! [`crate::common::paged::SharedBlockAllocator`] so they live until evicted.

use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;

use lru::LruCache;
use rustc_hash::FxHasher;

use crate::common::paged::SharedBlockAllocator;

/// One cached prefix block: the KV block id per layer, plus the exact tokens it
/// holds (kept to reject hash collisions on lookup).
struct PrefixEntry {
    block_ids: Vec<usize>,
    block_tokens: Vec<u32>,
}

/// An LRU cache mapping prompt-prefix block hashes to the KV blocks holding them.
///
/// Each full block of `block_size` tokens is keyed by a chain hash over its
/// tokens and the previous block's hash, so a lookup walks the longest matching
/// prefix. Recurrent (linear-attention) models cannot skip prefix tokens (their
/// state must observe every one), so the engine builds a
/// [`disabled`](Self::disabled) cache for those, where every operation is a
/// no-op.
pub struct PrefixCache {
    entries: LruCache<u64, PrefixEntry>,
    disabled: bool,
}

impl PrefixCache {
    /// Creates a prefix cache holding up to `capacity` block entries (LRU).
    pub fn new(capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity.max(1)).unwrap();
        Self {
            entries: LruCache::new(cap),
            disabled: false,
        }
    }

    /// Creates a permanently empty cache for models that cannot reuse prefixes
    /// (recurrent / linear-attention); every method short-circuits.
    pub fn disabled() -> Self {
        Self {
            entries: LruCache::new(NonZeroUsize::new(1).unwrap()),
            disabled: true,
        }
    }

    /// Counts how many leading `block_size`-token blocks of `tokens` are already
    /// cached, without disturbing LRU recency (a peek). Returns 0 when disabled.
    pub fn count_cached_blocks(&self, tokens: &[u32], block_size: usize) -> usize {
        if self.disabled {
            return 0;
        }
        let num_full_blocks = tokens.len() / block_size;
        let mut prev_hash: u64 = 0;
        let mut count = 0;

        for block_idx in 0..num_full_blocks {
            let block_tokens = &tokens[block_idx * block_size..(block_idx + 1) * block_size];
            let h = chain_hash(block_tokens, prev_hash);
            match self.entries.peek(&h) {
                Some(entry) if entry.block_tokens == block_tokens => {
                    count += 1;
                    prev_hash = h;
                }
                _ => break,
            }
        }

        count
    }

    /// Looks up the longest cached prefix of `tokens`, returning the number of
    /// matched blocks and their KV block ids (one `Vec<usize>` per block, holding
    /// one block id per layer). Refreshes LRU recency for the hits. Returns
    /// `(0, [])` when disabled.
    pub fn lookup(&mut self, tokens: &[u32], block_size: usize) -> (usize, Vec<Vec<usize>>) {
        if self.disabled {
            return (0, Vec::new());
        }
        let num_full_blocks = tokens.len() / block_size;
        let mut prev_hash: u64 = 0;
        let mut matched: Vec<Vec<usize>> = Vec::new();

        for block_idx in 0..num_full_blocks {
            let h = chain_hash(
                &tokens[block_idx * block_size..(block_idx + 1) * block_size],
                prev_hash,
            );
            let block_tokens = &tokens[block_idx * block_size..(block_idx + 1) * block_size];
            match self.entries.get(&h) {
                Some(entry) if entry.block_tokens == block_tokens => {
                    matched.push(entry.block_ids.clone());
                    prev_hash = h;
                }
                _ => break,
            }
        }

        let n = matched.len();
        (n, matched)
    }

    /// Records the KV blocks of freshly computed prefix blocks so later requests
    /// can reuse them.
    ///
    /// `new_block_ids[i]` holds one block id per layer for the block at
    /// `start_block + i`. Each newly cached block is reference-counted via
    /// `share` on the per-layer `allocators`, and any LRU-evicted entry's blocks
    /// are freed. No-op when disabled, or when a block's id count does not match
    /// `allocators.len()`.
    pub fn register(
        &mut self,
        tokens: &[u32],
        start_block: usize,
        new_block_ids: &[Vec<usize>],
        allocators: &[SharedBlockAllocator],
        block_size: usize,
    ) {
        if self.disabled {
            return;
        }
        let num_full_blocks = tokens.len() / block_size;
        debug_assert!(start_block + new_block_ids.len() <= num_full_blocks);

        if new_block_ids
            .iter()
            .any(|ids| ids.len() != allocators.len())
        {
            return;
        }

        let mut prev_hash: u64 = 0;
        for block_idx in 0..start_block {
            prev_hash = chain_hash(
                &tokens[block_idx * block_size..(block_idx + 1) * block_size],
                prev_hash,
            );
        }

        for (i, block_ids) in new_block_ids.iter().enumerate() {
            let block_idx = start_block + i;
            let h = chain_hash(
                &tokens[block_idx * block_size..(block_idx + 1) * block_size],
                prev_hash,
            );

            let block_tokens =
                tokens[block_idx * block_size..(block_idx + 1) * block_size].to_vec();
            if !self.entries.contains(&h) {
                for (layer_idx, &bid) in block_ids.iter().enumerate() {
                    allocators[layer_idx].lock().unwrap().share(bid);
                }

                if let Some((_, evicted)) = self.entries.push(
                    h,
                    PrefixEntry {
                        block_ids: block_ids.clone(),
                        block_tokens,
                    },
                ) {
                    for (layer_idx, bid) in evicted.block_ids.iter().enumerate() {
                        if layer_idx < allocators.len() {
                            allocators[layer_idx].lock().unwrap().free(*bid);
                        }
                    }
                }
            }

            prev_hash = h;
        }
    }
}

/// Rolling hash chaining a block's tokens onto the previous block's hash, so a
/// hash identifies a block only in the context of its full prefix.
fn chain_hash(block_tokens: &[u32], prev: u64) -> u64 {
    let mut h = FxHasher::default();
    prev.hash(&mut h);
    block_tokens.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::paged::{BlockAllocator, DEFAULT_BLOCK_SIZE};
    use candle_core::{DType, Device};
    use std::sync::{Arc, Mutex};

    /// A pseudo-random source with the seed in hand, so a failure reproduces by
    /// rerunning the same seed.
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed.wrapping_mul(6364136223846793005).wrapping_add(1))
        }

        fn between(&mut self, lo: usize, hi: usize) -> usize {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            lo + ((self.0 >> 33) as usize) % (hi - lo + 1)
        }
    }

    fn allocator(blocks: usize) -> Vec<SharedBlockAllocator> {
        vec![Arc::new(Mutex::new(
            BlockAllocator::growing(
                blocks,
                blocks,
                DEFAULT_BLOCK_SIZE,
                1,
                4,
                DType::F32,
                &Device::Cpu,
                None,
            )
            .unwrap(),
        ))]
    }

    /// Contract (soundness): a lookup only ever returns blocks that hold exactly
    /// the tokens it asked about.
    ///
    /// The failure this guards against is not a crash. Handing a sequence a
    /// block computed from different tokens splices another prompt's attention
    /// into it, and the model answers fluently about something nobody asked.
    /// The tokens are generated from a deliberately tiny alphabet so prefixes
    /// repeat and near-misses are common, which is where a chain hash compared
    /// without its tokens would give itself away.
    #[test]
    fn fuzz_lookup_never_returns_another_prompt_s_blocks() {
        const BS: usize = 4;
        for seed in 0u64..64 {
            let mut rng = Rng::new(seed);
            let allocators = allocator(4096);
            let mut cache = PrefixCache::new(rng.between(1, 32));
            // What a correct cache could return: the ids registered for each
            // exact token prefix, first registration winning as the cache does.
            let mut truth: std::collections::HashMap<Vec<u32>, Vec<usize>> =
                std::collections::HashMap::new();

            for _ in 0..rng.between(1, 8) {
                let blocks = rng.between(1, 5);
                let tokens: Vec<u32> = (0..blocks * BS).map(|_| rng.between(0, 2) as u32).collect();
                let ids: Vec<Vec<usize>> = (0..blocks)
                    .map(|_| vec![allocators[0].lock().unwrap().allocate().unwrap()])
                    .collect();

                cache.register(&tokens, 0, &ids, &allocators, BS);
                for (i, id) in ids.iter().enumerate() {
                    truth
                        .entry(tokens[..(i + 1) * BS].to_vec())
                        .or_insert_with(|| id.clone());
                }

                let probe: Vec<u32> = (0..rng.between(1, 6) * BS)
                    .map(|_| rng.between(0, 2) as u32)
                    .collect();
                let (n, matched) = cache.lookup(&probe, BS);
                assert_eq!(n, matched.len());
                for (i, got) in matched.iter().enumerate() {
                    let prefix = probe[..(i + 1) * BS].to_vec();
                    let expected = truth.get(&prefix).unwrap_or_else(|| {
                        panic!(
                            "seed {seed}: matched block {i} for a prefix that was never registered: {prefix:?}"
                        )
                    });
                    assert_eq!(
                        got, expected,
                        "seed {seed}: block {i} of {probe:?} came from another prompt"
                    );
                }
            }
        }
    }

    /// Contract: counting and looking up agree, so the scheduler's block
    /// arithmetic and the engine's reuse cannot disagree about how much of a
    /// prompt is already computed. The count is a peek, so it must not change
    /// what a following lookup finds either.
    #[test]
    fn fuzz_counting_agrees_with_looking_up() {
        const BS: usize = 4;
        for seed in 0u64..64 {
            let mut rng = Rng::new(seed ^ 0x5eed_c0de);
            let allocators = allocator(4096);
            let mut cache = PrefixCache::new(rng.between(1, 32));

            for _ in 0..rng.between(1, 8) {
                let blocks = rng.between(1, 5);
                let tokens: Vec<u32> = (0..blocks * BS).map(|_| rng.between(0, 2) as u32).collect();
                let ids: Vec<Vec<usize>> = (0..blocks)
                    .map(|_| vec![allocators[0].lock().unwrap().allocate().unwrap()])
                    .collect();
                cache.register(&tokens, 0, &ids, &allocators, BS);

                let probe: Vec<u32> = (0..rng.between(1, 6) * BS)
                    .map(|_| rng.between(0, 2) as u32)
                    .collect();
                let counted = cache.count_cached_blocks(&probe, BS);
                let (looked_up, _) = cache.lookup(&probe, BS);
                assert_eq!(
                    counted, looked_up,
                    "seed {seed}: counted {counted} blocks but looked up {looked_up}"
                );
            }
        }
    }
}
