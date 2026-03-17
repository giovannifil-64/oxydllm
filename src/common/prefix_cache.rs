use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;

use lru::LruCache;
use rustc_hash::FxHasher;

use crate::common::paged::SharedBlockAllocator;

struct PrefixEntry {
    block_ids: Vec<usize>,
    block_tokens: Vec<u32>,
}

pub struct PrefixCache {
    entries: LruCache<u64, PrefixEntry>,
}

impl PrefixCache {
    pub fn new(capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity.max(1)).unwrap();
        Self { entries: LruCache::new(cap) }
    }

    pub fn lookup(&mut self, tokens: &[u32], block_size: usize) -> (usize, Vec<Vec<usize>>) {
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

    pub fn register(
        &mut self,
        tokens: &[u32],
        start_block: usize,
        new_block_ids: &[Vec<usize>],
        allocators: &[SharedBlockAllocator],
        block_size: usize,
    ) {
        let num_full_blocks = tokens.len() / block_size;
        debug_assert!(start_block + new_block_ids.len() <= num_full_blocks);

        let mut prev_hash: u64 = 0;
        for block_idx in 0..start_block {
            prev_hash = chain_hash(
                &tokens[block_idx * block_size..(block_idx + 1) * block_size],
                prev_hash,
            );
        }

        for (i, block_ids) in new_block_ids.iter().enumerate() {
            if block_ids.len() != allocators.len() {
                eprintln!(
                    "[prefix_cache] register: block_ids.len()={} != allocators.len()={} at block {}, skipping remaining",
                    block_ids.len(), allocators.len(), start_block + i,
                );
                break;
            }

            let block_idx = start_block + i;
            let h = chain_hash(
                &tokens[block_idx * block_size..(block_idx + 1) * block_size],
                prev_hash,
            );

            let block_tokens = tokens[block_idx * block_size..(block_idx + 1) * block_size].to_vec();
            if !self.entries.contains(&h) {
                for (layer_idx, &bid) in block_ids.iter().enumerate() {
                    allocators[layer_idx].lock().unwrap().share(bid);
                }

                if let Some((_, evicted)) = self.entries.push(h, PrefixEntry { block_ids: block_ids.clone(), block_tokens }) {
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

fn chain_hash(block_tokens: &[u32], prev: u64) -> u64 {
    let mut h = FxHasher::default();
    prev.hash(&mut h);
    block_tokens.hash(&mut h);
    h.finish()
}
