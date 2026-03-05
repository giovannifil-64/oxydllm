use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Instant;

use crate::common::paged::SharedBlockAllocator;

struct PrefixEntry {
    block_ids: Vec<usize>,
    last_used: Instant,
}

pub struct PrefixCache {
    entries: HashMap<u64, PrefixEntry>,
    capacity: usize,
}

impl PrefixCache {
    pub fn new(capacity: usize) -> Self {
        Self { entries: HashMap::new(), capacity }
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
            match self.entries.get_mut(&h) {
                Some(entry) => {
                    entry.last_used = Instant::now();
                    matched.push(entry.block_ids.clone());
                    prev_hash = h;
                }
                None => break,
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
                break;
            }

            let block_idx = start_block + i;
            let h = chain_hash(
                &tokens[block_idx * block_size..(block_idx + 1) * block_size],
                prev_hash,
            );

            if !self.entries.contains_key(&h) {
                if self.entries.len() >= self.capacity {
                    self.evict_lru(allocators);
                }

                for (layer_idx, &bid) in block_ids.iter().enumerate() {
                    allocators[layer_idx].lock().unwrap().share(bid);
                }

                self.entries.insert(
                    h,
                    PrefixEntry { block_ids: block_ids.clone(), last_used: Instant::now() },
                );
            }

            prev_hash = h;
        }
    }

    fn evict_lru(&mut self, allocators: &[SharedBlockAllocator]) {
        let lru_hash = self
            .entries
            .iter()
            .min_by_key(|(_, e)| e.last_used)
            .map(|(&h, _)| h);

        if let Some(h) = lru_hash {
            if let Some(entry) = self.entries.remove(&h) {
                for (layer_idx, bid) in entry.block_ids.iter().enumerate() {
                    if layer_idx < allocators.len() {
                        allocators[layer_idx].lock().unwrap().free(*bid);
                    }
                }
            }
        }
    }

}

fn chain_hash(block_tokens: &[u32], prev: u64) -> u64 {
    let mut h = DefaultHasher::new();
    prev.hash(&mut h);
    block_tokens.hash(&mut h);
    h.finish()
}
