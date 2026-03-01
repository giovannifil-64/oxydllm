use crate::model::common::paged::PagedKvCache;
use crate::sampling::SamplingParams;

pub type SequenceId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequencePhase {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
    Waiting,
    Running,
    Finished,
}

pub struct SequenceState {
    pub id: SequenceId,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub all_tokens: Vec<u32>,
    pub sampling_params: SamplingParams,
    pub status: SequenceStatus,
    pub phase: SequencePhase,
    pub caches: Vec<PagedKvCache>,
    pub num_processed_tokens: usize,
    pub max_tokens: usize,
}
