use crate::common::paged::PagedKvCache;
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
    #[allow(dead_code)]
    pub prompt_len: usize,
    pub num_generated: usize,
    pub all_tokens: Vec<u32>,
    pub sampling_params: SamplingParams,
    pub status: SequenceStatus,
    pub phase: SequencePhase,
    pub caches: Vec<PagedKvCache>,
    pub num_processed_tokens: usize,
    pub max_tokens: usize,
    pub finish_reason: Option<String>,
}

impl SequenceState {
    #[allow(dead_code)]
    pub fn generated_tokens(&self) -> &[u32] {
        &self.all_tokens[self.prompt_len..]
    }

    pub fn tokens_and_caches(&mut self) -> (&[u32], &SamplingParams, &mut Vec<PagedKvCache>) {
        (&self.all_tokens, &self.sampling_params, &mut self.caches)
    }

    pub fn apply_token(&mut self, _next_token: u32, is_stop: bool) -> bool {
        if is_stop {
            self.status = SequenceStatus::Finished;
            self.finish_reason = Some("stop".to_string());
            false
        } else {
            self.num_generated += 1;
            if self.num_generated >= self.max_tokens {
                self.status = SequenceStatus::Finished;
                self.finish_reason = Some("length".to_string());
            }
            true
        }
    }
}
