use candle_core::{Device, Result, Tensor};

use crate::model::traits::BatchModel;
use crate::sampling::{self, SamplingParams};
use crate::scheduler::sequence::*;
use crate::scheduler::*;

pub struct NewToken {
    pub seq_id: SequenceId,
    pub token: u32,
}

pub struct StepOutput {
    pub new_tokens: Vec<NewToken>,
    pub completed: Vec<CompletedSequence>,
}

pub struct Engine {
    model: Box<dyn BatchModel>,
    scheduler: Scheduler,
    device: Device,
}

impl Engine {
    pub fn new(model: Box<dyn BatchModel>, config: SchedulerConfig) -> Self {
        let allocators = model.allocators().iter().map(|a| std::rc::Rc::clone(a)).collect();
        let num_layers = model.num_layers();
        let device = model.device().clone();
        let scheduler = Scheduler::new(config, allocators, num_layers);
        Self { model, scheduler, device }
    }

    pub fn add_request(
        &mut self,
        prompt_tokens: Vec<u32>,
        sampling_params: SamplingParams,
        max_tokens: usize,
    ) -> SequenceId {
        self.scheduler.add_request(prompt_tokens, sampling_params, max_tokens)
    }

    pub fn step(&mut self) -> Result<StepOutput> {
        let output = self.scheduler.schedule();
        let mut new_tokens = Vec::new();

        for sched_seq in &output.scheduled {
            let seq = self.scheduler.get_running_mut(sched_seq.id).unwrap();

            let (input_tokens, start_pos) = match sched_seq.phase {
                SequencePhase::Prefill => {
                    let tokens = seq.all_tokens.clone();
                    (tokens, 0usize)
                }
                SequencePhase::Decode => {
                    let last = *seq.all_tokens.last().unwrap();
                    (vec![last], seq.num_processed_tokens)
                }
            };

            let seq_len = input_tokens.len();
            let input = Tensor::from_vec(input_tokens, (1, seq_len), &self.device)?;
            let logits = self.model.forward_with_cache(&input, start_pos, &mut seq.caches)?;
            let last_logits = logits.squeeze(0)?.get(seq_len - 1)?;

            let next_token = sampling::sample(&last_logits, &seq.sampling_params, &seq.all_tokens)?;

            let is_eos = next_token == self.model.eos_token_id();

            seq.all_tokens.push(next_token);
            seq.num_processed_tokens = seq.all_tokens.len() - 1;
            seq.phase = SequencePhase::Decode;

            if is_eos {
                seq.status = SequenceStatus::Finished;
                seq.finish_reason = Some("stop".to_string());
            } else {
                seq.generated_tokens.push(next_token);
                new_tokens.push(NewToken { seq_id: seq.id, token: next_token });
                if seq.generated_tokens.len() >= seq.max_tokens {
                    seq.status = SequenceStatus::Finished;
                    seq.finish_reason = Some("length".to_string());
                }
            }
        }

        let completed = self.scheduler.retire_finished();
        Ok(StepOutput { new_tokens, completed })
    }

    pub fn has_pending_work(&self) -> bool {
        self.scheduler.has_pending_work()
    }

    pub fn run_to_completion(&mut self) -> Result<Vec<CompletedSequence>> {
        let mut all_completed = Vec::new();
        while self.has_pending_work() {
            let step = self.step()?;
            all_completed.extend(step.completed);
        }
        Ok(all_completed)
    }
}
