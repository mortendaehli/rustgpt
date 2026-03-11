use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct StageSummary {
    pub calls: u64,
    pub total: Duration,
}

#[derive(Debug, Default)]
pub struct RuntimeProfile {
    stages: RefCell<BTreeMap<&'static str, StageSummary>>,
}

impl RuntimeProfile {
    pub fn record(&self, stage: &'static str, elapsed: Duration) {
        let mut stages = self.stages.borrow_mut();
        let entry = stages.entry(stage).or_default();
        entry.calls += 1;
        entry.total += elapsed;
    }

    pub fn merge_from(&self, other: &RuntimeProfile) {
        for (stage, summary) in other.snapshot() {
            let mut stages = self.stages.borrow_mut();
            let entry = stages.entry(stage).or_default();
            entry.calls += summary.calls;
            entry.total += summary.total;
        }
    }

    pub fn snapshot(&self) -> Vec<(&'static str, StageSummary)> {
        self.stages
            .borrow()
            .iter()
            .map(|(stage, summary)| (*stage, summary.clone()))
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.stages.borrow().is_empty()
    }

    pub fn format_table(&self) -> String {
        let mut lines = Vec::new();
        for (stage, summary) in self.snapshot() {
            let avg = if summary.calls == 0 {
                Duration::ZERO
            } else {
                Duration::from_secs_f64(summary.total.as_secs_f64() / summary.calls as f64)
            };
            lines.push(format!(
                "{stage:<26} calls={:<6} total={:<10.4?} avg={avg:.4?}",
                summary.calls, summary.total
            ));
        }
        lines.join("\n")
    }
}

pub fn measure<T, F>(profile: Option<&RuntimeProfile>, stage: &'static str, work: F) -> T
where
    F: FnOnce() -> T,
{
    let started_at = Instant::now();
    let out = work();
    if let Some(profile) = profile {
        profile.record(stage, started_at.elapsed());
    }
    out
}
