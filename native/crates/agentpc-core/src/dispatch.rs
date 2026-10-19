//! The play-loop brain: routes demuxed agent actions to per-box arbiters, and applies the winner to
//! each box's sink — including the handoff-release that keeps a departed agent's keys from lingering.
//!
//! Arbitration is **per box** (per window): each box has its own driver contest, because the human
//! can focus one window while agents drive others. A [`Dispatcher`] holds many [`BoxRuntime`]s and
//! knows which box each agent is assigned to.
//!
//! Sinks are **per box**, not per agent: the Arbiter guarantees only one agent drives a given window
//! at a time, so a box needs a single keyboard sink (its window) or a single virtual pad (its
//! target) — whichever the current driver uses. Different boxes get different sinks.
//!
//! This module is pure logic over the [`Sink`] trait (no OS calls), so the whole join is unit-tested
//! with a mock sink.

use std::collections::BTreeMap;

use crate::{Action, AgentId, Arbiter, Decision, Effective, PriorityRule, Provenance, Sink, SourceId};

/// One box: a window/monitor, its arbiter, and its sink. Advanced one tick at a time.
pub struct BoxRuntime {
    pub source: SourceId,
    arbiter: Arbiter,
    sink: Box<dyn Sink + Send>,
    last_driver: Option<Provenance>,
}

impl BoxRuntime {
    pub fn new(source: SourceId, sink: Box<dyn Sink + Send>, rule: PriorityRule) -> Self {
        Self { source, arbiter: Arbiter::new(rule), sink, last_driver: None }
    }

    fn submit_agent(&mut self, agent: AgentId, action: Action, now_us: u64) {
        self.arbiter.ensure_in_order(Provenance::Agent(agent));
        self.arbiter.submit(Provenance::Agent(agent), action, now_us);
    }

    fn submit_human(&mut self, action: Action, now_us: u64) {
        self.arbiter.submit(Provenance::Human, action, now_us);
    }

    /// Resolve the driver and act: release a departed agent, inject the current agent winner (never
    /// the human — already applied by the OS). Returns the [`Decision`] for the recorder.
    fn tick(&mut self, now_us: u64) -> Decision {
        let d = self.arbiter.resolve(now_us);
        let cur = d.effective.as_ref().map(|e| e.source);

        // Handoff: the previous driver was an agent that is no longer driving → release its held
        // inputs so nothing lingers (the injector half of the Arbiter contract).
        if let Some(Provenance::Agent(a)) = self.last_driver {
            if cur != Some(Provenance::Agent(a)) {
                let _ = self.sink.release(a);
            }
        }

        // Inject the current driver iff it is an agent.
        if let Some(e) = &d.effective {
            if e.inject {
                if let Provenance::Agent(a) = e.source {
                    let _ = self.sink.apply(a, &e.action);
                }
            }
        }

        self.last_driver = cur;
        d
    }
}

/// Owns every box and the agent→box map. The transport calls [`Dispatcher::on_action`] for each
/// demuxed packet and [`Dispatcher::on_human`] for the human tap; a frame-cadence caller invokes
/// [`Dispatcher::tick`].
pub struct Dispatcher {
    boxes: BTreeMap<SourceId, BoxRuntime>,
    agent_box: BTreeMap<AgentId, SourceId>,
}

impl Dispatcher {
    pub fn new() -> Self {
        Self { boxes: BTreeMap::new(), agent_box: BTreeMap::new() }
    }

    /// Register a box (a capture source) with its sink and priority rule.
    pub fn add_box(&mut self, source: SourceId, sink: Box<dyn Sink + Send>, rule: PriorityRule) {
        self.boxes.insert(source, BoxRuntime::new(source, sink, rule));
    }

    /// Bind an agent to a box; it becomes a contender in that box's arbiter (lowest priority).
    pub fn assign_agent(&mut self, agent: AgentId, source: SourceId) {
        self.agent_box.insert(agent, source);
        if let Some(b) = self.boxes.get_mut(&source) {
            b.arbiter.ensure_in_order(Provenance::Agent(agent));
        }
    }

    /// Route an agent's action (demuxed off the wire, keyed by agent id) to its box.
    pub fn on_action(&mut self, agent: AgentId, action: Action, now_us: u64) {
        if let Some(&source) = self.agent_box.get(&agent) {
            if let Some(b) = self.boxes.get_mut(&source) {
                b.submit_agent(agent, action, now_us);
            }
        }
    }

    /// The human is inputting on a window (its box) — captured raw input, resampled to frames.
    pub fn on_human(&mut self, source: SourceId, action: Action, now_us: u64) {
        if let Some(b) = self.boxes.get_mut(&source) {
            b.submit_human(action, now_us);
        }
    }

    /// Read-only resolve of one box's current effective driver — no injection, no state change. The
    /// record path uses this per captured frame to tag the frame with the action that drove it.
    pub fn peek(&self, source: SourceId, now_us: u64) -> Option<Effective> {
        self.boxes.get(&source).and_then(|b| b.arbiter.resolve(now_us).effective)
    }

    /// Advance every box one frame. Returns `(source, Decision)` per box for the recorder — the
    /// effective action + provenance that actually drove each window (human OR agent).
    pub fn tick(&mut self, now_us: u64) -> Vec<(SourceId, Decision)> {
        self.boxes.iter_mut().map(|(s, b)| (*s, b.tick(now_us))).collect()
    }

    /// Pause/unpause an agent's effect (instant; inference/capture keep running).
    pub fn set_paused(&mut self, agent: AgentId, paused: bool) {
        if let Some(&source) = self.agent_box.get(&agent) {
            if let Some(b) = self.boxes.get_mut(&source) {
                b.arbiter.set_paused(Provenance::Agent(agent), paused);
            }
        }
    }
}

impl Default for Dispatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// Records apply/release calls so tests can assert what actually reached the sink.
    #[derive(Clone)]
    struct MockSink {
        log: Arc<Mutex<Vec<String>>>,
    }
    impl MockSink {
        fn new() -> Self {
            Self { log: Arc::new(Mutex::new(Vec::new())) }
        }
    }
    impl Sink for MockSink {
        fn apply(&mut self, agent: AgentId, action: &Action) -> Result<(), crate::SinkError> {
            self.log.lock().unwrap().push(format!("apply:{agent}:{:?}", action.keys_held));
            Ok(())
        }
        fn release(&mut self, agent: AgentId) -> Result<(), crate::SinkError> {
            self.log.lock().unwrap().push(format!("release:{agent}"));
            Ok(())
        }
    }

    fn held(k: &str) -> Action {
        Action { keys_held: vec![k.to_string()], ..Default::default() }
    }

    #[test]
    fn agent_action_reaches_its_box_sink() {
        let sink = MockSink::new();
        let log = sink.log.clone();
        let mut d = Dispatcher::new();
        d.add_box(100, Box::new(sink), PriorityRule::default());
        d.assign_agent(7, 100);

        d.on_action(7, held("W"), 1_000);
        d.tick(1_000);
        assert_eq!(&*log.lock().unwrap(), &["apply:7:[\"W\"]"]);
    }

    #[test]
    fn human_preempt_releases_the_agent_then_stops_injecting() {
        let sink = MockSink::new();
        let log = sink.log.clone();
        let mut d = Dispatcher::new();
        d.add_box(100, Box::new(sink), PriorityRule::default()); // order starts [Human]
        d.assign_agent(7, 100); // → [Human, Agent(7)]

        // Agent drives first.
        d.on_action(7, held("W"), 1_000_000);
        d.tick(1_000_000);

        // Human presses: it wins (not injected), and the agent is released so W doesn't linger.
        d.on_human(100, held("S"), 1_000_000);
        d.on_action(7, held("W"), 1_000_000);
        d.tick(1_000_000);

        assert_eq!(&*log.lock().unwrap(), &["apply:7:[\"W\"]", "release:7"]);
    }

    #[test]
    fn two_boxes_run_independently() {
        let s1 = MockSink::new();
        let s2 = MockSink::new();
        let l1 = s1.log.clone();
        let l2 = s2.log.clone();
        let mut d = Dispatcher::new();
        d.add_box(100, Box::new(s1), PriorityRule::default());
        d.add_box(200, Box::new(s2), PriorityRule::default());
        d.assign_agent(7, 100);
        d.assign_agent(8, 200);

        d.on_action(7, held("W"), 1_000);
        d.on_action(8, held("X"), 1_000);
        d.tick(1_000);

        assert_eq!(&*l1.lock().unwrap(), &["apply:7:[\"W\"]"]);
        assert_eq!(&*l2.lock().unwrap(), &["apply:8:[\"X\"]"]);
    }

    #[test]
    fn paused_agent_does_not_drive() {
        let sink = MockSink::new();
        let log = sink.log.clone();
        let mut d = Dispatcher::new();
        d.add_box(100, Box::new(sink), PriorityRule::default());
        d.assign_agent(7, 100);

        d.set_paused(7, true);
        d.on_action(7, held("W"), 1_000);
        d.tick(1_000);
        assert!(log.lock().unwrap().is_empty(), "paused agent must not reach the sink");
    }
}
