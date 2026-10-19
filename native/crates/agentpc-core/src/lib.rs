//! agentpc-core — the I/O abstractions every other crate agrees on.
//!
//! Designed so a **policy can be RL or LLM** and a **driver can be human or agent** without any
//! downstream code caring which. The two load-bearing ideas:
//!
//! 1. Actions are **level/state based** (held keys + axis values), not edge events. The injector
//!    derives down/up edges. A 2 Hz LLM and a 20 Hz RL net share one path because the last state
//!    *holds* until replaced. See [`Action`].
//! 2. The [`Arbiter`] is the single place human/agent inputs are prioritized, and its output is what
//!    the *recorder* reads — so an agent's own inputs land in the shards during autonomous capture.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

mod dispatch;
pub use dispatch::{BoxRuntime, Dispatcher};

mod vocab;
pub use vocab::token_for_vk;

/// Process-wide monotonic microsecond clock. Every producer of a timeline value — the play-loop
/// tick, the recorder's `t_us`, and the human tap's `on_human` submits — reads from this one epoch.
/// The Arbiter compares a source's submit time against the tick's `now_us` to decide liveness
/// (human idle-return, agent-stale), so those two readings MUST come from the same clock; a tap on
/// its own `Instant` origin would make every human-idle window meaningless. `OnceLock` fixes the
/// epoch at first call (process start, for practical purposes) with no unsafe and no init ordering.
pub fn now_us() -> u64 {
    use std::sync::OnceLock;
    use std::time::Instant;
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    EPOCH.get_or_init(Instant::now).elapsed().as_micros() as u64
}

/// Stable id for a capture source (a box's window/monitor). 0 is reserved/invalid.
pub type SourceId = u32;
/// Stable id for a policy instance (a sprite). 0 is reserved/invalid.
pub type AgentId = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputDevice { Keyboard, Mouse, Gamepad }

/// A physical input observation before arbitration. Keyboard/mouse entries originate at WM_INPUT
/// cadence; gamepad entries originate when the quantized XInput state changes.
#[derive(Clone, Debug, PartialEq)]
pub struct RawInputEvent {
    pub source_id: SourceId,
    pub t_us: u64,
    pub device: InputDevice,
    pub action: Action,
}

/// Bounded bridge from the system-wide RawInput thread to a session recorder. Overflow is counted
/// explicitly so the trajectory can contain a gap marker rather than silently claiming continuity.
pub struct RawInputBuffer {
    capacity: usize,
    queue: Mutex<VecDeque<RawInputEvent>>,
    dropped: AtomicU64,
}

impl RawInputBuffer {
    pub fn new(capacity: usize) -> Self {
        Self { capacity: capacity.max(1), queue: Mutex::new(VecDeque::new()), dropped: AtomicU64::new(0) }
    }
    pub fn push(&self, event: RawInputEvent) {
        let mut queue = self.queue.lock().unwrap();
        if queue.len() == self.capacity {
            queue.pop_front();
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
        queue.push_back(event);
    }
    pub fn drain(&self) -> (Vec<RawInputEvent>, u64) {
        let events = self.queue.lock().unwrap().drain(..).collect();
        let dropped = self.dropped.swap(0, Ordering::Relaxed);
        (events, dropped)
    }
}

#[cfg(test)]
mod raw_input_tests {
    use super::*;
    #[test]
    fn bounded_input_buffer_reports_and_replaces_oldest() {
        let buffer = RawInputBuffer::new(2);
        for t_us in 1..=3 {
            buffer.push(RawInputEvent { source_id: 9, t_us, device: InputDevice::Keyboard, action: Action::default() });
        }
        let (events, dropped) = buffer.drain();
        assert_eq!(dropped, 1);
        assert_eq!(events.iter().map(|e| e.t_us).collect::<Vec<_>>(), [2, 3]);
        assert_eq!(buffer.drain().1, 0);
    }
}

#[cfg(test)]
mod agent_profile_tests {
    use super::*;

    #[test]
    fn legacy_agent_payload_defaults_profile() {
        let json = r#"{"id":7,"name":"OLD","species":137,"description":"legacy","host":"cradle"}"#;
        let agent: AgentDef = serde_json::from_str(json).unwrap();
        assert_eq!(agent.profile, AgentProfile::default());
    }

    #[test]
    fn structured_profile_round_trips() {
        let agent = AgentDef {
            id: 31,
            name: "FIXTURE-A".into(),
            species: 25,
            description: "fixture".into(),
            host: "cradle".into(),
            profile: AgentProfile {
                kind: "fixture".into(),
                compatible_targets: vec!["AgentPC Fixture LIVE-A".into()],
                limitations: vec!["Not trained".into()],
                ..Default::default()
            },
        };
        let decoded: AgentDef = serde_json::from_str(&serde_json::to_string(&agent).unwrap()).unwrap();
        assert_eq!(decoded, agent);
    }
}

/// A model/policy registered on the control plane (agentpc-server). The desktop lists these instead
/// of hardcoding an agent: each becomes a placeable pokébox sprite. `id` is the wire agent id the
/// cradle answers on; `species` is the national-dex number for the sprite icon.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AgentDef {
    pub id: AgentId,
    pub name: String,
    pub species: u32,
    #[serde(default)]
    pub description: String,
    /// The cradle that hosts this policy (its advertised name/host). The desktop names the box that
    /// holds a cradle's agents after this, and groups agents from different cradles into their own
    /// boxes. Empty when an older cradle didn't advertise it.
    #[serde(default)]
    pub host: String,
    /// Authoritative capability metadata supplied by the host. Defaults keep older cradle POSTs
    /// and cached registry payloads readable; the desktop must never invent these facts.
    #[serde(default)]
    pub profile: AgentProfile,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentProfile {
    /// `fixture`, `scripted`, `trained`, `llm`, or another host-defined category.
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub algorithm: String,
    #[serde(default)]
    pub checkpoint: String,
    #[serde(default)]
    pub observation: String,
    #[serde(default)]
    pub action_space: Vec<String>,
    #[serde(default)]
    pub input_mapping: Vec<String>,
    #[serde(default)]
    pub compatible_targets: Vec<String>,
    #[serde(default)]
    pub trained_on: Vec<String>,
    #[serde(default)]
    pub limitations: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Who produced an action. Written into the shard as `source` (see docs/SCHEMA.md Addendum A).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Provenance {
    Human,
    Agent(AgentId),
}

impl Provenance {
    /// Flat form for the shard `source` field: `"human"` or `"agent:<id>"` (SCHEMA.md Addendum A).
    pub fn as_source_str(&self) -> String {
        match self {
            Provenance::Human => "human".to_string(),
            Provenance::Agent(id) => format!("agent:{id}"),
        }
    }
    pub fn is_agent(&self) -> bool {
        matches!(self, Provenance::Agent(_))
    }
}

/// A desired input *state* — not events. The injector diffs consecutive states to emit edges.
/// Vocabulary mirrors docs/SCHEMA.md §4/§5 and docs/WIRE.md.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Action {
    /// Microseconds since session `t0` (never wall-clock; see rlrec clock rules).
    pub t_us: u64,
    /// Physical HID keys currently held. Scancodes/vk names only — never resolved characters (§11).
    pub keys_held: Vec<String>,
    pub axes: Axes,
    pub buttons: Buttons,
    pub mouse: Mouse,
}

impl Action {
    /// True when nothing is held / no movement — the "doing nothing" state. Used to detect when a
    /// human has stopped inputting so control can return to an agent (see [`Arbiter`]).
    pub fn is_neutral(&self) -> bool {
        const EPS: f32 = 1e-3;
        self.keys_held.is_empty()
            && self.buttons == Buttons::default()
            && self.mouse.dx == 0
            && self.mouse.dy == 0
            && self.mouse.wheel == 0
            && self.mouse.buttons == 0
            && self.axes.lx.abs() < EPS
            && self.axes.ly.abs() < EPS
            && self.axes.rx.abs() < EPS
            && self.axes.ry.abs() < EPS
            && self.axes.lt.abs() < EPS
            && self.axes.rt.abs() < EPS
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Axes {
    pub lx: f32, pub ly: f32, pub rx: f32, pub ry: f32, // sticks [-1,1]
    pub lt: f32, pub rt: f32,                            // triggers [0,1]
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Buttons {
    pub a: bool, pub b: bool, pub x: bool, pub y: bool,
    pub lb: bool, pub rb: bool, pub back: bool, pub start: bool,
    pub ls: bool, pub rs: bool,
    pub dpad_u: bool, pub dpad_d: bool, pub dpad_l: bool, pub dpad_r: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mouse { pub dx: i32, pub dy: i32, pub buttons: u8, pub wheel: i32 }

/// What a policy sees. Serves RL (frame/tensor) and LLM (frame + text context) alike. The heavy
/// pixel payload is JPEG on the wire in the play loop; this is the logical shape the cradle sees.
#[derive(Clone, Debug)]
pub struct Observation {
    pub source: SourceId,
    pub t_us: u64,
    pub frame_jpeg: Vec<u8>,
    pub context: Option<serde_json::Value>,
}

/// An action consumer that applies state to the world (see agentpc-inject). The sink keeps the last
/// applied state per agent and emits only the **edges** — that's how a slow policy holds keys.
pub trait Sink {
    /// Apply `action` (a level state) for `agent`, deriving edges against the previously applied
    /// state.
    fn apply(&mut self, agent: AgentId, action: &Action) -> Result<(), SinkError>;

    /// Release everything `agent` holds (handoff / stop-driving). Because actions are level-based,
    /// this is just "apply neutral": the edge derivation in [`Sink::apply`] emits the up-edges for
    /// whatever was held. This is the injector side of the Arbiter's handoff contract — when a
    /// driver switches away from an agent, call this so its held keys don't linger.
    fn release(&mut self, agent: AgentId) -> Result<(), SinkError> {
        self.apply(agent, &Action::default())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SinkError {
    #[error("sink unavailable: {0}")]
    Unavailable(String),
}

// ===========================================================================
// Arbiter
// ===========================================================================

/// The priority rule deciding who drives. Human sits on top in hybrid mode with an idle timeout;
/// agents are ordered explicitly. Assigning who-drives is editing this list — the Arbiter reads it
/// live, so a change (or a pause) takes effect on the very next [`Arbiter::resolve`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PriorityRule {
    /// Highest-priority first. Human, if present, is typically index 0.
    pub order: Vec<Provenance>,
    /// After the human's input goes neutral for this long, control returns to the next-priority
    /// agent. The window that makes "I press keys → I override; I stop → they resume" work.
    pub human_idle_return_ms: u64,
    /// Sources whose *effect* is paused: still submitted/recorded-as-shadow, never the driver.
    pub paused: Vec<Provenance>,
}

impl Default for PriorityRule {
    fn default() -> Self {
        Self { order: vec![Provenance::Human], human_idle_return_ms: 400, paused: vec![] }
    }
}

/// The Arbiter's chosen driver for one tick. `inject` is false when the winner is the human — their
/// real keystrokes already reached the focused window, so re-injecting would double them.
#[derive(Clone, Debug, PartialEq)]
pub struct Effective {
    pub action: Action,
    pub source: Provenance,
    pub inject: bool,
}

/// The full outcome of a tick: who drives (if anyone), plus the other live agents (shadows) whose
/// actions can be recorded/evaluated but are not injected.
#[derive(Clone, Debug, Default)]
pub struct Decision {
    pub effective: Option<Effective>,
    pub shadow: Vec<(Provenance, Action)>,
}

const DEFAULT_AGENT_STALE_MS: u64 = 500;

struct SourceState {
    prov: Provenance,
    last: Action,
    last_us: u64,
}

/// The single arbitration point. Sources **submit** their current level-state as it arrives (human
/// raw-input resampled to frames; agent packets off the wire); consumers **resolve** at frame
/// cadence. Injection fires only for an agent winner; the recorder writes the effective action for
/// *every* winner, tagged with provenance — so autonomous agent play is captured, not just human play.
pub struct Arbiter {
    rule: PriorityRule,
    sources: Vec<SourceState>,
    /// Last time the human's submitted action was non-neutral. `None` = human has never acted, so
    /// the human is not a contender (agents drive freely) until real input arrives.
    last_human_activity_us: Option<u64>,
    agent_stale_us: u64,
}

impl Arbiter {
    pub fn new(rule: PriorityRule) -> Self {
        Self {
            rule,
            sources: Vec::new(),
            last_human_activity_us: None,
            agent_stale_us: DEFAULT_AGENT_STALE_MS * 1000,
        }
    }

    /// How long an agent's last action stays "live" before it drops out of contention (a crashed or
    /// disconnected policy should stop driving). Agents emit continuously while driving.
    pub fn with_agent_stale_ms(mut self, ms: u64) -> Self {
        self.agent_stale_us = ms * 1000;
        self
    }

    pub fn rule(&self) -> &PriorityRule {
        &self.rule
    }

    /// Replace the priority rule (reorder drivers, change the idle window, edit pauses). Live.
    pub fn set_rule(&mut self, rule: PriorityRule) {
        self.rule = rule;
    }

    /// Pause/unpause a source's *effect* instantly. Inference and capture keep running; the source
    /// simply stops being eligible to drive.
    pub fn set_paused(&mut self, prov: Provenance, paused: bool) {
        self.rule.paused.retain(|&p| p != prov);
        if paused {
            self.rule.paused.push(prov);
        }
    }

    /// Ensure `prov` participates in the priority order (appended at lowest priority if absent).
    /// Assigning an agent to a box calls this so it becomes a contender.
    pub fn ensure_in_order(&mut self, prov: Provenance) {
        if !self.rule.order.contains(&prov) {
            self.rule.order.push(prov);
        }
    }

    /// A source reports its current desired state. For the human, a non-neutral action refreshes the
    /// idle timer that governs when control returns to an agent.
    pub fn submit(&mut self, prov: Provenance, action: Action, now_us: u64) {
        if prov == Provenance::Human && !action.is_neutral() {
            self.last_human_activity_us = Some(now_us);
        }
        match self.sources.iter_mut().find(|s| s.prov == prov) {
            Some(s) => {
                s.last = action;
                s.last_us = now_us;
            }
            None => self.sources.push(SourceState { prov, last: action, last_us: now_us }),
        }
    }

    /// Drop a source entirely (agent left / window closed).
    pub fn remove(&mut self, prov: Provenance) {
        self.sources.retain(|s| s.prov != prov);
    }

    /// Resolve the effective driver for `now_us`: the highest-priority live, non-paused source.
    pub fn resolve(&self, now_us: u64) -> Decision {
        let mut live: Vec<&SourceState> = self
            .sources
            .iter()
            .filter(|s| !self.is_paused(s.prov))
            .filter(|s| self.is_live(s, now_us))
            .collect();
        // Lowest priority index (earliest in `order`) wins; unlisted sources sort last.
        live.sort_by_key(|s| self.priority_of(s.prov));

        let effective = live.first().map(|w| Effective {
            action: w.last.clone(),
            source: w.prov,
            inject: w.prov.is_agent(), // human already applied by the OS
        });
        let shadow = live
            .iter()
            .skip(1)
            .filter(|s| s.prov.is_agent())
            .map(|s| (s.prov, s.last.clone()))
            .collect();

        Decision { effective, shadow }
    }

    fn is_paused(&self, p: Provenance) -> bool {
        self.rule.paused.contains(&p)
    }

    fn priority_of(&self, p: Provenance) -> usize {
        self.rule.order.iter().position(|&x| x == p).unwrap_or(usize::MAX)
    }

    fn is_live(&self, s: &SourceState, now_us: u64) -> bool {
        match s.prov {
            // Human: live only within the idle window, measured from the last non-neutral input.
            Provenance::Human => match self.last_human_activity_us {
                Some(t) => now_us.saturating_sub(t) <= self.rule.human_idle_return_ms * 1000,
                None => false,
            },
            // Agent: live while it keeps emitting (a driving policy submits every tick).
            Provenance::Agent(_) => now_us.saturating_sub(s.last_us) <= self.agent_stale_us,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn held(key: &str) -> Action {
        Action { keys_held: vec![key.to_string()], ..Default::default() }
    }
    fn neutral() -> Action {
        Action::default()
    }
    fn rule(order: Vec<Provenance>) -> PriorityRule {
        PriorityRule { order, human_idle_return_ms: 400, paused: vec![] }
    }

    const A1: Provenance = Provenance::Agent(1);
    const A2: Provenance = Provenance::Agent(2);
    const H: Provenance = Provenance::Human;

    #[test]
    fn agent_drives_when_human_absent() {
        let mut a = Arbiter::new(rule(vec![H, A1]));
        a.submit(A1, held("W"), 1_000);
        let d = a.resolve(1_000);
        let e = d.effective.unwrap();
        assert_eq!(e.source, A1);
        assert!(e.inject, "agent actions must be injected");
        assert_eq!(e.action.keys_held, vec!["W"]);
    }

    #[test]
    fn human_preempts_agent_and_is_not_injected() {
        let mut a = Arbiter::new(rule(vec![H, A1]));
        a.submit(A1, held("W"), 1_000_000);
        a.submit(H, held("S"), 1_000_000);
        let d = a.resolve(1_000_000);
        let e = d.effective.unwrap();
        assert_eq!(e.source, H);
        assert!(!e.inject, "human input already reached the OS; must NOT be re-injected");
        // The agent is still live — it becomes a shadow for eval.
        assert_eq!(d.shadow, vec![(A1, held("W"))]);
    }

    #[test]
    fn human_yields_to_agent_after_idle_window() {
        let mut a = Arbiter::new(rule(vec![H, A1]));
        // Human acts, agent also live.
        a.submit(H, held("S"), 1_000_000);
        a.submit(A1, held("W"), 1_000_000);
        assert_eq!(a.resolve(1_000_000).effective.unwrap().source, H);

        // 100 ms later the human releases (neutral) but is still within the 400 ms grace window.
        a.submit(H, neutral(), 1_100_000);
        a.submit(A1, held("W"), 1_100_000);
        assert_eq!(a.resolve(1_100_000).effective.unwrap().source, H, "still in grace window");

        // 500 ms after the last non-neutral input: human is idle, control returns to the agent.
        a.submit(A1, held("W"), 1_500_000);
        let e = a.resolve(1_500_000).effective.unwrap();
        assert_eq!(e.source, A1);
        assert!(e.inject);
    }

    #[test]
    fn pause_removes_a_source_from_contention() {
        let mut a = Arbiter::new(rule(vec![A1]));
        a.submit(A1, held("W"), 1_000);
        assert_eq!(a.resolve(1_000).effective.unwrap().source, A1);

        a.set_paused(A1, true);
        assert!(a.resolve(1_000).effective.is_none(), "paused source must not drive");

        a.set_paused(A1, false);
        assert_eq!(a.resolve(1_000).effective.unwrap().source, A1);
    }

    #[test]
    fn priority_orders_two_agents_and_lists_the_shadow() {
        let mut a = Arbiter::new(rule(vec![A1, A2]));
        a.submit(A2, held("X"), 1_000);
        a.submit(A1, held("W"), 1_000);
        let d = a.resolve(1_000);
        assert_eq!(d.effective.unwrap().source, A1, "A1 outranks A2");
        assert_eq!(d.shadow, vec![(A2, held("X"))]);
    }

    #[test]
    fn a_stale_agent_drops_out() {
        let mut a = Arbiter::new(rule(vec![A1])).with_agent_stale_ms(500);
        a.submit(A1, held("W"), 1_000_000);
        assert!(a.resolve(1_400_000).effective.is_some(), "within stale window");
        assert!(a.resolve(1_600_000).effective.is_none(), "past stale window, agent dropped");
    }

    #[test]
    fn provenance_source_string() {
        assert_eq!(Provenance::Human.as_source_str(), "human");
        assert_eq!(Provenance::Agent(7).as_source_str(), "agent:7");
    }
}
