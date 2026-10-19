//! Application-neutral event contract. Raw events are canonical; trainer-specific alignment is a
//! derived view, so games, rewards, and environment APIs do not belong here.

use agentpc_core::{Action, AgentId, InputDevice, Provenance, RawInputEvent, SourceId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const SCHEMA: &str = "agentpc.trajectory/v3";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryEvent {
    pub schema: String,
    pub seq: u64,
    pub t_us: u64,
    #[serde(flatten)]
    pub kind: EventKind,
}

impl TrajectoryEvent {
    pub fn frame(
        seq: u64,
        source_id: SourceId,
        frame_id: u64,
        captured_t_us: u64,
        observed_t_us: u64,
        heartbeat: bool,
    ) -> Self {
        Self {
            schema: SCHEMA.into(),
            seq,
            t_us: observed_t_us,
            kind: EventKind::Frame {
                source_id,
                frame_id,
                captured_t_us,
                observed_t_us,
                origin: if heartbeat {
                    FrameOrigin::Heartbeat
                } else {
                    FrameOrigin::Captured
                },
            },
        }
    }

    pub fn effective_action(
        seq: u64,
        t_us: u64,
        source_id: SourceId,
        action: Action,
        provenance: Provenance,
    ) -> Self {
        let mut ownership = ActionOwnership::default();
        for key in &action.keys_held {
            ownership.keys.insert(key.clone(), provenance);
        }
        if action.mouse != Default::default() {
            ownership.mouse = Some(provenance);
        }
        let b = action.buttons;
        for (name, pressed) in [
            ("a", b.a),
            ("b", b.b),
            ("x", b.x),
            ("y", b.y),
            ("lb", b.lb),
            ("rb", b.rb),
            ("back", b.back),
            ("start", b.start),
            ("ls", b.ls),
            ("rs", b.rs),
            ("dpad_u", b.dpad_u),
            ("dpad_d", b.dpad_d),
            ("dpad_l", b.dpad_l),
            ("dpad_r", b.dpad_r),
        ] {
            if pressed {
                ownership.gamepad_buttons.insert(name.into(), provenance);
            }
        }
        let a = action.axes;
        for (name, value) in [
            ("lx", a.lx),
            ("ly", a.ly),
            ("rx", a.rx),
            ("ry", a.ry),
            ("lt", a.lt),
            ("rt", a.rt),
        ] {
            if value.abs() >= 1e-3 {
                ownership.gamepad_axes.insert(name.into(), provenance);
            }
        }
        Self {
            schema: SCHEMA.into(),
            seq,
            t_us,
            kind: EventKind::EffectiveAction {
                source_id,
                provenance,
                action,
                ownership,
            },
        }
    }

    pub fn gap(
        seq: u64,
        t_us: u64,
        source_id: SourceId,
        component: GapComponent,
        dropped: u64,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            schema: SCHEMA.into(),
            seq,
            t_us,
            kind: EventKind::Gap {
                source_id,
                component,
                dropped,
                detail: detail.into(),
            },
        }
    }

    pub fn raw_input(seq: u64, event: RawInputEvent) -> Self {
        Self {
            schema: SCHEMA.into(),
            seq,
            t_us: event.t_us,
            kind: EventKind::RawInput {
                source_id: event.source_id,
                device: event.device,
                action: event.action,
                injected: false,
            },
        }
    }

    pub fn handoff(seq: u64, event: crate::HandoffRecord) -> Self {
        Self {
            schema: SCHEMA.into(),
            seq,
            t_us: event.t_us,
            kind: EventKind::Handoff {
                source_id: event.source_id,
                from: event.from,
                to: event.to,
                reason: event.reason,
            },
        }
    }

    pub fn policy_proposal(seq: u64, event: crate::PolicyProposalRecord) -> Self {
        Self {
            schema: SCHEMA.into(),
            seq,
            t_us: event.received_t_us,
            kind: EventKind::PolicyProposal {
                proposal_id: event.proposal_id,
                session_id: event.session_id,
                source_id: event.source_id,
                agent_id: event.agent_id,
                observed_frame_ids: event.observed_frame_ids,
                manifest_sha256: event.manifest_sha256,
                action: event.action,
                inference_started_t_us: event.inference_started_t_us,
                inference_finished_t_us: event.inference_finished_t_us,
                inference_clock: event.inference_clock,
                received_t_us: event.received_t_us,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum EventKind {
    Frame {
        source_id: SourceId,
        frame_id: u64,
        captured_t_us: u64,
        observed_t_us: u64,
        origin: FrameOrigin,
    },
    RawInput {
        source_id: SourceId,
        device: InputDevice,
        action: Action,
        injected: bool,
    },
    PolicyProposal {
        proposal_id: String,
        session_id: String,
        source_id: SourceId,
        agent_id: AgentId,
        observed_frame_ids: Vec<u64>,
        manifest_sha256: Option<String>,
        action: Action,
        inference_started_t_us: u64,
        inference_finished_t_us: u64,
        inference_clock: String,
        received_t_us: u64,
    },
    EffectiveAction {
        source_id: SourceId,
        provenance: Provenance,
        action: Action,
        ownership: ActionOwnership,
    },
    Handoff {
        source_id: SourceId,
        from: Option<Provenance>,
        to: Option<Provenance>,
        reason: HandoffReason,
    },
    Gap {
        source_id: SourceId,
        component: GapComponent,
        dropped: u64,
        detail: String,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrameOrigin {
    Captured,
    Heartbeat,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ActionOwnership {
    pub keys: BTreeMap<String, Provenance>,
    pub mouse: Option<Provenance>,
    pub gamepad_buttons: BTreeMap<String, Provenance>,
    pub gamepad_axes: BTreeMap<String, Provenance>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HandoffReason {
    HumanActivity,
    HumanIdle,
    Pause,
    Resume,
    Stale,
    DriverChanged,
    Disconnect,
    Stop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GapComponent {
    Capture,
    Input,
    Proposal,
    Recorder,
    Encoder,
    Transport,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn heartbeat_preserves_capture_identity_and_time() {
        let json = serde_json::to_value(TrajectoryEvent::frame(8, 3, 42, 100, 900, true)).unwrap();
        assert_eq!(json["schema"], SCHEMA);
        assert_eq!(json["seq"], 8);
        assert_eq!(json["frame_id"], 42);
        assert_eq!(json["source_id"], 3);
        assert_eq!(json["captured_t_us"], 100);
        assert_eq!(json["origin"], "heartbeat");
    }
    #[test]
    fn ownership_can_mix_modalities() {
        let mut own = ActionOwnership::default();
        own.keys.insert("W".into(), Provenance::Human);
        own.gamepad_axes.insert("lx".into(), Provenance::Agent(7));
        let json = serde_json::to_value(own).unwrap();
        assert_eq!(json["keys"]["W"], "human");
        assert_eq!(json["gamepad_axes"]["lx"]["agent"], 7);
    }
    #[test]
    fn effective_action_marks_active_modalities_with_driver() {
        let action = Action {
            keys_held: vec!["W".into()],
            axes: agentpc_core::Axes {
                lx: 0.5,
                ..Default::default()
            },
            buttons: agentpc_core::Buttons {
                a: true,
                ..Default::default()
            },
            mouse: agentpc_core::Mouse {
                dx: 2,
                ..Default::default()
            },
            ..Default::default()
        };
        let json = serde_json::to_value(TrajectoryEvent::effective_action(
            2,
            50,
            3,
            action,
            Provenance::Agent(7),
        ))
        .unwrap();
        assert_eq!(json["event"], "effective_action");
        assert_eq!(json["provenance"]["agent"], 7);
        assert_eq!(json["ownership"]["keys"]["W"]["agent"], 7);
        assert_eq!(json["ownership"]["gamepad_axes"]["lx"]["agent"], 7);
        assert_eq!(json["ownership"]["gamepad_buttons"]["a"]["agent"], 7);
        assert_eq!(json["ownership"]["mouse"]["agent"], 7);
    }
    #[test]
    fn gap_is_explicit_and_counted() {
        let json = serde_json::to_value(TrajectoryEvent::gap(
            4,
            90,
            3,
            GapComponent::Recorder,
            12,
            "queue full",
        ))
        .unwrap();
        assert_eq!(json["event"], "gap");
        assert_eq!(json["component"], "recorder");
        assert_eq!(json["dropped"], 12);
    }
    #[test]
    fn raw_mouse_event_preserves_motion_and_physical_origin() {
        let event = RawInputEvent {
            source_id: 3,
            t_us: 90,
            device: InputDevice::Mouse,
            action: Action {
                mouse: agentpc_core::Mouse {
                    dx: 4,
                    dy: -2,
                    wheel: 120,
                    ..Default::default()
                },
                ..Default::default()
            },
        };
        let json = serde_json::to_value(TrajectoryEvent::raw_input(5, event)).unwrap();
        assert_eq!(json["event"], "raw_input");
        assert_eq!(json["device"], "mouse");
        assert_eq!(json["action"]["mouse"]["dx"], 4);
        assert_eq!(json["injected"], false);
    }
    #[test]
    fn handoff_links_previous_and_next_driver() {
        let event = crate::HandoffRecord {
            source_id: 3,
            t_us: 90,
            from: Some(Provenance::Agent(7)),
            to: Some(Provenance::Human),
            reason: HandoffReason::HumanActivity,
        };
        let json = serde_json::to_value(TrajectoryEvent::handoff(6, event)).unwrap();
        assert_eq!(json["event"], "handoff");
        assert_eq!(json["from"]["agent"], 7);
        assert_eq!(json["to"], "human");
        assert_eq!(json["reason"], "human_activity");
    }
    #[test]
    fn policy_proposal_preserves_cross_host_correlation_and_clock_domains() {
        let event = crate::PolicyProposalRecord {
            proposal_id: "ad440027-f6b1-4ea3-83a2-334970999f76".into(),
            session_id: "session-a".into(),
            source_id: 3,
            agent_id: 7,
            observed_frame_ids: vec![41, 42],
            manifest_sha256: Some("a".repeat(64)),
            action: Action {
                keys_held: vec!["W".into()],
                ..Default::default()
            },
            inference_started_t_us: 1_000,
            inference_finished_t_us: 1_200,
            inference_clock: "cradle_monotonic".into(),
            received_t_us: 9_000,
        };
        let json = serde_json::to_value(TrajectoryEvent::policy_proposal(7, event)).unwrap();
        assert_eq!(json["event"], "policy_proposal");
        assert_eq!(json["proposal_id"], "ad440027-f6b1-4ea3-83a2-334970999f76");
        assert_eq!(json["observed_frame_ids"], serde_json::json!([41, 42]));
        assert_eq!(json["inference_clock"], "cradle_monotonic");
        assert_eq!(json["t_us"], 9_000);
        assert_eq!(json["received_t_us"], 9_000);
    }
}
