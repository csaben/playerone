//! The virtual Xbox 360 controller (ViGEmBus) a model plays through, and the passthrough that keeps
//! one plugged in for as long as the sidecar runs.
//!
//! Games that bind a controller by device (Eden binds Player 1 to one pad's SDL GUID) never see a
//! virtual pad that only exists while a model plays. With passthrough on, one virtual pad stays
//! plugged in and carries your real controller's input, polled every few ms; while a model drives
//! it carries the model's input merged with yours (buttons OR'd, each axis whichever is further
//! from centre), so you can always reach in. Bind the game to the "Xbox 360 Controller" once and it
//! gets both. Rumble the game sends to the virtual pad goes on to the real one(s).
//!
//! XInput can't tell a virtual pad from a real one, so every virtual pad's slot is reported to the
//! human tap, which skips it: recordings and takeover see only your real controller.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use agentpc_core::{Axes, Buttons};
use serde_json::json;
use windows::Win32::UI::Input::XboxController::{XInputGetState, XInputSetState, XINPUT_STATE, XINPUT_VIBRATION};

use crate::{emit, log};

pub struct VirtualPad {
    target: vigem_client::Xbox360Wired<vigem_client::Client>,
    pub slot: Option<u32>,
}

impl VirtualPad {
    pub fn new() -> Result<Self, String> {
        let client = vigem_client::Client::connect().map_err(|e| format!("ViGEmBus not available ({e})"))?;
        let mut target = vigem_client::Xbox360Wired::new(client, vigem_client::TargetId::XBOX360_WIRED);
        target.plugin().map_err(|e| format!("virtual pad plugin failed: {e}"))?;
        target.wait_ready().map_err(|e| format!("virtual pad not ready: {e}"))?;
        let slot = find_slot(&mut target);
        match slot {
            Some(s) => agentpc_capture::virtual_pad_slot(s, true),
            None => agentpc_capture::virtual_pad_attached(),
        }
        log(format!("virtual pad plugged in (XInput slot {slot:?})"));
        Ok(Self { target, slot })
    }

    pub fn write(&mut self, a: &Axes, b: &Buttons) {
        let _ = self.target.update(&report(a, b));
    }
}

impl Drop for VirtualPad {
    fn drop(&mut self) {
        let _ = self.target.update(&vigem_client::XGamepad::default());
        let _ = self.target.unplug();
        match self.slot {
            Some(s) => agentpc_capture::virtual_pad_slot(s, false),
            None => agentpc_capture::virtual_pad_detached(),
        }
    }
}

/// agentpc-core's pad state (sticks [-1,1], ly +1 up; triggers [0,1]) as an XInput report.
pub fn report(a: &Axes, b: &Buttons) -> vigem_client::XGamepad {
    let mut raw = 0u16;
    for (on, bit) in [
        (b.dpad_u, 0x0001u16), (b.dpad_d, 0x0002), (b.dpad_l, 0x0004), (b.dpad_r, 0x0008),
        (b.start, 0x0010), (b.back, 0x0020), (b.ls, 0x0040), (b.rs, 0x0080),
        (b.lb, 0x0100), (b.rb, 0x0200), (b.a, 0x1000), (b.b, 0x2000), (b.x, 0x4000), (b.y, 0x8000),
    ] {
        if on {
            raw |= bit;
        }
    }
    let stick = |v: f32| (v.clamp(-1.0, 1.0) * 32767.0) as i16;
    let trig = |v: f32| (v.clamp(0.0, 1.0) * 255.0) as u8;
    vigem_client::XGamepad {
        buttons: vigem_client::XButtons { raw },
        thumb_lx: stick(a.lx), thumb_ly: stick(a.ly), thumb_rx: stick(a.rx), thumb_ry: stick(a.ry),
        left_trigger: trig(a.lt), right_trigger: trig(a.rt),
    }
}

/// Which XInput slot the new pad landed in. ViGEm's own get_user_index can be wrong when a real
/// controller is also connected, so this writes a signature (stick values inside the deadzone, so
/// a game reads it as centred) and looks for it, for up to 1.5 s while XInput picks the pad up.
fn find_slot(target: &mut vigem_client::Xbox360Wired<vigem_client::Client>) -> Option<u32> {
    const SIG: (i16, i16) = (1234, -2345);
    let sig = vigem_client::XGamepad { thumb_lx: SIG.0, thumb_ry: SIG.1, ..Default::default() };
    let mut found = None;
    for _ in 0..30 {
        let _ = target.update(&sig);
        std::thread::sleep(Duration::from_millis(50));
        found = (0..4u32).find(|&i| {
            let mut st = XINPUT_STATE::default();
            (unsafe { XInputGetState(i, &mut st) }) == 0 && (st.Gamepad.sThumbLX, st.Gamepad.sThumbRY) == SIG
        });
        if found.is_some() {
            break;
        }
    }
    let _ = target.update(&vigem_client::XGamepad::default());
    found
}

/// Bumped to stop the running passthrough thread (each thread runs while it holds the current one).
static EPOCH: AtomicU64 = AtomicU64::new(0);
/// The passthrough pad is plugged in and its slot is known: models play through it.
static ACTIVE: AtomicBool = AtomicBool::new(false);
/// Real controllers seen by the passthrough poll, for rumble.
static REAL: AtomicU8 = AtomicU8::new(0);
/// What the model holds while it drives; None passes your controller through alone.
static AGENT: Mutex<Option<vigem_client::XGamepad>> = Mutex::new(None);

const POLL: Duration = Duration::from_millis(4);
/// empty XInput slots are slow to probe, so they're looked at every ~2 s
const PROBE_EVERY: u32 = 500;

pub fn passthrough_active() -> bool {
    ACTIVE.load(Ordering::Relaxed)
}

/// Turns the passthrough pad on or off. Plugging in takes up to ~1.5 s, so it happens on its own
/// thread and reports `{"event":"passthrough","on","slot","error"}`.
pub fn set_passthrough(on: bool) {
    let epoch = EPOCH.fetch_add(1, Ordering::SeqCst) + 1;
    if on {
        std::thread::spawn(move || run(epoch));
    }
}

/// The model's pad state while it drives (None once it lets go).
pub fn set_agent(state: Option<(&Axes, &Buttons)>) {
    *AGENT.lock().unwrap() = state.map(|(a, b)| report(a, b));
}

fn run(epoch: u64) {
    // a previous thread may still hold its pad for up to one poll
    while ACTIVE.load(Ordering::SeqCst) && EPOCH.load(Ordering::SeqCst) == epoch {
        std::thread::sleep(POLL);
    }
    let mut pad = match VirtualPad::new() {
        Ok(p) => p,
        Err(e) => {
            emit(json!({ "event": "passthrough", "on": false, "slot": null, "error": e }));
            return;
        }
    };
    let Some(own) = pad.slot else {
        emit(json!({ "event": "passthrough", "on": false, "slot": null, "error": "couldn't find the virtual pad's XInput slot" }));
        return;
    };
    if EPOCH.load(Ordering::SeqCst) != epoch {
        return;
    }
    match pad.target.request_notification() {
        Ok(req) => {
            req.spawn_thread(|_, n| rumble(n.large_motor, n.small_motor));
        }
        Err(e) => log(format!("passthrough: no rumble ({e})")),
    }
    ACTIVE.store(true, Ordering::SeqCst);
    emit(json!({ "event": "passthrough", "on": true, "slot": own, "error": null }));
    log(format!("passthrough pad on (XInput slot {own})"));

    let mut real = 0u8;
    let mut last = None;
    let mut n = 0u32;
    while EPOCH.load(Ordering::SeqCst) == epoch {
        let human = read_real(own, &mut real, n % PROBE_EVERY == 0);
        n = n.wrapping_add(1);
        REAL.store(real, Ordering::Relaxed);
        let out = match *AGENT.lock().unwrap() {
            Some(agent) => merge(agent, human),
            None => human,
        };
        if last != Some(out) && pad.target.update(&out).is_ok() {
            last = Some(out);
        }
        std::thread::sleep(POLL);
    }
    rumble(0, 0);
    drop(pad);
    ACTIVE.store(false, Ordering::SeqCst);
    emit(json!({ "event": "passthrough", "on": false, "slot": null, "error": null }));
}

/// Every real controller merged into one raw report (no deadzone: the game applies its own).
fn read_real(own: u32, connected: &mut u8, probe_all: bool) -> vigem_client::XGamepad {
    let mut out = vigem_client::XGamepad::default();
    for slot in 0..4u32 {
        let bit = 1u8 << slot;
        if slot == own || (*connected & bit == 0 && !probe_all) {
            continue;
        }
        let mut st = XINPUT_STATE::default();
        if unsafe { XInputGetState(slot, &mut st) } != 0 {
            *connected &= !bit;
            continue;
        }
        *connected |= bit;
        let g = st.Gamepad;
        out = merge(out, vigem_client::XGamepad {
            buttons: vigem_client::XButtons { raw: g.wButtons.0 },
            left_trigger: g.bLeftTrigger, right_trigger: g.bRightTrigger,
            thumb_lx: g.sThumbLX, thumb_ly: g.sThumbLY, thumb_rx: g.sThumbRX, thumb_ry: g.sThumbRY,
        });
    }
    out
}

fn merge(a: vigem_client::XGamepad, b: vigem_client::XGamepad) -> vigem_client::XGamepad {
    let far = |x: i16, y: i16| if (y as i32).abs() > (x as i32).abs() { y } else { x };
    vigem_client::XGamepad {
        buttons: vigem_client::XButtons { raw: a.buttons.raw | b.buttons.raw },
        left_trigger: a.left_trigger.max(b.left_trigger),
        right_trigger: a.right_trigger.max(b.right_trigger),
        thumb_lx: far(a.thumb_lx, b.thumb_lx),
        thumb_ly: far(a.thumb_ly, b.thumb_ly),
        thumb_rx: far(a.thumb_rx, b.thumb_rx),
        thumb_ry: far(a.thumb_ry, b.thumb_ry),
    }
}

fn rumble(large: u8, small: u8) {
    let real = REAL.load(Ordering::Relaxed);
    let mut v = XINPUT_VIBRATION { wLeftMotorSpeed: (large as u16) << 8, wRightMotorSpeed: (small as u16) << 8 };
    for slot in (0..4u32).filter(|s| real & (1 << s) != 0) {
        unsafe { XInputSetState(slot, &mut v) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_ors_buttons_and_keeps_the_further_axis() {
        let a = vigem_client::XGamepad { buttons: vigem_client::XButtons { raw: 0x1000 }, thumb_lx: 20000, thumb_ly: -300, left_trigger: 10, ..Default::default() };
        let b = vigem_client::XGamepad { buttons: vigem_client::XButtons { raw: 0x0001 }, thumb_lx: -5000, thumb_ly: -30000, left_trigger: 200, ..Default::default() };
        let m = merge(a, b);
        assert_eq!(m.buttons.raw, 0x1001);
        assert_eq!((m.thumb_lx, m.thumb_ly, m.left_trigger), (20000, -30000, 200));
    }

    /// Plugs in a real ViGEm pad (needs ViGEmBus) and reads it back through XInput: run with
    /// `cargo test -p player-one-rec -- --ignored`. Moves a stick only, no buttons.
    #[test]
    #[ignore]
    fn virtual_pad_reaches_xinput() {
        let mut pad = VirtualPad::new().expect("ViGEmBus");
        let slot = pad.slot.expect("slot");
        pad.write(&Axes { lx: 0.5, ..Default::default() }, &Buttons::default());
        std::thread::sleep(Duration::from_millis(100));
        let mut st = XINPUT_STATE::default();
        assert_eq!(unsafe { XInputGetState(slot, &mut st) }, 0);
        assert!((st.Gamepad.sThumbLX as i32 - 16383).abs() < 2, "lx {}", st.Gamepad.sThumbLX);
        drop(pad);
        // XInput notices the unplug within a second or so
        let gone = (0..30).any(|_| {
            std::thread::sleep(Duration::from_millis(100));
            (unsafe { XInputGetState(slot, &mut st) }) != 0
        });
        assert!(gone, "pad still plugged in");
    }

    /// Passthrough with a model holding A and the left stick right: the virtual pad shows both.
    #[test]
    #[ignore]
    fn passthrough_carries_the_agent() {
        set_passthrough(true);
        assert!((0..40).any(|_| { std::thread::sleep(Duration::from_millis(100)); passthrough_active() }), "never came up");
        set_agent(Some((&Axes { lx: 1.0, ..Default::default() }, &Buttons { a: true, ..Default::default() })));
        std::thread::sleep(Duration::from_millis(50));
        let slot = (0..4u32).find(|&i| {
            let mut st = XINPUT_STATE::default();
            (unsafe { XInputGetState(i, &mut st) }) == 0 && st.Gamepad.sThumbLX == 32767 && st.Gamepad.wButtons.0 & 0x1000 != 0
        });
        set_agent(None);
        set_passthrough(false);
        assert!(slot.is_some(), "agent state not on any pad");
    }
}
