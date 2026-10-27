//! The virtual Xbox 360 controller (ViGEmBus) a model plays through, and the passthrough that keeps
//! one plugged in for as long as the sidecar runs.
//!
//! Games that bind a controller by device (Eden binds Player 1 to one pad's SDL GUID) never see a
//! virtual pad that only exists while a model plays. With passthrough on, one virtual pad stays
//! plugged in and carries your real controller's input, polled every few ms; while a model drives
//! it carries the model's input merged with yours (buttons OR'd, each axis whichever is further
//! from centre), so you can always reach in. Rumble the game sends to it goes on to the real pad(s).
//!
//! Nothing to set up in the game: the passthrough reports your real pad's vendor/product id, and
//! with HidHide installed the real pad is hidden from every other app (hidhide.rs), so a game
//! bound to your controller finds the passthrough in its place. Hiding is undone when it stops.
//!
//! XInput can't tell a virtual pad from a real one, so every virtual pad's slot is reported to the
//! human tap, which skips it: recordings and takeover see only your real controller.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use agentpc_core::{Axes, Buttons};
use serde_json::json;
use windows::Win32::UI::Input::XboxController::{XInputGetState, XInputSetState, XINPUT_STATE, XINPUT_VIBRATION};

use crate::{emit, hidhide, log};

pub struct VirtualPad {
    target: vigem_client::Xbox360Wired<vigem_client::Client>,
    pub slot: Option<u32>,
}

impl VirtualPad {
    pub fn new() -> Result<Self, String> {
        Self::with_id(vigem_client::TargetId::XBOX360_WIRED)
    }

    /// A pad that reports this vendor/product id (a real controller's, so apps bound to it take this one).
    pub fn with_id(id: vigem_client::TargetId) -> Result<Self, String> {
        let client = vigem_client::Client::connect().map_err(|e| format!("ViGEmBus not available ({e})"))?;
        let mut target = vigem_client::Xbox360Wired::new(client, id);
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
    } else if !ACTIVE.load(Ordering::SeqCst) {
        // nothing running to undo it (e.g. the last sidecar was killed while hiding was on)
        hidhide::unhide();
    }
}

/// Stops the passthrough and waits (up to 2 s) until your controller is visible to apps again.
pub fn shutdown() {
    set_passthrough(false);
    for _ in 0..200 {
        if !ACTIVE.load(Ordering::SeqCst) {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
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
    // take on the real pad's identity, so apps bound to it take this one instead
    let reals = hidhide::real_pads();
    let id = reals.first().map_or(vigem_client::TargetId::XBOX360_WIRED, |p| vigem_client::TargetId { vendor: p.vendor, product: p.product });
    let mut pad = match VirtualPad::with_id(id) {
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
    let hidden = hide(&reals);
    ACTIVE.store(true, Ordering::SeqCst);
    let id_str = format!("{:04x}:{:04x}", id.vendor, id.product);
    emit(json!({ "event": "passthrough", "on": true, "slot": own, "id": id_str, "hidden": hidden, "error": null }));
    log(format!("passthrough pad on (XInput slot {own}, id {id_str}, real pads hidden: {hidden})"));

    let mut real = 0u8;
    let mut last = None;
    let mut n = 0u32;
    let mut chord = Chord::default();
    let mut nav: Option<Nav> = None;
    while EPOCH.load(Ordering::SeqCst) == epoch {
        let seen = real;
        let mut human = read_real(own, &mut real, n % PROBE_EVERY == 0);
        let now = Instant::now();
        let raw = human.buttons.raw;
        let (menu_btns, fired) = chord.step(now, raw);
        human.buttons.raw = (raw & !MENU_BTNS) | menu_btns;
        if fired {
            emit(json!({ "event": "pad_menu" }));
        }
        // the panel has the pad: the game gets only the model
        if MENU.load(Ordering::Relaxed) {
            nav.get_or_insert_with(|| Nav::new(raw)).step(now, &human);
            human = vigem_client::XGamepad::default();
        } else {
            nav = None;
        }
        n = n.wrapping_add(1);
        REAL.store(real, Ordering::Relaxed);
        // a controller was plugged in: hide it too, and if it's the only one, become it
        if real & !seen != 0 {
            let now = hidhide::real_pads();
            let first = now.first().map(|p| (p.vendor, p.product));
            if first.is_some_and(|f| f != (id.vendor, id.product)) {
                log("passthrough: a different controller, plugging in again as it");
                set_passthrough(true);
                break;
            }
            hide(&now);
        }
        let out = match *AGENT.lock().unwrap() {
            Some(agent) => merge(agent, human),
            None => human,
        };
        if last != Some(out) && pad.target.update(&out).is_ok() {
            last = Some(out);
        }
        std::thread::sleep(POLL);
    }
    // your controller comes back for every app before the passthrough goes away
    hidhide::unhide();
    rumble(0, 0);
    drop(pad);
    ACTIVE.store(false, Ordering::SeqCst);
    emit(json!({ "event": "passthrough", "on": false, "slot": null, "error": null }));
}

/// View + Start: opens and closes player-one's panel.
const MENU_BTNS: u16 = 0x0030;
/// How long View or Start alone is held back, waiting to see if the other joins it.
const CHORD_WINDOW: Duration = Duration::from_millis(100);
/// A quick tap that was held back is still delivered, pressed this long.
const TAP: Duration = Duration::from_millis(50);

/// The panel has the pad (the dock says so with `{"op":"pad_menu","on":true}`).
static MENU: AtomicBool = AtomicBool::new(false);

pub fn set_menu(on: bool) {
    MENU.store(on, Ordering::Relaxed);
    agentpc_capture::set_pad_muted(on);
}

/// Spots View + Start without the game seeing either: one of them alone is held back for
/// CHORD_WINDOW, then passed on late (a tap still arrives, as a short press).
#[derive(Default)]
struct Chord {
    /// the lone menu button that's down, and since when
    down: Option<(u16, Instant)>,
    /// both were pressed: neither goes through until both are up again
    swallow: bool,
    /// a held-back tap being replayed: (bits, until)
    tap: Option<(u16, Instant)>,
}

impl Chord {
    /// The menu buttons to pass on this poll, and whether the chord just fired.
    fn step(&mut self, now: Instant, raw: u16) -> (u16, bool) {
        let m = raw & MENU_BTNS;
        if self.swallow {
            self.swallow = m != 0;
            return (0, false);
        }
        if m == MENU_BTNS {
            *self = Self { swallow: true, ..Self::default() };
            return (0, true);
        }
        let mut out = 0;
        match self.tap {
            Some((bits, until)) if now < until => out |= bits,
            _ => self.tap = None,
        }
        if m == 0 {
            if let Some((bits, since)) = self.down.take() {
                if now - since < CHORD_WINDOW {
                    self.tap = Some((bits, now + TAP));
                    out |= bits;
                }
            }
            return (out, false);
        }
        let since = match self.down {
            Some((_, since)) => since,
            None => now,
        };
        self.down = Some((m, since));
        if now - since >= CHORD_WINDOW {
            out |= m;
        }
        (out, false)
    }
}

const NAV_FIRST: Duration = Duration::from_millis(350);
const NAV_REPEAT: Duration = Duration::from_millis(110);

/// Menu mode: the pad's presses as `{"event":"pad_nav","btn"}` for the panel. Directions (d-pad or
/// left stick) repeat while held; buttons fire on press.
struct Nav {
    prev: u16,
    dir: Option<(&'static str, Instant)>,
}

impl Nav {
    /// starts from the buttons already down, so the press that opened the panel isn't a click
    fn new(raw: u16) -> Self {
        Self { prev: raw, dir: None }
    }

    fn step(&mut self, now: Instant, g: &vigem_client::XGamepad) {
        let b = g.buttons.raw;
        for (bit, name) in [(0x1000u16, "a"), (0x2000, "b"), (0x4000, "x"), (0x8000, "y"), (0x0100, "lb"), (0x0200, "rb")] {
            if b & bit != 0 && self.prev & bit == 0 {
                emit(json!({ "event": "pad_nav", "btn": name }));
            }
        }
        self.prev = b;
        const TILT: i16 = 16000;
        let dir = if b & 0x0001 != 0 || g.thumb_ly > TILT {
            Some("up")
        } else if b & 0x0002 != 0 || g.thumb_ly < -TILT {
            Some("down")
        } else if b & 0x0004 != 0 || g.thumb_lx < -TILT {
            Some("left")
        } else if b & 0x0008 != 0 || g.thumb_lx > TILT {
            Some("right")
        } else {
            None
        };
        match (dir, self.dir) {
            (None, _) => self.dir = None,
            (Some(d), Some((held, next))) if d == held => {
                if now >= next {
                    emit(json!({ "event": "pad_nav", "btn": d }));
                    self.dir = Some((d, now + NAV_REPEAT));
                }
            }
            (Some(d), _) => {
                emit(json!({ "event": "pad_nav", "btn": d }));
                self.dir = Some((d, now + NAV_FIRST));
            }
        }
    }
}

/// Hides the real pads from other apps when HidHide is installed; false if they stay visible.
fn hide(reals: &[hidhide::RealPad]) -> bool {
    if reals.is_empty() || !hidhide::installed() {
        return false;
    }
    match hidhide::hide(reals) {
        Ok(()) => true,
        Err(e) => {
            log(format!("passthrough: couldn't hide your controller ({e})"));
            false
        }
    }
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
    fn chord_holds_back_view_and_start() {
        let t0 = Instant::now();
        let ms = |n: u64| t0 + Duration::from_millis(n);
        let mut c = Chord::default();
        // View then Start 40 ms later: the chord, and the game never sees either
        assert_eq!(c.step(ms(0), 0x0020), (0, false));
        assert_eq!(c.step(ms(40), 0x0030), (0, true));
        assert_eq!(c.step(ms(60), 0x0010), (0, false));
        assert_eq!(c.step(ms(80), 0), (0, false));
        // a quick View tap is replayed as a short press
        assert_eq!(c.step(ms(200), 0x0020), (0, false));
        assert_eq!(c.step(ms(230), 0), (0x0020, false));
        assert_eq!(c.step(ms(260), 0), (0x0020, false));
        assert_eq!(c.step(ms(300), 0), (0, false));
        // a held Start goes through after the window
        assert_eq!(c.step(ms(400), 0x1010), (0, false));
        assert_eq!(c.step(ms(520), 0x1010), (0x0010, false));
    }

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

