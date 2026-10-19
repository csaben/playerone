//! Human RawInput tap — the missing `Source` that makes hybrid precedence and human recording real.
//!
//! Until now `Dispatcher::on_human` had no caller: pure-human play recorded nothing (the recorder
//! skips a frame with no live driver) and hybrid play mis-attributed the human's keystrokes to the
//! agent. This tap fixes both at once. It runs one dedicated thread with a **message-only window**
//! and `RegisterRawInputDevices(... RIDEV_INPUTSINK)`, so it sees real keyboard/mouse input
//! **system-wide regardless of focus** — then attributes each moment to the box whose window is
//! foreground and submits a level-state [`Action`] to that box's [`Dispatcher`] as `Provenance::Human`.
//!
//! Why it only sees the *human*: PostMessage, CDP, and the extension all inject **below** the OS
//! input queue (a window message / inside the renderer), so none of them surface as `WM_INPUT`.
//! GAMEPADS are the addition: the human's physical pad is read by polling XInput on the 50 ms tick
//! (folded into the same level state as keys/mouse buttons). XInput cannot tell an agent's ViGEm
//! pad from a real one, so while any virtual pad exists ([`virtual_pad_attached`]) the pad poll is
//! suspended — keyboard/mouse attribution keeps working throughout.
//!
//! Feeding cadence (the subtle part): the Arbiter keeps the human "live" for `human_idle_return_ms`
//! measured from the last **non-neutral** submit ([`agentpc_core::Arbiter`]). A single submit per
//! key event would therefore let a *held* key go idle-stale in 400 ms and hand the window back to
//! the agent while the human is still pressing it. So we submit on every input edge **and** re-submit
//! the current non-neutral state on a 50 ms timer; releasing everything submits a neutral state once,
//! which does not refresh the clock and thus starts the 400 ms return-to-agent countdown.

use std::collections::{BTreeSet, HashMap};
use std::ffi::c_void;
use std::mem::size_of;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use agentpc_core::{token_for_vk, Action, Axes, Buttons, Dispatcher, InputDevice, Mouse, RawInputBuffer, RawInputEvent, SourceId};

use windows::core::PCWSTR;
use windows::Win32::Foundation::{HINSTANCE, HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::Input::{
    GetRawInputData, RegisterRawInputDevices, HRAWINPUT, RAWINPUT, RAWINPUTDEVICE, RAWINPUTHEADER,
    RIDEV_INPUTSINK, RID_INPUT, RIM_TYPEKEYBOARD, RIM_TYPEMOUSE,
};
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DispatchMessageW, GetAncestor, GetForegroundWindow,
    GetMessageW, GetWindowThreadProcessId, RegisterClassW, SetTimer, TranslateMessage, GA_ROOT,
    HMENU, HWND_MESSAGE, MSG, WINDOW_EX_STYLE, WINDOW_STYLE, WM_INPUT, WM_TIMER, WNDCLASSW,
};

/// One recorded/played session's binding for the tap: the box's capture source id and a handle to
/// the [`Dispatcher`] driving it. Keyed in the [`TapRegistry`] by the session window's **root HWND**
/// (as `isize`), which is what a foreground lookup resolves to.
pub struct TapTarget {
    pub source: SourceId,
    pub dispatcher: Arc<Mutex<Dispatcher>>,
    pub input_events: Option<Arc<RawInputBuffer>>,
}

/// Live ViGEm virtual pads. Keyboard/mouse injection is invisible to RawInput by construction, but
/// XInput cannot distinguish an agent's virtual pad from the human's real one — so while any
/// virtual pad exists the tap's gamepad poll is suspended (real-pad frames in a hybrid PAD session
/// are the trade-off; keyboard/mouse attribution is unaffected).
static VIRTUAL_PADS: AtomicUsize = AtomicUsize::new(0);

/// A ViGEm pad was plugged in (agent gamepad session started).
pub fn virtual_pad_attached() {
    VIRTUAL_PADS.fetch_add(1, Ordering::Relaxed);
}

/// A ViGEm pad went away (agent gamepad session ended). Saturating: never underflows.
pub fn virtual_pad_detached() {
    let _ = VIRTUAL_PADS.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_sub(1));
}

/// Which human devices the tap reads (player-one's device settings): bits 0-3 are XInput slots
/// 1-4, then keyboard and mouse. A device that is off reads as neutral, so it never preempts an
/// agent and never lands in a recording. Default: everything.
static DEVICES: AtomicU8 = AtomicU8::new(0x3F);
pub const DEV_PADS: u8 = 0x0F;
pub const DEV_KEYBOARD: u8 = 1 << 4;
pub const DEV_MOUSE: u8 = 1 << 5;

pub fn set_devices(mask: u8) {
    DEVICES.store(mask & 0x3F, Ordering::Relaxed);
}

pub fn devices() -> u8 {
    DEVICES.load(Ordering::Relaxed)
}

/// Bitmask of XInput slots with a controller plugged in. Probes all four, which is slow on empty
/// slots, so call it when the user opens the device settings, not on a timer.
pub fn connected_pads() -> u8 {
    use windows::Win32::UI::Input::XboxController::{XInputGetState, XINPUT_STATE};
    let mut mask = 0u8;
    for slot in 0..4u32 {
        let mut st = XINPUT_STATE::default();
        if unsafe { XInputGetState(slot, &mut st) } == 0 {
            mask |= 1 << slot;
        }
    }
    mask
}

/// Live sessions the tap can attribute human input to. The desktop inserts on Start and removes on
/// Stop; the tap reads it on each input event. Root-HWND → target.
pub type TapRegistry = Arc<Mutex<HashMap<isize, TapTarget>>>;

/// Start the tap on its own thread. One tap serves every session (RawInput is system-wide); it lives
/// for the app's lifetime. Spawned once at app startup with the shared [`TapRegistry`].
pub fn start_human_tap<L>(registry: TapRegistry, log: L)
where
    L: Fn(String) + Send + 'static,
{
    let _ = std::thread::Builder::new()
        .name("agentpc-human-tap".into())
        .spawn(move || unsafe { run_tap(registry, log) });
}

/// XInput deadzones (the standard left-stick / trigger values); inside them the axis reads 0 so a
/// resting pad is a NEUTRAL action and the 400 ms return-to-agent countdown can run.
const STICK_DEADZONE: f32 = 7849.0;
const TRIGGER_DEADZONE: u8 = 30;

/// Quantized pad fingerprint for change detection (buttons exact; sticks/triggers in 1/15 steps so
/// sensor jitter doesn't count as an edge). Submitted actions carry FULL precision — while
/// non-neutral the 50 ms tick re-submits fresh values regardless of this key.
type PadKey = (u16, u8, u8, i8, i8, i8, i8);
const PAD_NEUTRAL: PadKey = (0, 0, 0, 0, 0, 0, 0);

/// Polls the human's XInput pad(s) and folds them into the tap's level state. Disconnected slots
/// are probed only every ~2 s — XInputGetState on an empty slot is slow enough that polling all
/// four every tick is a known stutter source.
struct PadPoll {
    connected: u8,
    reprobe: u32,
}

impl PadPoll {
    fn new() -> Self {
        Self { connected: 0, reprobe: 0 }
    }

    /// Read all connected pads merged into one state (max magnitude per axis, OR of buttons).
    /// Suspended (neutral) while an agent's virtual pad exists — see [`VIRTUAL_PADS`].
    fn poll(&mut self) -> (Buttons, Axes, PadKey) {
        use windows::Win32::UI::Input::XboxController::{XInputGetState, XINPUT_STATE};
        if VIRTUAL_PADS.load(Ordering::Relaxed) > 0 {
            return (Buttons::default(), Axes::default(), PAD_NEUTRAL);
        }
        let probe_all = self.reprobe == 0;
        self.reprobe = if probe_all { 40 } else { self.reprobe - 1 }; // ~2 s at the 50 ms tick
        let mut btn = 0u16;
        let (mut lt, mut rt) = (0u8, 0u8);
        let (mut lx, mut ly, mut rx, mut ry) = (0i16, 0i16, 0i16, 0i16);
        let amax = |a: i16, b: i16| if (b as i32).abs() > (a as i32).abs() { b } else { a };
        let enabled = DEVICES.load(Ordering::Relaxed);
        for slot in 0..4u32 {
            let bit = 1u8 << slot;
            if enabled & bit == 0 {
                continue;
            }
            if self.connected & bit == 0 && !probe_all {
                continue;
            }
            let mut st = XINPUT_STATE::default();
            if unsafe { XInputGetState(slot, &mut st) } == 0 {
                self.connected |= bit;
                let g = st.Gamepad;
                btn |= g.wButtons.0;
                lt = lt.max(g.bLeftTrigger);
                rt = rt.max(g.bRightTrigger);
                lx = amax(lx, g.sThumbLX);
                ly = amax(ly, g.sThumbLY);
                rx = amax(rx, g.sThumbRX);
                ry = amax(ry, g.sThumbRY);
            } else {
                self.connected &= !bit;
            }
        }
        let stick = |v: i16| -> f32 {
            let a = v as f32;
            if a.abs() <= STICK_DEADZONE {
                0.0
            } else {
                (((a.abs() - STICK_DEADZONE) / (32767.0 - STICK_DEADZONE)).min(1.0)).copysign(a)
            }
        };
        let trig = |v: u8| -> f32 {
            if v <= TRIGGER_DEADZONE {
                0.0
            } else {
                (v - TRIGGER_DEADZONE) as f32 / (255 - TRIGGER_DEADZONE) as f32
            }
        };
        // XINPUT_GAMEPAD wButtons bits (winerror-stable, spelled out to avoid the flag newtype).
        let buttons = Buttons {
            dpad_u: btn & 0x0001 != 0, dpad_d: btn & 0x0002 != 0,
            dpad_l: btn & 0x0004 != 0, dpad_r: btn & 0x0008 != 0,
            start: btn & 0x0010 != 0, back: btn & 0x0020 != 0,
            ls: btn & 0x0040 != 0, rs: btn & 0x0080 != 0,
            lb: btn & 0x0100 != 0, rb: btn & 0x0200 != 0,
            a: btn & 0x1000 != 0, b: btn & 0x2000 != 0,
            x: btn & 0x4000 != 0, y: btn & 0x8000 != 0,
        };
        let axes = Axes {
            lx: stick(lx), ly: stick(ly), rx: stick(rx), ry: stick(ry),
            lt: trig(lt), rt: trig(rt),
        };
        let q = |v: f32| (v * 15.0).round() as i8;
        let key = (btn, q(axes.lt) as u8, q(axes.rt) as u8, q(axes.lx), q(axes.ly), q(axes.rx), q(axes.ry));
        (buttons, axes, key)
    }
}

/// Mutable tap state, owned by the message loop (WM_INPUT / WM_TIMER are pulled off the queue there,
/// so no window-proc pointer juggling is needed).
struct TapState {
    registry: TapRegistry,
    /// Currently-held virtual-key codes (auto-repeat collapses to set membership).
    held_vks: BTreeSet<u32>,
    /// Mouse button bitfield, SCHEMA §5.2 layout: bit0 left, 1 right, 2 middle.
    mouse_buttons: u8,
    /// Human gamepad level state, refreshed by the 50 ms tick (see [`PadPoll`]).
    pad: PadPoll,
    pad_buttons: Buttons,
    pad_axes: Axes,
    pad_key: PadKey,
    /// The (keys, mouse, pad) fingerprint of the last submit — so a bare mouse-move (state
    /// unchanged) doesn't spam submits, and the 50 ms timer knows whether anything is held.
    last: Option<(Vec<String>, u8, PadKey)>,
    log: Box<dyn Fn(String) + Send>,
    submits: u64,
    /// One-shot proof in the console that the gamepad poll is actually seeing the pad.
    pad_seen: bool,
    /// Foreground roots already warned about (input arrived while sessions exist but none matched)
    /// — once per root, so a mis-bound game window can't fail silently for a whole session again.
    warned_roots: BTreeSet<isize>,
}

impl TapState {
    fn tokens(&self) -> Vec<String> {
        self.held_vks.iter().filter_map(|&vk| token_for_vk(vk).map(String::from)).collect()
    }

    /// Recompute the level state after an input edge; submit only if it changed. Uses the pad state
    /// from the latest 50 ms poll (fresh enough — the tick refreshes it continuously).
    fn on_edge(&mut self, device: InputDevice, event_mouse: Mouse) {
        let cur = (self.tokens(), self.mouse_buttons, self.pad_key);
        let mouse_moved = event_mouse.dx != 0 || event_mouse.dy != 0 || event_mouse.wheel != 0;
        if self.last.as_ref() != Some(&cur) || mouse_moved {
            self.submit(&cur.0, cur.1, Some(device), event_mouse);
            self.last = Some(cur);
        }
    }

    /// 50 ms cadence: poll the gamepad (its "edges" surface here rather than via WM_INPUT), then
    /// while ANYTHING is held — keys, mouse buttons, pad — re-submit so the Arbiter's human-activity
    /// clock stays fresh and held sticks stream full-precision values. A neutral state is submitted
    /// once (via the change branch) and then left alone so the 400 ms return-to-agent window elapses.
    fn on_tick(&mut self) {
        // a device switched off mid-press must not stay held
        let enabled = DEVICES.load(Ordering::Relaxed);
        if enabled & DEV_KEYBOARD == 0 {
            self.held_vks.clear();
        }
        if enabled & DEV_MOUSE == 0 {
            self.mouse_buttons = 0;
        }
        let (pb, pa, pk) = self.pad.poll();
        let pad_changed = pk != self.pad_key;
        self.pad_buttons = pb;
        self.pad_axes = pa;
        self.pad_key = pk;
        if pk != PAD_NEUTRAL && !self.pad_seen {
            self.pad_seen = true;
            (self.log)("human tap: gamepad input detected (XInput)".into());
        }
        let cur = (self.tokens(), self.mouse_buttons, pk);
        let non_neutral = !(cur.0.is_empty() && cur.1 == 0 && pk == PAD_NEUTRAL);
        if self.last.as_ref() != Some(&cur) || non_neutral {
            self.submit(&cur.0.clone(), cur.1, pad_changed.then_some(InputDevice::Gamepad), Mouse::default());
            self.last = Some(cur);
        }
    }

    /// Attribute the current state to the foreground window's session, if any, and submit it as
    /// human input. No match (foreground isn't a live session) → dropped, which is exactly right:
    /// the human typing into some other window must not preempt an agent driving elsewhere.
    fn submit(&mut self, tokens: &[String], buttons: u8, event_device: Option<InputDevice>, event_mouse: Mouse) {
        let root = unsafe { foreground_root() };
        if root == 0 {
            return;
        }
        let now = agentpc_core::now_us();
        let action = Action {
            t_us: now,
            keys_held: tokens.to_vec(),
            mouse: Mouse { buttons, ..Default::default() },
            buttons: self.pad_buttons,
            axes: self.pad_axes,
        };
        // Snapshot the target under the registry lock, then release it before touching the
        // dispatcher lock (start/stop only ever hold the registry lock — no lock-order coupling).
        let (target, session_count) = {
            let g = self.registry.lock().unwrap();
            let hit = g.get(&root).map(|t| (t.source, t.dispatcher.clone(), t.input_events.clone())).or_else(|| {
                // Window-identity fallback: games recreate their top-level window (launcher → game,
                // fullscreen flips), so the bound HWND's root can go stale while the PROCESS is the
                // same. Attribute by owning process when the exact root has no session.
                let fg_pid = unsafe { pid_of(root) };
                if fg_pid == 0 {
                    return None;
                }
                g.iter()
                    .find(|(k, _)| unsafe { pid_of(**k) } == fg_pid)
                    .map(|(_, t)| (t.source, t.dispatcher.clone(), t.input_events.clone()))
            });
            (hit, g.len())
        };
        if let Some((source, disp, events)) = target {
            if let (Some(device), Some(events)) = (event_device, events) {
                let mut raw_action = action.clone();
                raw_action.mouse.dx = event_mouse.dx;
                raw_action.mouse.dy = event_mouse.dy;
                raw_action.mouse.wheel = event_mouse.wheel;
                events.push(RawInputEvent { source_id: source, t_us: now, device, action: raw_action });
            }
            if let Ok(mut d) = disp.lock() {
                d.on_human(source, action, now);
            }
            self.submits = self.submits.wrapping_add(1);
            if self.submits == 1 {
                (self.log)(
                    "human tap: first input reached a live session (hybrid preempt + recording now active)"
                        .into(),
                );
            }
        } else if session_count > 0
            && !(tokens.is_empty() && buttons == 0 && self.pad_key == PAD_NEUTRAL)
            && self.warned_roots.insert(root)
        {
            // Sessions exist but this foreground window belongs to none of them (by root OR by
            // process). Say so once per window — an unattributed game session must not be silent.
            (self.log)(format!(
                "human tap: input at foreground window {root:#x} matches no live session — if this IS your bound window, its handle changed; rebind it"
            ));
        }
    }

    /// Parse one `WM_INPUT` into a held-key / mouse-button state change.
    unsafe fn handle_raw_input(&mut self, lparam: LPARAM) {
        let hri = HRAWINPUT(lparam.0 as *mut c_void);
        let header_size = size_of::<RAWINPUTHEADER>() as u32;
        let mut size: u32 = 0;
        // Size query: with pData = None, a successful call returns 0 and sets `size`.
        if GetRawInputData(hri, RID_INPUT, None, &mut size, header_size) != 0 || size == 0 {
            return;
        }
        let mut buf = vec![0u8; size as usize];
        let got =
            GetRawInputData(hri, RID_INPUT, Some(buf.as_mut_ptr() as *mut c_void), &mut size, header_size);
        if got == u32::MAX || got == 0 {
            return;
        }
        let raw = &*(buf.as_ptr() as *const RAWINPUT);
        // player-one: SendInput arrives with no device handle. That is the agent playing, never
        // the human, so it must not count as a takeover (or be recorded as human input).
        if raw.header.hDevice.0.is_null() {
            return;
        }
        let dwtype = raw.header.dwType;
        let enabled = DEVICES.load(Ordering::Relaxed);
        if dwtype == RIM_TYPEKEYBOARD.0 && enabled & DEV_KEYBOARD == 0
            || dwtype == RIM_TYPEMOUSE.0 && enabled & DEV_MOUSE == 0
        {
            return;
        }
        if dwtype == RIM_TYPEKEYBOARD.0 {
            let kb = raw.data.keyboard;
            let vk = kb.VKey as u32;
            if vk == 0 || vk == 0xFF {
                return; // 0xFF is raw input's "fake key" filler for some overlays
            }
            let is_break = (kb.Flags & 0x01) != 0; // RI_KEY_BREAK
            if is_break {
                self.held_vks.remove(&vk);
            } else {
                self.held_vks.insert(vk);
            }
            self.on_edge(InputDevice::Keyboard, Mouse::default());
        } else if dwtype == RIM_TYPEMOUSE.0 {
            let m = raw.data.mouse;
            let flags = m.Anonymous.Anonymous.usButtonFlags;
            let wheel = if flags & 0x0400 != 0 {
                m.Anonymous.Anonymous.usButtonData as i16 as i32
            } else {
                0
            };
            let motion = Mouse { dx: m.lLastX, dy: m.lLastY, wheel, ..Default::default() };
            let before = self.mouse_buttons;
            // usButtonFlags down/up bits → our bit0 left, 1 right, 2 middle.
            if flags & 0x0001 != 0 {
                self.mouse_buttons |= 1 << 0;
            }
            if flags & 0x0002 != 0 {
                self.mouse_buttons &= !(1 << 0);
            }
            if flags & 0x0004 != 0 {
                self.mouse_buttons |= 1 << 1;
            }
            if flags & 0x0008 != 0 {
                self.mouse_buttons &= !(1 << 1);
            }
            if flags & 0x0010 != 0 {
                self.mouse_buttons |= 1 << 2;
            }
            if flags & 0x0020 != 0 {
                self.mouse_buttons &= !(1 << 2);
            }
            if self.mouse_buttons != before || motion.dx != 0 || motion.dy != 0 || motion.wheel != 0 {
                self.on_edge(InputDevice::Mouse, motion);
            }
        }
    }
}

/// Minimal window proc — everything we care about (WM_INPUT / WM_TIMER) is pulled off the queue in
/// the message loop, so this just hands the rest to the default handler. Needed as a real
/// `extern "system"` fn because the crate's `DefWindowProcW` wrapper is Rust-ABI and can't be used
/// as a raw `WNDPROC` pointer directly.
unsafe extern "system" fn wnd_proc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    DefWindowProcW(hwnd, msg, wparam, lparam)
}

/// The foreground window's root HWND as `isize`, or 0 if there is none.
unsafe fn foreground_root() -> isize {
    let fg = GetForegroundWindow();
    if fg.0.is_null() {
        return 0;
    }
    GetAncestor(fg, GA_ROOT).0 as isize
}

/// Owning process id of a window, or 0.
unsafe fn pid_of(hwnd_root: isize) -> u32 {
    let mut pid = 0u32;
    let _ = GetWindowThreadProcessId(HWND(hwnd_root as *mut c_void), Some(&mut pid));
    pid
}

unsafe fn run_tap<L>(registry: TapRegistry, log: L)
where
    L: Fn(String) + Send + 'static,
{
    let hmod = match GetModuleHandleW(PCWSTR::null()) {
        Ok(h) => h,
        Err(e) => {
            log(format!("human tap: GetModuleHandle failed: {e}"));
            return;
        }
    };
    let hinstance = HINSTANCE(hmod.0);
    let class_name: Vec<u16> = "agentpc_raw_input_tap\0".encode_utf16().collect();
    let wc = WNDCLASSW {
        lpfnWndProc: Some(wnd_proc),
        hInstance: hinstance,
        lpszClassName: PCWSTR(class_name.as_ptr()),
        ..Default::default()
    };
    if RegisterClassW(&wc) == 0 {
        log("human tap: RegisterClass failed".into());
        return;
    }

    let hwnd = match CreateWindowExW(
        WINDOW_EX_STYLE(0),
        PCWSTR(class_name.as_ptr()),
        PCWSTR(class_name.as_ptr()),
        WINDOW_STYLE(0),
        0,
        0,
        0,
        0,
        HWND_MESSAGE, // message-only window: no display, just a queue for WM_INPUT/WM_TIMER
        HMENU::default(),
        hinstance,
        None,
    ) {
        Ok(h) => h,
        Err(e) => {
            log(format!("human tap: CreateWindow failed: {e}"));
            return;
        }
    };

    // Keyboard (usage page 0x01, usage 0x06) + mouse (0x02), both RIDEV_INPUTSINK so they arrive
    // even while our message window is not foreground — the whole point of an unfocused tap.
    let devices = [
        RAWINPUTDEVICE { usUsagePage: 0x01, usUsage: 0x06, dwFlags: RIDEV_INPUTSINK, hwndTarget: hwnd },
        RAWINPUTDEVICE { usUsagePage: 0x01, usUsage: 0x02, dwFlags: RIDEV_INPUTSINK, hwndTarget: hwnd },
    ];
    if let Err(e) = RegisterRawInputDevices(&devices, size_of::<RAWINPUTDEVICE>() as u32) {
        log(format!("human tap: RegisterRawInputDevices failed: {e}"));
        return;
    }

    // 50 ms cadence timer (see module docs: keeps a held key from going idle-stale).
    SetTimer(hwnd, 1, 50, None);
    log("human tap: RawInput sink live (keyboard + mouse, RIDEV_INPUTSINK)".into());

    let mut state = TapState {
        registry,
        held_vks: BTreeSet::new(),
        mouse_buttons: 0,
        pad: PadPoll::new(),
        pad_buttons: Buttons::default(),
        pad_axes: Axes::default(),
        pad_key: PAD_NEUTRAL,
        last: None,
        log: Box::new(log),
        submits: 0,
        pad_seen: false,
        warned_roots: BTreeSet::new(),
    };

    let mut msg = MSG::default();
    while GetMessageW(&mut msg, HWND::default(), 0, 0).0 > 0 {
        match msg.message {
            WM_INPUT => state.handle_raw_input(msg.lParam),
            WM_TIMER => state.on_tick(),
            _ => {}
        }
        let _ = TranslateMessage(&msg);
        DispatchMessageW(&msg); // let DefWindowProc perform WM_INPUT cleanup
    }
}
