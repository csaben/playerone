//! Play mode: a policy server drives the window through real input (SendInput), the way a person
//! would, while the human can take over at any moment.
//!
//! Frames go to the policy over a WebSocket (train/serve.py on cradle): one binary JPEG per frame,
//! one JSON reply `{"keys":[..],"mb":[l,r],"cam":[dx,dy],"version":..}` back. Once a second it
//! reports `{"event":"play_stats","fps","rtt_ms","rtt_max_ms","infer_ms"}`. Held keys and mouse
//! buttons go through agentpc's Dispatcher, whose Arbiter puts the human first: any real input
//! preempts the agent (which then releases everything it holds) and the agent resumes 400 ms after
//! the human goes idle. Camera motion is relative and per frame, so it is sent directly, only while
//! the agent is the driver.
//!
//! In Minecraft with the playerone-bridge mod (mcmod/), actions go to the mod over local TCP and
//! the game plays in the background. Otherwise SendInput, which only reaches the foreground window,
//! so nothing is injected while the target is not focused. The human tap counts only input while
//! the window is foreground and ignores injected input (no device handle), so the agent never
//! "takes over" from itself, and typing in another window doesn't interrupt a bridged game.

use std::collections::BTreeSet;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use agentpc_core::{Action, AgentId, Dispatcher, Mouse, Provenance, SourceId};
use anyhow::{Context, Result};
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::sync::watch;
use tokio_tungstenite::tungstenite::Message;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    MapVirtualKeyW, SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, INPUT_MOUSE, KEYBDINPUT, KEYBD_EVENT_FLAGS,
    KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP, KEYEVENTF_SCANCODE, MAPVK_VK_TO_VSC, MOUSEEVENTF_LEFTDOWN,
    MOUSEEVENTF_LEFTUP, MOUSEEVENTF_MOVE, MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP, MOUSEINPUT,
    MOUSE_EVENT_FLAGS, VIRTUAL_KEY,
};
use windows::Win32::Foundation::HWND;
use windows::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetWindowTextW, GetWindowThreadProcessId};

use crate::{emit, log, root_of};

/// Token → virtual key, the inverse of agentpc_core::vocab::token_for_vk for the keys a policy uses.
fn vk_for(token: &str) -> Option<u16> {
    let b = token.as_bytes();
    if b.len() == 1 && (b[0].is_ascii_uppercase() || b[0].is_ascii_digit()) {
        return Some(b[0] as u16);
    }
    Some(match token {
        "SPACE" => 0x20,
        "ENTER" => 0x0D,
        "ESC" => 0x1B,
        "TAB" => 0x09,
        "LEFT" => 0x25,
        "UP" => 0x26,
        "RIGHT" => 0x27,
        "DOWN" => 0x28,
        "LSHIFT" => 0xA0,
        "RSHIFT" => 0xA1,
        "LCTRL" => 0xA2,
        "RCTRL" => 0xA3,
        "LALT" => 0xA4,
        _ => return None,
    })
}

fn key_input(vk: u16, up: bool) -> INPUT {
    // Scan codes, not VKs: games (GLFW/Minecraft, DirectInput) read the scan code.
    let scan = unsafe { MapVirtualKeyW(vk as u32, MAPVK_VK_TO_VSC) } as u16;
    let extended = matches!(vk, 0x25..=0x28 | 0xA3);
    let mut flags = KEYEVENTF_SCANCODE;
    if extended {
        flags |= KEYEVENTF_EXTENDEDKEY;
    }
    if up {
        flags |= KEYEVENTF_KEYUP;
    }
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT { wVk: VIRTUAL_KEY(0), wScan: scan, dwFlags: KEYBD_EVENT_FLAGS(flags.0), time: 0, dwExtraInfo: 0 },
        },
    }
}

fn mouse_input(dx: i32, dy: i32, flags: MOUSE_EVENT_FLAGS) -> INPUT {
    INPUT {
        r#type: INPUT_MOUSE,
        Anonymous: INPUT_0 { mi: MOUSEINPUT { dx, dy, mouseData: 0, dwFlags: flags, time: 0, dwExtraInfo: 0 } },
    }
}

fn send(inputs: &[INPUT]) {
    if !inputs.is_empty() {
        unsafe { SendInput(inputs, std::mem::size_of::<INPUT>() as i32) };
    }
}

/// What the agent is holding down right now, so it can be released exactly.
#[derive(Default)]
struct Held {
    keys: BTreeSet<u16>,
    buttons: u8,
    /// the same keys as policy tokens, for the bridge
    tokens: Vec<String>,
    /// the bridge has been told to release since the last act
    released: bool,
}

/// playerone-bridge (mcmod/): a Fabric mod inside Minecraft that takes held keys, mouse buttons and
/// camera over local TCP and applies them in the game, so the agent plays while Minecraft is in the
/// background. Newline-delimited JSON; the mod releases everything if nothing arrives for 1 s, so an
/// idle connection is kept alive with pings.
struct Bridge {
    stream: TcpStream,
    last_send: Instant,
}

const BRIDGE_ADDR: &str = "127.0.0.1:8970";

impl Bridge {
    /// Connects and pings. None when the mod isn't running, or when it answers for a Minecraft that
    /// isn't this window (its pong carries the game's pid; without one, the title must say Minecraft).
    fn connect(root: isize) -> Option<Self> {
        let addr = BRIDGE_ADDR.parse().ok()?;
        let stream = TcpStream::connect_timeout(&addr, Duration::from_millis(300)).ok()?;
        stream.set_nodelay(true).ok();
        stream.set_read_timeout(Some(Duration::from_millis(500))).ok();
        let mut reader = BufReader::new(stream.try_clone().ok()?);
        (&stream).write_all(b"{\"ping\":1}\n").ok()?;
        let mut line = String::new();
        reader.read_line(&mut line).ok()?;
        let pong: Value = serde_json::from_str(&line).ok()?;
        pong.get("pong")?;
        let ours = match pong["pid"].as_u64() {
            Some(pid) => window_pid(root) == pid as u32,
            None => window_title(root).contains("Minecraft"),
        };
        if !ours {
            return None;
        }
        // drain the mod's replies (pongs) so its socket never backs up
        stream.set_read_timeout(None).ok();
        std::thread::spawn(move || for _ in reader.lines().map_while(|l| l.ok()) {});
        Some(Self { stream, last_send: Instant::now() })
    }

    fn send(&mut self, v: Value) -> std::io::Result<()> {
        self.last_send = Instant::now();
        (&self.stream).write_all(format!("{v}\n").as_bytes())
    }
}

fn window_pid(root: isize) -> u32 {
    let mut pid = 0u32;
    unsafe { GetWindowThreadProcessId(HWND(root as *mut _), Some(&mut pid)) };
    pid
}

fn window_title(root: isize) -> String {
    let mut buf = [0u16; 256];
    let n = unsafe { GetWindowTextW(HWND(root as *mut _), &mut buf) };
    String::from_utf16_lossy(&buf[..n.max(0) as usize])
}

/// Edge-deriving input sink for one window: through the bridge when Minecraft has it, else
/// SendInput (which only reaches the window while it is foreground). Cloned: one copy lives in the
/// Dispatcher, the other lets the play task send the camera and release everything when it stops.
#[derive(Clone)]
pub struct InputSink {
    root: isize,
    /// over the bridge, keep acting while the window isn't foreground
    unfocused: bool,
    held: Arc<Mutex<Held>>,
    bridge: Arc<Mutex<Option<Bridge>>>,
}

impl InputSink {
    pub fn new(root: isize, unfocused: bool) -> Self {
        Self { root, unfocused, held: Arc::default(), bridge: Arc::default() }
    }

    pub fn focused(&self) -> bool {
        root_of(unsafe { GetForegroundWindow() }.0 as isize) == self.root
    }

    /// Tries the bridge; true when actions now go through it.
    pub fn attach_bridge(&self) -> bool {
        let b = Bridge::connect(self.root);
        let on = b.is_some();
        *self.bridge.lock().unwrap() = b;
        on
    }

    pub fn bridged(&self) -> bool {
        self.bridge.lock().unwrap().is_some()
    }

    /// Whether the agent's input reaches the game right now.
    pub fn reaches(&self) -> bool {
        self.focused() || (self.unfocused && self.bridged())
    }

    /// Sends to the bridge; a failed write drops it, and the sink falls back to SendInput.
    fn bridge_send(&self, v: Value) -> bool {
        let mut b = self.bridge.lock().unwrap();
        let Some(br) = b.as_mut() else { return false };
        if let Err(e) = br.send(v) {
            log(format!("bridge lost ({e}); back to SendInput"));
            *b = None;
        }
        true
    }

    fn set(&self, keys: BTreeSet<u16>, tokens: Vec<String>, buttons: u8) {
        let mut h = self.held.lock().unwrap();
        let act = json!({ "act": { "keys": tokens, "mb": [buttons & 1, (buttons >> 1) & 1], "cam": [0, 0] } });
        if !self.bridge_send(act) {
            let mut inputs = Vec::new();
            for &vk in h.keys.difference(&keys) {
                inputs.push(key_input(vk, true));
            }
            for &vk in keys.difference(&h.keys) {
                inputs.push(key_input(vk, false));
            }
            for (bit, down, up) in [(1u8, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP), (2, MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP)] {
                match (h.buttons & bit != 0, buttons & bit != 0) {
                    (false, true) => inputs.push(mouse_input(0, 0, down)),
                    (true, false) => inputs.push(mouse_input(0, 0, up)),
                    _ => {}
                }
            }
            send(&inputs);
        }
        h.keys = keys;
        h.tokens = tokens;
        h.buttons = buttons;
        h.released = false;
    }

    pub fn release_all(&self) {
        if self.bridged() {
            let mut h = self.held.lock().unwrap();
            // once per release: this runs every tick while the agent can't act
            if !h.tokens.is_empty() || h.buttons != 0 || !h.released {
                self.bridge_send(json!({ "release": true }));
            }
            *h = Held { released: true, ..Held::default() };
            return;
        }
        self.set(BTreeSet::new(), Vec::new(), 0);
    }

    pub fn look(&self, dx: i32, dy: i32) {
        if dx == 0 && dy == 0 {
            return;
        }
        let act = {
            let h = self.held.lock().unwrap();
            json!({ "act": { "keys": h.tokens, "mb": [h.buttons & 1, (h.buttons >> 1) & 1], "cam": [dx, dy] } })
        };
        if !self.bridge_send(act) {
            send(&[mouse_input(dx, dy, MOUSEEVENTF_MOVE)]);
        }
    }

    /// Keeps an idle bridge connection alive, so the mod doesn't release or let the game pause.
    pub fn keepalive(&self) {
        let due = self.bridge.lock().unwrap().as_ref().is_some_and(|b| b.last_send.elapsed() > Duration::from_millis(400));
        if due {
            self.bridge_send(json!({ "ping": 1 }));
        }
    }
}

impl agentpc_core::Sink for InputSink {
    fn apply(&mut self, _agent: AgentId, action: &Action) -> Result<(), agentpc_core::SinkError> {
        if !self.reaches() {
            self.release_all();
            return Ok(());
        }
        let tokens: Vec<String> = action.keys_held.iter().filter(|t| vk_for(t).is_some()).cloned().collect();
        let keys = tokens.iter().filter_map(|t| vk_for(t)).collect();
        self.set(keys, tokens, action.mouse.buttons & 0b11);
        Ok(())
    }

    fn release(&mut self, _agent: AgentId) -> Result<(), agentpc_core::SinkError> {
        self.release_all();
        Ok(())
    }
}

/// Play-mode settings from the dock (its Settings view).
pub struct PlayOpts {
    pub unfocused: bool,
    pub yield_to_human: bool,
    pub resume_ms: u64,
    /// sampling settings for the policy server ({"cfg":{...}}), null for its defaults
    pub cfg: Value,
}

/// One policy reply → the level-state Action (keys + mouse buttons) and this frame's camera move.
fn parse_reply(v: &Value) -> (Action, i32, i32) {
    let keys = v["keys"].as_array().map(|a| a.iter().filter_map(|k| k.as_str().map(String::from)).collect()).unwrap_or_default();
    let mb = |i: usize| v["mb"][i].as_f64().unwrap_or(0.0) > 0.5;
    let buttons = (mb(0) as u8) | ((mb(1) as u8) << 1);
    let cam = |i: usize| v["cam"][i].as_f64().unwrap_or(0.0).round() as i32;
    (Action { keys_held: keys, mouse: Mouse { buttons, ..Default::default() }, ..Default::default() }, cam(0), cam(1))
}

/// Streams the newest frame to the policy and applies its replies until `stop` is set or the
/// connection fails. At most one frame is in flight, so a slow round trip drops frames instead of
/// building up lag.
#[allow(clippy::too_many_arguments)]
pub async fn run_policy(
    hwnd: String,
    url: String,
    agent: AgentId,
    source: SourceId,
    dispatcher: Arc<Mutex<Dispatcher>>,
    sink: InputSink,
    cfg: Value,
    mut frames: watch::Receiver<Option<Arc<Vec<u8>>>>,
    stop: Arc<AtomicBool>,
) -> Result<()> {
    let (ws, _) = tokio_tungstenite::connect_async(&url).await.with_context(|| format!("connect {url}"))?;
    let (mut tx, mut rx) = ws.split();
    tx.send(Message::Text(json!({ "reset": true }).to_string())).await?;
    if cfg.is_object() {
        tx.send(Message::Text(json!({ "cfg": cfg }).to_string())).await?;
    }
    log(format!("policy connected: {url}"));
    let bridged = sink.attach_bridge();
    emit(json!({ "event": "bridge", "hwnd": hwnd, "on": bridged }));
    let mut bridge_tried = std::time::Instant::now();
    let mut version = Value::Null;
    let mut driving = false;
    // once a second: replies per second and the round trip (send frame → reply), for the dock
    let mut window_start = std::time::Instant::now();
    let (mut replies, mut rtt_sum, mut rtt_max) = (0u32, 0f64, 0f64);
    let result: Result<()> = async {
        while !stop.load(Ordering::Relaxed) {
            sink.keepalive();
            // the mod may load after play starts (or come back after a restart of the game)
            if !sink.bridged() && bridge_tried.elapsed().as_secs() >= 5 {
                bridge_tried = std::time::Instant::now();
                if sink.attach_bridge() {
                    emit(json!({ "event": "bridge", "hwnd": hwnd, "on": true }));
                }
            }
            if tokio::time::timeout(std::time::Duration::from_millis(200), frames.changed()).await.is_err() {
                continue;
            }
            let Some(jpeg) = frames.borrow_and_update().clone() else { continue };
            let sent = std::time::Instant::now();
            tx.send(Message::Binary(jpeg.to_vec())).await?;
            let reply = loop {
                match rx.next().await {
                    Some(Ok(Message::Text(t))) => break serde_json::from_str::<Value>(&t)?,
                    Some(Ok(Message::Close(_))) | None => anyhow::bail!("policy closed the connection"),
                    Some(Ok(_)) => continue,
                    Some(Err(e)) => return Err(e.into()),
                }
            };
            let rtt = sent.elapsed().as_secs_f64() * 1000.0;
            replies += 1;
            rtt_sum += rtt;
            rtt_max = rtt_max.max(rtt);
            if window_start.elapsed().as_secs_f64() >= 1.0 {
                let secs = window_start.elapsed().as_secs_f64();
                // acting: the agent drives and its input reaches the game (bridge, or window focused)
                emit(json!({ "event": "play_stats", "hwnd": hwnd, "fps": replies as f64 / secs,
                    "rtt_ms": rtt_sum / replies as f64, "rtt_max_ms": rtt_max, "infer_ms": reply["infer_ms"],
                    "acting": driving && sink.reaches(), "bridge": sink.bridged() }));
                window_start = std::time::Instant::now();
                (replies, rtt_sum, rtt_max) = (0, 0.0, 0.0);
            }
            if reply["version"] != version {
                version = reply["version"].clone();
                emit(json!({ "event": "policy", "hwnd": hwnd, "version": version, "step": reply["step"] }));
            }
            let (action, dx, dy) = parse_reply(&reply);
            let now = agentpc_core::now_us();
            let agent_drives = {
                let mut d = dispatcher.lock().unwrap();
                d.on_action(agent, action, now);
                d.tick(now);
                matches!(d.peek(source, now), Some(e) if e.source == Provenance::Agent(agent))
            };
            if agent_drives != driving {
                driving = agent_drives;
                emit(json!({ "event": "driver", "hwnd": hwnd, "agent": driving }));
            }
            if agent_drives && sink.reaches() {
                sink.look(dx, dy);
            }
        }
        Ok(())
    }
    .await;
    sink.release_all();
    let _ = tx.send(Message::Close(None)).await;
    result
}
