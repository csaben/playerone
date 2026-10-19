//! player-one-rec â€” records windows for player-one: the window's frames plus the human's keyboard,
//! mouse and gamepad, written as agentpc shards and optionally uploaded to a Hugging Face dataset.
//!
//! The record path is agentpc's human demonstration mode (`start_record` in its desktop app and
//! `run_record_loop` in agentpc-client), without the Tauri app or a cradle connection.
//!
//! One JSON request per stdin line, one JSON reply per stdout line, matched by `id`. Events arrive
//! unprompted as `{"event": ...}`.
//!
//!   {"id":1,"op":"devices"}                  -> {"connected":<pad slot mask>,"enabled":<device mask>}
//!   {"id":2,"op":"set_devices","mask":63}    bits 0-3 pad slots 1-4, 16 keyboard, 32 mouse
//!   {"id":3,"op":"start","hwnd":"123","label":"Notepad","out_dir":"...",
//!    "repo":"user/ds","folder":"Notepad-2026...","hf_cmd":["uv","run","hf"],"hf_cwd":"...",
//!    "fps":20,"max_width":640,"stream":true}
//!                                            fps/max_width set the capture profile (process-wide);
//!                                            stream writes each chunk loose to <dir>/chunks
//!                                            -> {"session":"s...","dir":"..."}
//!   {"id":4,"op":"stop","hwnd":"123"}        replies at once; {"event":"finished"} follows when the
//!                                            shard is complete (and uploaded, if a repo was given)
//!
//!   "policy":{"url":"ws://cradle:8962/","agent":1001}   optional on start: play mode (see play.rs).
//!     also "unfocused" (bool: keep playing in the background over the bridge), "yield" (bool: pause
//!     while you give input), "resume_ms", "cfg" (sampling settings sent to the policy server)
//!                                            The recording keeps running, so the agent's play and
//!                                            the human's corrections are both recorded.
//!
//! Events: {"event":"log","msg"}, {"event":"policy","hwnd","version","step"} (checkpoint changed),
//!         {"event":"driver","hwnd","agent":bool} (who is playing), {"event":"policy_error","hwnd","error"},
//!         {"event":"finished","hwnd","dir","uploaded","error"},
//!         {"event":"chunk","hwnd","session","key","dir"} (stream only: the chunk's files are in dir).

mod play;

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use tokio::sync::mpsc as tmpsc;

use agentpc_capture::{start_human_tap, start_window_capture, CaptureCtl, CapturedFrame, TapRegistry, TapTarget};
use agentpc_core::{Dispatcher, PriorityRule, Provenance, SourceId};
use agentpc_record::upload::{self, UploadConfig};
use agentpc_record::{FrameRecord, RecordConfig, Recorder};

// ~3 s at 20 fps: frames keep queueing while the previous chunk's mp4 is being encoded
const RECORD_QUEUE_CAPACITY: usize = 64;

static OUT: Mutex<()> = Mutex::new(());
static NEXT_SOURCE: AtomicU32 = AtomicU32::new(1);

pub(crate) fn emit(v: Value) {
    let _g = OUT.lock().unwrap();
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{v}");
    let _ = out.flush();
}

pub(crate) fn log(msg: impl Into<String>) {
    emit(json!({ "event": "log", "msg": msg.into() }));
}

/// The recording loop gives every frame to a Dispatcher whose only box is this window; with no
/// agent attached the effective action is always the human's, so the sink has nothing to do.
struct NoSink;

impl agentpc_core::Sink for NoSink {
    fn apply(&mut self, _agent: agentpc_core::AgentId, _action: &agentpc_core::Action) -> Result<(), agentpc_core::SinkError> {
        Ok(())
    }
}

struct Session {
    capture: Option<CaptureCtl>,
    stop: Arc<AtomicBool>,
    tap_key: isize,
}

struct App {
    tap: TapRegistry,
    sessions: Mutex<HashMap<String, Session>>,
}

pub(crate) fn root_of(hwnd: isize) -> isize {
    use windows::Win32::Foundation::HWND;
    use windows::Win32::UI::WindowsAndMessaging::{GetAncestor, GA_ROOT};
    unsafe { GetAncestor(HWND(hwnd as *mut std::ffi::c_void), GA_ROOT).0 as isize }
}

fn start(app: &Arc<App>, req: &Value) -> Result<Value> {
    let hwnd = req["hwnd"].as_str().context("hwnd")?.to_string();
    let hwnd_val: isize = hwnd.parse().map_err(|_| anyhow!("bad hwnd"))?;
    if app.sessions.lock().unwrap().contains_key(&hwnd) {
        return Err(anyhow!("already recording that window"));
    }
    let label = req["label"].as_str().filter(|l| !l.trim().is_empty()).map(String::from).unwrap_or_else(|| format!("hwnd:{hwnd_val:x}"));
    let out_dir = req["out_dir"].as_str().map(PathBuf::from).unwrap_or_else(|| std::env::temp_dir().join("player-one-rec"));
    let upload = req["repo"].as_str().map(str::trim).filter(|r| !r.is_empty()).map(|repo| UploadConfig {
        repo: repo.to_string(),
        folder: req["folder"].as_str().map(str::trim).filter(|f| !f.is_empty()).map(String::from),
        hf_cmd: req["hf_cmd"].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).filter(|v: &Vec<String>| !v.is_empty()).unwrap_or_else(|| vec!["hf".into()]),
        cwd: req["hf_cwd"].as_str().filter(|c| !c.is_empty()).map(PathBuf::from),
    });

    let source: SourceId = NEXT_SOURCE.fetch_add(1, Ordering::Relaxed);
    let tap_key = root_of(hwnd_val);
    let mut dispatcher = Dispatcher::new();
    let policy = req["policy"]["url"].as_str().filter(|u| !u.is_empty()).map(|url| {
        let p = &req["policy"];
        let agent = p["agent"].as_u64().unwrap_or(1001) as agentpc_core::AgentId;
        let opts = play::PlayOpts {
            unfocused: p["unfocused"].as_bool().unwrap_or(true),
            yield_to_human: p["yield"].as_bool().unwrap_or(true),
            resume_ms: p["resume_ms"].as_u64().unwrap_or(400),
            cfg: p["cfg"].clone(),
        };
        (url.to_string(), agent, play::InputSink::new(tap_key, opts.unfocused), opts)
    });
    match &policy {
        Some((_, agent, sink, opts)) => {
            // yield: any real input preempts the agent, which resumes resume_ms after the human idles.
            // Off: shared control, the agent keeps acting while you play too.
            let order = if opts.yield_to_human { vec![Provenance::Human, Provenance::Agent(*agent)] } else { vec![Provenance::Agent(*agent), Provenance::Human] };
            let rule = PriorityRule { order, human_idle_return_ms: opts.resume_ms, paused: vec![] };
            dispatcher.add_box(source, Box::new(sink.clone()), rule);
            dispatcher.assign_agent(*agent, source);
        }
        None => dispatcher.add_box(source, Box::new(NoSink), PriorityRule::default()),
    }
    let dispatcher = Arc::new(Mutex::new(dispatcher));
    let input_events = Arc::new(agentpc_core::RawInputBuffer::new(4096));
    app.tap.lock().unwrap().insert(
        tap_key,
        TapTarget { source, dispatcher: dispatcher.clone(), input_events: Some(input_events.clone()) },
    );

    let mut cfg = RecordConfig::new(out_dir, label);
    let session_id = cfg.session_id.clone();
    let dir = cfg.out_dir.join(&session_id);
    if req["stream"].as_bool().unwrap_or(false) {
        cfg.loose_dir = Some(dir.join("chunks"));
    }
    if let Some(fps) = req["fps"].as_u64() {
        agentpc_capture::set_capture_profile(fps as u32, req["max_width"].as_u64().unwrap_or(0) as u32);
    }
    let (capture, frame_rx) = match start_window_capture(hwnd_val) {
        Ok(v) => v,
        Err(e) => {
            app.tap.lock().unwrap().remove(&tap_key);
            return Err(anyhow!("capture: {e}"));
        }
    };
    let stop = Arc::new(AtomicBool::new(false));
    app.sessions.lock().unwrap().insert(hwnd.clone(), Session { capture: Some(capture), stop: stop.clone(), tap_key });

    let (frames_tx, frames_rx) = tokio::sync::watch::channel(None);
    if let Some((url, agent, sink, opts)) = policy {
        let (hwnd, dispatcher, stop) = (hwnd.clone(), dispatcher.clone(), stop.clone());
        tokio::spawn(async move {
            if let Err(e) = play::run_policy(hwnd.clone(), url, agent, source, dispatcher, sink, opts.cfg, frames_rx, stop).await {
                emit(json!({ "event": "policy_error", "hwnd": hwnd, "error": format!("{e:#}") }));
            }
        });
    }

    let app2 = app.clone();
    tokio::spawn(async move {
        let outcome = record_loop(&hwnd, source, dispatcher, frame_rx, stop, cfg, upload.clone(), input_events, frames_tx).await;
        // the session may still be registered if capture ended on its own (window closed)
        if let Some(mut s) = app2.sessions.lock().unwrap().remove(&hwnd) {
            if let Some(c) = s.capture.take() {
                let _ = c.stop();
            }
            app2.tap.lock().unwrap().remove(&s.tap_key);
        }
        match outcome {
            Ok(dir) => emit(json!({ "event": "finished", "hwnd": hwnd, "dir": dir, "uploaded": upload.is_some(), "error": null })),
            Err(e) => emit(json!({ "event": "finished", "hwnd": hwnd, "dir": null, "uploaded": false, "error": format!("{e:#}") })),
        }
    });
    Ok(json!({ "session": session_id, "dir": dir }))
}

fn stop(app: &Arc<App>, req: &Value) -> Result<Value> {
    let hwnd = req["hwnd"].as_str().context("hwnd")?;
    let mut s = app.sessions.lock().unwrap().remove(hwnd).ok_or_else(|| anyhow!("not recording that window"))?;
    app.tap.lock().unwrap().remove(&s.tap_key);
    s.stop.store(true, Ordering::Relaxed);
    // Stopping capture closes the frame channel; the record loop then finalizes (never aborted, so
    // the shard is never left `.partial`) and reports through the "finished" event.
    if let Some(c) = s.capture.take() {
        std::thread::spawn(move || {
            let _ = c.stop();
        });
    }
    Ok(json!({}))
}

/// agentpc-client's `run_record_loop`: pair each captured frame with the human's effective action
/// and raw input events, write them on a recorder thread, then finalize and upload.
#[allow(clippy::too_many_arguments)]
async fn record_loop(
    hwnd: &str,
    source: SourceId,
    dispatcher: Arc<Mutex<Dispatcher>>,
    mut frame_rx: tmpsc::Receiver<CapturedFrame>,
    stop: Arc<AtomicBool>,
    cfg: RecordConfig,
    upload_cfg: Option<UploadConfig>,
    input_events: Arc<agentpc_core::RawInputBuffer>,
    frames_tx: tokio::sync::watch::Sender<Option<Arc<Vec<u8>>>>,
) -> Result<PathBuf> {
    let (tx, rx) = mpsc::sync_channel::<FrameRecord>(RECORD_QUEUE_CAPACITY);
    let dropped = Arc::new(AtomicU64::new(0));
    let thread_dropped = dropped.clone();
    log(format!("recording {} -> {}", cfg.session_id, cfg.out_dir.display()));
    let hwnd = hwnd.to_string();
    let handle = std::thread::spawn(move || -> Result<PathBuf> {
        let loose = cfg.loose_dir.clone();
        let session = cfg.session_id.clone();
        let mut recorder = Recorder::new(cfg)?;
        let mut last: Option<String> = None;
        let mut announce = |recorder: &Recorder| {
            let key = recorder.last_chunk_key().map(String::from);
            if let (Some(dir), Some(k)) = (&loose, &key) {
                if last.as_ref() != Some(k) {
                    emit(json!({ "event": "chunk", "hwnd": hwnd, "session": session, "key": k, "dir": dir }));
                }
            }
            last = key;
        };
        while let Ok(frame) = rx.recv() {
            recorder.push(frame)?;
            announce(&recorder);
        }
        recorder.note_recorder_gap(source, thread_dropped.load(Ordering::Relaxed), agentpc_core::now_us());
        recorder.flush_pending()?;
        announce(&recorder);
        let dir = recorder.finish()?;
        if let Some(u) = upload_cfg {
            log(format!("uploading {} to {}", dir.display(), u.repo));
            upload::upload(&dir, &u)?;
        }
        Ok(dir)
    });

    let mut pressure_logged = false;
    loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let frame = tokio::select! {
            frame = frame_rx.recv() => match frame { Some(frame) => frame, None => break },
            _ = tokio::time::sleep(Duration::from_millis(50)) => continue,
        };
        // play mode: the policy always gets the newest frame
        if !frames_tx.is_closed() {
            frames_tx.send_replace(Some(Arc::new(frame.jpeg.clone())));
        }
        let now_us = agentpc_core::now_us();
        let _ = dispatcher.lock().unwrap().tick(now_us);
        let effective = dispatcher.lock().unwrap().peek(source, now_us);
        let (action, provenance) = effective
            .map(|value| (value.action, value.source))
            .unwrap_or_else(|| (agentpc_core::Action::default(), agentpc_core::Provenance::Human));
        let (raw_inputs, input_dropped) = input_events.drain();
        let record = FrameRecord {
            source_id: source,
            frame_id: frame.id,
            captured_t_us: frame.captured_t_us,
            t_us: now_us,
            heartbeat: false,
            jpeg: frame.jpeg,
            action,
            source: provenance,
            raw_inputs,
            input_dropped,
            handoffs: Vec::new(),
            proposals: Vec::new(),
        };
        match tx.try_send(record) {
            Ok(()) => {}
            Err(mpsc::TrySendError::Full(_)) => {
                dropped.fetch_add(1, Ordering::Relaxed);
                if !pressure_logged {
                    pressure_logged = true;
                    log("recorder queue full; dropped frames are written as a gap event");
                }
            }
            Err(mpsc::TrySendError::Disconnected(_)) => break,
        }
    }

    drop(tx);
    tokio::task::spawn_blocking(move || handle.join().map_err(|_| anyhow!("recorder thread panicked"))?)
        .await
        .map_err(|e| anyhow!("{e}"))?
}

fn handle(app: &Arc<App>, req: &Value) -> Result<Value> {
    match req["op"].as_str().unwrap_or("") {
        "devices" => Ok(json!({ "connected": agentpc_capture::connected_pads(), "enabled": agentpc_capture::devices() })),
        "set_devices" => {
            agentpc_capture::set_devices(req["mask"].as_u64().context("mask")? as u8);
            Ok(json!({ "enabled": agentpc_capture::devices() }))
        }
        "start" => start(app, req),
        "stop" => stop(app, req),
        "list" => Ok(json!({ "recording": app.sessions.lock().unwrap().keys().cloned().collect::<Vec<_>>() })),
        op => Err(anyhow!("unknown op {op}")),
    }
}

#[tokio::main]
async fn main() {
    let app = Arc::new(App { tap: Arc::new(Mutex::new(HashMap::new())), sessions: Mutex::new(HashMap::new()) });
    start_human_tap(app.tap.clone(), |m: String| log(m));

    let (line_tx, mut line_rx) = tmpsc::unbounded_channel::<String>();
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines() {
            let Ok(line) = line else { break };
            if line_tx.send(line).is_err() {
                break;
            }
        }
    });

    while let Some(line) = line_rx.recv().await {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let req: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                emit(json!({ "ok": false, "error": format!("bad json: {e}") }));
                continue;
            }
        };
        let id = req["id"].clone();
        match handle(&app, &req) {
            Ok(Value::Object(mut m)) => {
                m.insert("id".into(), id);
                m.insert("ok".into(), true.into());
                emit(Value::Object(m));
            }
            Ok(_) => emit(json!({ "id": id, "ok": true })),
            Err(e) => emit(json!({ "id": id, "ok": false, "error": format!("{e:#}") })),
        }
    }

    // stdin closed (player-one quit): stop every recording and let each finalize before exiting
    let hwnds: Vec<String> = app.sessions.lock().unwrap().keys().cloned().collect();
    for h in &hwnds {
        let _ = stop(&app, &json!({ "hwnd": h }));
    }
    for _ in 0..600 {
        if app.sessions.lock().unwrap().is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}
