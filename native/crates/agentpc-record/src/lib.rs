//! agentpc-record — turns the play loop's frames + the Arbiter's effective actions into
//! WebDataset-style shards, labeled by window, for upload to a Hugging Face dataset.
//!
//! One [`FrameRecord`] per captured frame carries the JPEG, its timestamp, the effective [`Action`]
//! the Arbiter chose, and its [`Provenance`]. The [`Recorder`] buffers these, closes a chunk every
//! ~`chunk_secs`, and writes per chunk into a single session tar shard:
//!
//!   {key}.p0.mp4       — the chunk's frames encoded via ffmpeg
//!   {key}.p0.jsonl     — one line per frame: the effective action + `source` provenance
//!   {key}.meta.json    — session/chunk identity, the window `label`, and a per-source frame tally
//!
//! plus `index.json` for the session. **Label = the window title** (the user's choice); provenance
//! is per-frame `source`, so a "pure AI" session is one whose every frame is `agent:*` — hybrid
//! human frames are visible and filterable, not baked into the label. See docs/SCHEMA.md.
//!
//! Simplifications vs strict MIRA (tracked, not hidden): actions use our flat `keys_held` shape
//! rather than MIRA's 9-wide multi-hot; one shard per session (no ~1GB rolling); `index.json` is our
//! shape, not yet round-tripped through MIRA's loader (SCHEMA.md M4).

use std::fs::{self, File, OpenOptions};
#[cfg(test)]
use std::io::Read;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};

use agentpc_core::{Action, Axes, Buttons, Mouse, Provenance};

pub mod trajectory;
pub mod upload;

/// One captured frame with the action that actually drove the window at that instant.
pub struct FrameRecord {
    pub source_id: agentpc_core::SourceId,
    pub frame_id: u64,
    pub captured_t_us: u64,
    pub t_us: u64,
    pub heartbeat: bool,
    pub jpeg: Vec<u8>,
    pub action: Action,
    pub source: Provenance,
    pub raw_inputs: Vec<agentpc_core::RawInputEvent>,
    pub input_dropped: u64,
    pub handoffs: Vec<HandoffRecord>,
    pub proposals: Vec<PolicyProposalRecord>,
}

#[derive(Clone, Debug)]
pub struct HandoffRecord {
    pub source_id: agentpc_core::SourceId,
    pub t_us: u64,
    pub from: Option<Provenance>,
    pub to: Option<Provenance>,
    pub reason: trajectory::HandoffReason,
}

#[derive(Clone, Debug)]
pub struct PolicyProposalRecord {
    pub proposal_id: String,
    pub session_id: String,
    pub source_id: agentpc_core::SourceId,
    pub agent_id: agentpc_core::AgentId,
    pub observed_frame_ids: Vec<u64>,
    pub manifest_sha256: Option<String>,
    pub action: Action,
    pub inference_started_t_us: u64,
    pub inference_finished_t_us: u64,
    pub inference_clock: String,
    pub received_t_us: u64,
}

#[derive(Clone)]
pub struct RecordConfig {
    /// Local scratch root; a `<session_id>/` dir is created under it.
    pub out_dir: PathBuf,
    /// The window title — the session label the user asked for.
    pub label: String,
    pub session_id: String,
    pub chunk_secs: u64,
    /// ffmpeg binary (default "ffmpeg").
    pub ffmpeg_bin: String,
    /// player-one: also write each chunk's files loose into this dir as the chunk closes, so they
    /// can be shipped before the session ends. `.meta.json` is written last and marks completion.
    pub loose_dir: Option<PathBuf>,
}

impl RecordConfig {
    pub fn new(out_dir: impl Into<PathBuf>, label: impl Into<String>) -> Self {
        Self {
            out_dir: out_dir.into(),
            label: label.into(),
            session_id: default_session_id(),
            chunk_secs: 4,
            ffmpeg_bin: "ffmpeg".into(),
            loose_dir: None,
        }
    }
}

/// A wall-clock-derived id: `YYYYmmddTHHMMSS-<nanos>`. Wall clock is fine for an *identifier*
/// (rlrec's clock rules are about frame timing, which uses the monotonic `t_us` on each record).
pub fn default_session_id() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("s{}-{:x}", now.as_secs(), now.subsec_nanos())
}

fn should_close(first_t_us: u64, next_t_us: u64, chunk_secs: u64) -> bool {
    next_t_us.saturating_sub(first_t_us) >= chunk_secs * 1_000_000
}

#[derive(Serialize)]
struct ActionLine<'a> {
    frame: u32,
    frame_id: u64,
    captured_t_us: u64,
    t_us: u64,
    heartbeat: bool,
    keys_held: &'a [String],
    axes: &'a Axes,
    buttons: &'a Buttons,
    mouse: &'a Mouse,
    source: String,
}

#[derive(Serialize)]
struct ChunkMeta {
    session_id: String,
    chunk: u32,
    label: String,
    player_ids: [&'static str; 1],
    fps: u32,
    width: u32,
    height: u32,
    frame_count: usize,
    t0_unix_ns: u128,
    /// Per-source frame tally, e.g. {"agent:7": 80}. A pure-AI chunk has no "human" key.
    sources: std::collections::BTreeMap<String, usize>,
    tool_version: &'static str,
    schema_version: u32,
    trajectory_schema: &'static str,
}

#[derive(Serialize)]
struct IndexChunk {
    chunk: u32,
    key: String,
    frames: usize,
    duration_us: u64,
}

#[derive(Serialize)]
struct SessionIndex {
    session_id: String,
    label: String,
    shard: String,
    chunks: Vec<IndexChunk>,
}

const SHARD_NAME: &str = "dataset_00000.tar";
const SHARD_PARTIAL: &str = "dataset_00000.tar.partial";
const INDEX_NAME: &str = "index.json";
const INDEX_PARTIAL: &str = "index.json.partial";
const COMPLETE_NAME: &str = "complete.json";
const COMPLETE_PARTIAL: &str = "complete.json.partial";

#[derive(Serialize)]
struct CompletedFile {
    name: &'static str,
    size: u64,
    sha256: String,
}

#[derive(Serialize)]
struct CompletionManifest {
    schema: &'static str,
    session_id: String,
    files: [CompletedFile; 2],
}

/// Find session directories that never reached the completion-marker commit point. Callers may
/// resume, inspect, or quarantine them; upload must only accept directories absent from this list.
pub fn scan_incomplete_sessions(root: &Path) -> Result<Vec<PathBuf>> {
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut incomplete = Vec::new();
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        if path.is_dir() && !path.join(COMPLETE_NAME).is_file() {
            incomplete.push(path);
        }
    }
    incomplete.sort();
    Ok(incomplete)
}

/// Accumulates frames and writes one session shard. `push` per frame; `finish` at the end.
pub struct Recorder {
    cfg: RecordConfig,
    session_dir: PathBuf,
    t0_unix_ns: u128,
    tar: tar::Builder<File>,
    buf: Vec<FrameRecord>,
    chunk_idx: u32,
    index: Vec<IndexChunk>,
    next_event_seq: u64,
    recorder_gap: Option<(agentpc_core::SourceId, u64, u64)>,
}

impl Recorder {
    pub fn new(cfg: RecordConfig) -> Result<Self> {
        let session_dir = cfg.out_dir.join(&cfg.session_id);
        fs::create_dir_all(&session_dir)
            .with_context(|| format!("create session dir {}", session_dir.display()))?;
        let partial = session_dir.join(SHARD_PARTIAL);
        let tar_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&partial)
            .with_context(|| format!("create new partial shard {}", partial.display()))?;
        let tar = tar::Builder::new(tar_file);
        let t0_unix_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        Ok(Self {
            cfg,
            session_dir,
            t0_unix_ns,
            tar,
            buf: Vec::new(),
            chunk_idx: 0,
            index: Vec::new(),
            next_event_seq: 0,
            recorder_gap: None,
        })
    }

    pub fn session_dir(&self) -> &Path {
        &self.session_dir
    }

    /// Write the buffered frames as a chunk now (finish() would do it too, but consumes self).
    pub fn flush_pending(&mut self) -> Result<()> {
        self.flush_chunk()
    }

    /// Key of the most recently written chunk (`None` before the first one closes).
    pub fn last_chunk_key(&self) -> Option<&str> {
        self.index.last().map(|c| c.key.as_str())
    }

    /// Record one frame. Closes and writes the current chunk when it spans `chunk_secs`.
    pub fn push(&mut self, rec: FrameRecord) -> Result<()> {
        if let Some(first) = self.buf.first() {
            if should_close(first.t_us, rec.t_us, self.cfg.chunk_secs) {
                self.flush_chunk()?;
            }
        }
        self.buf.push(rec);
        Ok(())
    }

    /// Preserve recorder queue loss in the canonical event stream instead of silently shortening
    /// the trajectory. Multiple reports are accumulated into one session-level loss marker.
    pub fn note_recorder_gap(
        &mut self,
        source_id: agentpc_core::SourceId,
        dropped: u64,
        t_us: u64,
    ) {
        if dropped == 0 {
            return;
        }
        match &mut self.recorder_gap {
            Some((_, total, last_t_us)) => {
                *total += dropped;
                *last_t_us = t_us;
            }
            slot @ None => *slot = Some((source_id, dropped, t_us)),
        }
    }

    /// Flush the last partial chunk, write `index.json`, finalize the tar. Returns the session dir.
    pub fn finish(mut self) -> Result<PathBuf> {
        if !self.buf.is_empty() {
            self.flush_chunk()?;
        }
        let index = SessionIndex {
            session_id: self.cfg.session_id.clone(),
            label: self.cfg.label.clone(),
            shard: SHARD_NAME.to_string(),
            chunks: self.index,
        };
        let index_bytes = serde_json::to_vec_pretty(&index)?;
        let index_partial = self.session_dir.join(INDEX_PARTIAL);
        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&index_partial)?;
        f.write_all(&index_bytes)?;
        f.sync_all()?;
        drop(f);
        self.tar.finish()?;
        let tar_file = self.tar.into_inner()?;
        tar_file.sync_all()?;
        drop(tar_file);

        let shard_final = self.session_dir.join(SHARD_NAME);
        let index_final = self.session_dir.join(INDEX_NAME);
        fs::rename(self.session_dir.join(SHARD_PARTIAL), &shard_final)?;
        fs::rename(&index_partial, &index_final)?;

        let completion = CompletionManifest {
            schema: "agentpc.recording-complete/v1",
            session_id: self.cfg.session_id.clone(),
            files: [
                completed_file(SHARD_NAME, &shard_final)?,
                completed_file(INDEX_NAME, &index_final)?,
            ],
        };
        let complete_partial = self.session_dir.join(COMPLETE_PARTIAL);
        let mut complete = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&complete_partial)?;
        complete.write_all(&serde_json::to_vec_pretty(&completion)?)?;
        complete.sync_all()?;
        drop(complete);
        fs::rename(complete_partial, self.session_dir.join(COMPLETE_NAME))?;
        Ok(self.session_dir)
    }

    fn flush_chunk(&mut self) -> Result<()> {
        let frames = std::mem::take(&mut self.buf);
        if frames.is_empty() {
            return Ok(());
        }
        let chunk = self.chunk_idx;
        let key = format!("{}_c{:05}", self.cfg.session_id, chunk);
        let first_t = frames.first().unwrap().t_us;
        let last_t = frames.last().unwrap().t_us;
        let duration_us = last_t.saturating_sub(first_t).max(1);
        // Real capture rate for this chunk (frames may be sparse); clamp to a sane range.
        let fps = ((frames.len() as f64) / (duration_us as f64 / 1_000_000.0))
            .round()
            .clamp(1.0, 60.0) as u32;

        let (width, height) = jpeg_dims(&frames[0].jpeg).unwrap_or((0, 0));

        // .p0.mp4 — encode the chunk's JPEGs.
        let mp4 = encode_mp4(&self.cfg.ffmpeg_bin, &frames, fps)
            .with_context(|| format!("ffmpeg encode chunk {chunk}"))?;
        self.append(&format!("{key}.p0.mp4"), &mp4)?;

        // .p0.jsonl — one line per frame: effective action + source provenance.
        let mut jsonl = String::new();
        let mut sources: std::collections::BTreeMap<String, usize> = Default::default();
        for (i, r) in frames.iter().enumerate() {
            let src = r.source.as_source_str();
            *sources.entry(src.clone()).or_default() += 1;
            let line = ActionLine {
                frame: i as u32,
                frame_id: r.frame_id,
                captured_t_us: r.captured_t_us,
                t_us: r.t_us,
                heartbeat: r.heartbeat,
                keys_held: &r.action.keys_held,
                axes: &r.action.axes,
                buttons: &r.action.buttons,
                mouse: &r.action.mouse,
                source: src,
            };
            jsonl.push_str(&serde_json::to_string(&line)?);
            jsonl.push('\n');
        }
        self.append(&format!("{key}.p0.jsonl"), jsonl.as_bytes())?;

        let mut events = String::new();
        for r in &frames {
            if r.input_dropped > 0 {
                let event = trajectory::TrajectoryEvent::gap(
                    self.next_event_seq,
                    r.t_us,
                    r.source_id,
                    trajectory::GapComponent::Input,
                    r.input_dropped,
                    "bounded raw-input queue overflow",
                );
                self.next_event_seq += 1;
                events.push_str(&serde_json::to_string(&event)?);
                events.push('\n');
            }
            for input in &r.raw_inputs {
                let event =
                    trajectory::TrajectoryEvent::raw_input(self.next_event_seq, input.clone());
                self.next_event_seq += 1;
                events.push_str(&serde_json::to_string(&event)?);
                events.push('\n');
            }
            for handoff in &r.handoffs {
                let event =
                    trajectory::TrajectoryEvent::handoff(self.next_event_seq, handoff.clone());
                self.next_event_seq += 1;
                events.push_str(&serde_json::to_string(&event)?);
                events.push('\n');
            }
            for proposal in &r.proposals {
                let event = trajectory::TrajectoryEvent::policy_proposal(
                    self.next_event_seq,
                    proposal.clone(),
                );
                self.next_event_seq += 1;
                events.push_str(&serde_json::to_string(&event)?);
                events.push('\n');
            }
            let frame_event = trajectory::TrajectoryEvent::frame(
                self.next_event_seq,
                r.source_id,
                r.frame_id,
                r.captured_t_us,
                r.t_us,
                r.heartbeat,
            );
            self.next_event_seq += 1;
            let action_event = trajectory::TrajectoryEvent::effective_action(
                self.next_event_seq,
                r.t_us,
                r.source_id,
                r.action.clone(),
                r.source,
            );
            self.next_event_seq += 1;
            for event in [frame_event, action_event] {
                events.push_str(&serde_json::to_string(&event)?);
                events.push('\n');
            }
        }
        if let Some((source_id, dropped, t_us)) = self.recorder_gap.take() {
            let event = trajectory::TrajectoryEvent::gap(
                self.next_event_seq,
                t_us,
                source_id,
                trajectory::GapComponent::Recorder,
                dropped,
                "bounded recorder queue full",
            );
            self.next_event_seq += 1;
            events.push_str(&serde_json::to_string(&event)?);
            events.push('\n');
        }
        self.append(&format!("{key}.events.jsonl"), events.as_bytes())?;

        // .meta.json
        let meta = ChunkMeta {
            session_id: self.cfg.session_id.clone(),
            chunk,
            label: self.cfg.label.clone(),
            player_ids: ["p0"],
            fps,
            width,
            height,
            frame_count: frames.len(),
            t0_unix_ns: self.t0_unix_ns,
            sources,
            tool_version: concat!("agentpc-record ", env!("CARGO_PKG_VERSION")),
            schema_version: 1,
            trajectory_schema: trajectory::SCHEMA,
        };
        self.append(
            &format!("{key}.meta.json"),
            serde_json::to_string_pretty(&meta)?.as_bytes(),
        )?;

        self.index.push(IndexChunk {
            chunk,
            key,
            frames: frames.len(),
            duration_us,
        });
        self.chunk_idx += 1;
        Ok(())
    }

    fn append(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        let mut header = tar::Header::new_gnu();
        header.set_size(bytes.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        self.tar.append_data(&mut header, name, bytes)?;
        if let Some(dir) = &self.cfg.loose_dir {
            fs::create_dir_all(dir)?;
            let tmp = dir.join(format!("{name}.tmp"));
            fs::write(&tmp, bytes)?;
            fs::rename(&tmp, dir.join(name))?;
        }
        Ok(())
    }
}

fn completed_file(name: &'static str, path: &Path) -> Result<CompletedFile> {
    let bytes = fs::read(path)?;
    Ok(CompletedFile {
        name,
        size: bytes.len() as u64,
        sha256: format!("{:x}", Sha256::digest(&bytes)),
    })
}

/// Read JPEG dimensions from the header (cheap; does not fully decode).
fn jpeg_dims(jpeg: &[u8]) -> Option<(u32, u32)> {
    image::ImageReader::new(std::io::Cursor::new(jpeg))
        .with_guessed_format()
        .ok()?
        .into_dimensions()
        .ok()
}

/// Pipe the chunk's JPEG frames into ffmpeg (image2pipe) and return the encoded mp4 bytes.
fn encode_mp4(ffmpeg_bin: &str, frames: &[FrameRecord], fps: u32) -> Result<Vec<u8>> {
    // ffmpeg can't write a seekable mp4 (moov atom) to a pipe, so go via a temp file.
    let tmp = std::env::temp_dir().join(format!("agentpc_chunk_{}.mp4", default_session_id()));
    // `-f mjpeg` treats stdin as a stream of concatenated JPEGs (more robust than image2pipe for a
    // piped frame stream). Output framerate is set with `-r` after the input.
    let mut child = Command::new(ffmpeg_bin)
        .args([
            "-y",
            "-f",
            "mjpeg",
            "-framerate",
            &fps.to_string(),
            "-i",
            "-",
            "-r",
            &fps.to_string(),
            // Force even dimensions: libx264 + yuv420p rejects odd width/height (a real window is
            // often an odd size), failing with "Invalid argument". Round down to the nearest even.
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ])
        .arg(&tmp)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("spawn ffmpeg ({ffmpeg_bin})"))?;

    {
        let mut stdin = child.stdin.take().context("ffmpeg stdin")?;
        for r in frames {
            stdin.write_all(&r.jpeg)?;
        }
        // stdin dropped here → EOF, ffmpeg finalizes.
    }
    let output = child.wait_with_output()?;
    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        let tail: String = err
            .lines()
            .rev()
            .take(4)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join(" | ");
        anyhow::bail!("ffmpeg exited with {}: {tail}", output.status);
    }
    let bytes = fs::read(&tmp).with_context(|| format!("read {}", tmp.display()))?;
    let _ = fs::remove_file(&tmp);
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_closes_on_span() {
        assert!(!should_close(1_000, 3_000_000, 4)); // < 4s span
        assert!(should_close(1_000, 4_001_000, 4)); // >= 4s span
        assert!(should_close(0, 4_000_000, 4)); // exactly 4s
    }

    fn ffmpeg_available() -> bool {
        Command::new("ffmpeg")
            .arg("-version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    fn tiny_jpeg(w: u32, h: u32, val: u8) -> Vec<u8> {
        let buf = vec![val; (w * h * 3) as usize];
        let mut out = Vec::new();
        image::codecs::jpeg::JpegEncoder::new(&mut out)
            .encode(&buf, w, h, image::ExtendedColorType::Rgb8)
            .unwrap();
        out
    }

    #[test]
    fn records_a_shard_with_mp4_jsonl_and_meta() {
        if !ffmpeg_available() {
            eprintln!("skipping: ffmpeg not on PATH");
            return;
        }
        let out = std::env::temp_dir().join(format!("agentpc_rec_test_{}", default_session_id()));
        let mut rec = Recorder::new(RecordConfig::new(&out, "Untitled - Notepad")).unwrap();
        assert_eq!(
            scan_incomplete_sessions(&out).unwrap(),
            [rec.session_dir().to_path_buf()]
        );
        for i in 0..3u64 {
            rec.push(FrameRecord {
                source_id: 3,
                frame_id: i,
                captured_t_us: i * 200_000,
                t_us: i * 200_000,
                heartbeat: false,
                jpeg: tiny_jpeg(64, 48, (i * 80) as u8),
                action: Action {
                    keys_held: vec!["A".into()],
                    ..Default::default()
                },
                source: Provenance::Agent(7),
                raw_inputs: Vec::new(),
                input_dropped: 0,
                handoffs: Vec::new(),
                proposals: Vec::new(),
            })
            .unwrap();
        }
        rec.note_recorder_gap(3, 2, 700_000);
        let dir = rec.finish().unwrap();
        assert!(dir.join("dataset_00000.tar").exists());
        assert!(dir.join("index.json").exists());
        assert!(dir.join(COMPLETE_NAME).exists());
        assert!(!dir.join(SHARD_PARTIAL).exists());
        assert!(!dir.join(INDEX_PARTIAL).exists());
        assert!(scan_incomplete_sessions(&out).unwrap().is_empty());

        let mut ar = tar::Archive::new(File::open(dir.join(SHARD_NAME)).unwrap());
        let names: Vec<String> = ar
            .entries()
            .unwrap()
            .map(|e| e.unwrap().path().unwrap().to_string_lossy().into_owned())
            .collect();
        assert!(
            names.iter().any(|n| n.ends_with("_c00000.p0.mp4")),
            "mp4 missing: {names:?}"
        );
        assert!(
            names.iter().any(|n| n.ends_with("_c00000.p0.jsonl")),
            "jsonl missing"
        );
        assert!(
            names.iter().any(|n| n.ends_with("_c00000.events.jsonl")),
            "events missing"
        );
        assert!(
            names.iter().any(|n| n.ends_with("_c00000.meta.json")),
            "meta missing"
        );

        let idx: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("index.json")).unwrap())
                .unwrap();
        assert_eq!(idx["label"], "Untitled - Notepad");
        assert_eq!(idx["chunks"][0]["frames"], 3);

        let completed: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join(COMPLETE_NAME)).unwrap()).unwrap();
        assert_eq!(completed["schema"], "agentpc.recording-complete/v1");
        for file in completed["files"].as_array().unwrap() {
            let name = file["name"].as_str().unwrap();
            let bytes = std::fs::read(dir.join(name)).unwrap();
            assert_eq!(file["size"], bytes.len());
            assert_eq!(file["sha256"], format!("{:x}", Sha256::digest(&bytes)));
        }

        let mut ar = tar::Archive::new(File::open(dir.join(SHARD_NAME)).unwrap());
        let mut event_jsonl = String::new();
        for entry in ar.entries().unwrap() {
            let mut entry = entry.unwrap();
            if entry
                .path()
                .unwrap()
                .to_string_lossy()
                .ends_with(".events.jsonl")
            {
                entry.read_to_string(&mut event_jsonl).unwrap();
            }
        }
        let events: Vec<serde_json::Value> = event_jsonl
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let gap = events
            .iter()
            .find(|event| event["event"] == "gap")
            .expect("recorder gap event missing");
        assert_eq!(gap["component"], "recorder");
        assert_eq!(gap["dropped"], 2);

        let _ = std::fs::remove_dir_all(&out);
    }

    #[test]
    fn action_line_shape_is_flat_with_source() {
        let a = Action {
            keys_held: vec!["A".into()],
            ..Default::default()
        };
        let line = ActionLine {
            frame: 0,
            frame_id: 9,
            captured_t_us: 100,
            t_us: 123,
            heartbeat: true,
            keys_held: &a.keys_held,
            axes: &a.axes,
            buttons: &a.buttons,
            mouse: &a.mouse,
            source: Provenance::Agent(7).as_source_str(),
        };
        let v: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&line).unwrap()).unwrap();
        assert_eq!(v["keys_held"][0], "A");
        assert_eq!(v["source"], "agent:7");
        assert_eq!(v["frame_id"], 9);
        assert_eq!(v["heartbeat"], true);
        assert_eq!(v["buttons"]["a"], false);
        assert!(v.get("agent").is_none(), "no top-level agent field");
    }
}
