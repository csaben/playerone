//! agentpc-capture — reads the world.
//!
//! Two profiles from one WGC core (see docs/ARCHITECTURE.md):
//! - **Play** (implemented here): per-**window** capture, ~5 fps JPEG, works while the window is
//!   NOT foreground (WGC composites from DWM). Feeds both the wire uplink and the box wallpaper —
//!   one local tee. Extracted from demuxer-workspace.
//! - **Record** (TODO): **monitor**, full-rate BGRA → encoder, strict SCHEMA.md §7.3 assertions.
//!   Plus the human raw-input tap (WM_INPUT) that becomes a `Source` for the Arbiter. From rlrec.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Serialize;
use tokio::sync::mpsc;

use windows_capture::capture::{CaptureControl, Context, GraphicsCaptureApiHandler};
use windows_capture::frame::Frame;
use windows_capture::graphics_capture_api::{GraphicsCaptureApi, InternalCaptureControl};
use windows_capture::settings::{
    ColorFormat, CursorCaptureSettings, DirtyRegionSettings, DrawBorderSettings,
    MinimumUpdateIntervalSettings, SecondaryWindowSettings, Settings,
};
use windows_capture::window::Window;

use windows::Win32::Foundation::{BOOL, HWND, LPARAM};
use windows::Win32::UI::WindowsAndMessaging::{
    EnumWindows, GetWindowTextLengthW, GetWindowTextW, IsWindowVisible,
};

mod human_tap;
pub mod snapshot;
pub use human_tap::{
    connected_pads, devices, set_devices, start_human_tap, virtual_pad_attached, virtual_pad_detached, virtual_pad_slot, TapRegistry,
    TapTarget, DEV_KEYBOARD, DEV_MOUSE, DEV_PADS,
};

/// Throttle capture to ~5 fps regardless of how fast WGC delivers frames.
const FRAME_INTERVAL: Duration = Duration::from_millis(200);
const JPEG_QUALITY: u8 = 55;

/// player-one: process-wide overrides of the play capture rate and an optional downscale (0 = keep
/// full size), so recordings meant for training can run at 20 fps without full-res JPEG cost.
static FRAME_INTERVAL_US: AtomicU64 = AtomicU64::new(FRAME_INTERVAL.as_micros() as u64);
static MAX_WIDTH: AtomicU32 = AtomicU32::new(0);

pub fn set_capture_profile(fps: u32, max_width: u32) {
    FRAME_INTERVAL_US.store(1_000_000 / fps.clamp(1, 60) as u64, Ordering::Relaxed);
    MAX_WIDTH.store(max_width, Ordering::Relaxed);
}

fn frame_interval() -> Duration {
    Duration::from_micros(FRAME_INTERVAL_US.load(Ordering::Relaxed))
}

/// A visible, titled top-level window — the picker feed for a box binding.
/// `hwnd` is an isize serialized as a string; JS numbers can't hold all 64-bit handles safely.
#[derive(Debug, Clone, Serialize)]
pub struct WinInfo {
    pub hwnd: String,
    pub title: String,
}

/// A frame timestamped and identified by the capture backend, before any queueing or heartbeat.
#[derive(Debug, Clone)]
pub struct CapturedFrame {
    pub id: u64,
    pub captured_t_us: u64,
    pub jpeg: Vec<u8>,
}

/// WGC frame handler: BGRA → RGB → JPEG, throttled, dropped-on-backpressure (never blocks capture).
pub struct Capture {
    tx: mpsc::Sender<CapturedFrame>,
    last: Instant,
    next_frame_id: u64,
}

impl GraphicsCaptureApiHandler for Capture {
    type Flags = mpsc::Sender<CapturedFrame>;
    type Error = Box<dyn std::error::Error + Send + Sync>;

    fn new(ctx: Context<Self::Flags>) -> Result<Self, Self::Error> {
        Ok(Capture {
            tx: ctx.flags,
            last: Instant::now() - Duration::from_secs(1),
            next_frame_id: 0,
        })
    }

    fn on_frame_arrived(
        &mut self,
        frame: &mut Frame,
        _ctl: InternalCaptureControl,
    ) -> Result<(), Self::Error> {
        let interval = frame_interval();
        let el = self.last.elapsed();
        if el < interval {
            return Ok(());
        }
        // Steady cadence: step by whole intervals instead of snapping to now (which drifts slow).
        self.last = if el < interval * 2 { self.last + interval } else { Instant::now() };

        let w = frame.width();
        let h = frame.height();
        if w == 0 || h == 0 {
            return Ok(());
        }

        let mut fb = frame.buffer()?;
        let pitch = fb.row_pitch() as usize; // bytes per row, may include padding
        let raw = fb.as_raw_buffer(); // BGRA8, possibly padded rows

        let max_w = MAX_WIDTH.load(Ordering::Relaxed);
        let jpeg = if max_w > 0 && w > max_w {
            encode_bgra_jpeg_scaled(raw, pitch, w, h, w.div_ceil(max_w))
        } else {
            encode_bgra_jpeg(raw, pitch, w, h)
        };
        if let Some(jpeg) = jpeg {
            // Drop the frame if the ws sender is backed up — never block capture.
            let frame = CapturedFrame { id: self.next_frame_id, captured_t_us: agentpc_core::now_us(), jpeg };
            self.next_frame_id = self.next_frame_id.wrapping_add(1);
            let _ = self.tx.try_send(frame);
        }
        Ok(())
    }

    fn on_closed(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

pub type CaptureCtl = CaptureControl<Capture, Box<dyn std::error::Error + Send + Sync>>;

/// Start play-profile capture of a window on its own thread. Returns the control handle (call
/// `.stop()` to tear down) and the receiver of JPEG frames (~5 fps).
pub fn start_window_capture(
    hwnd: isize,
) -> Result<(CaptureCtl, mpsc::Receiver<CapturedFrame>), String> {
    let (frame_tx, frame_rx) = mpsc::channel::<CapturedFrame>(3);
    let window = Window::from_raw_hwnd(hwnd as *mut c_void);
    // IsBorderRequired is not available on Windows 10 (including build 19045). Asking for
    // WithoutBorder there makes the entire capture session fail with BorderConfigUnsupported, so
    // retain live preview with the system default and use borderless capture on newer Windows.
    let border = if GraphicsCaptureApi::is_border_settings_supported().unwrap_or(false) {
        DrawBorderSettings::WithoutBorder
    } else {
        DrawBorderSettings::Default
    };
    let settings = Settings::new(
        window,
        CursorCaptureSettings::Default,
        border,
        SecondaryWindowSettings::Default,
        MinimumUpdateIntervalSettings::Default,
        DirtyRegionSettings::Default,
        ColorFormat::Bgra8,
        frame_tx,
    );
    let capture =
        Capture::start_free_threaded(settings).map_err(|e| format!("capture start failed: {e}"))?;
    Ok((capture, frame_rx))
}

/// BGRA (possibly row-padded) → JPEG. Shared by the WGC handler and the GDI preview loop.
fn encode_bgra_jpeg(raw: &[u8], pitch: usize, w: u32, h: u32) -> Option<Vec<u8>> {
    let (wu, hu) = (w as usize, h as usize);
    if raw.len() < (hu - 1) * pitch + wu * 4 {
        return None;
    }
    let mut rgb = vec![0u8; wu * hu * 3];
    for y in 0..hu {
        let row = &raw[y * pitch..y * pitch + wu * 4];
        let drow = &mut rgb[y * wu * 3..(y + 1) * wu * 3];
        for x in 0..wu {
            let s = &row[x * 4..x * 4 + 4]; // B,G,R,A
            drow[x * 3] = s[2]; // R
            drow[x * 3 + 1] = s[1]; // G
            drow[x * 3 + 2] = s[0]; // B
        }
    }
    let mut jpeg = Vec::new();
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, JPEG_QUALITY);
    enc.encode(&rgb, w, h, image::ExtendedColorType::Rgb8).ok()?;
    Some(jpeg)
}

/// BGRA → JPEG at 1/k size, averaging each k×k block (a cheap box filter).
fn encode_bgra_jpeg_scaled(raw: &[u8], pitch: usize, w: u32, h: u32, k: u32) -> Option<Vec<u8>> {
    let (k, wu, hu) = (k as usize, w as usize, h as usize);
    if raw.len() < (hu - 1) * pitch + wu * 4 {
        return None;
    }
    let (ow, oh) = (wu / k, hu / k);
    if ow == 0 || oh == 0 {
        return None;
    }
    let n = (k * k) as u32;
    let mut rgb = vec![0u8; ow * oh * 3];
    let mut acc = vec![[0u32; 3]; ow];
    for oy in 0..oh {
        acc.iter_mut().for_each(|a| *a = [0; 3]);
        for y in oy * k..oy * k + k {
            let row = &raw[y * pitch..y * pitch + ow * k * 4];
            for (ox, a) in acc.iter_mut().enumerate() {
                for s in row[ox * k * 4..(ox + 1) * k * 4].chunks_exact(4) {
                    a[0] += s[2] as u32; // B,G,R,A → R,G,B
                    a[1] += s[1] as u32;
                    a[2] += s[0] as u32;
                }
            }
        }
        let drow = &mut rgb[oy * ow * 3..(oy + 1) * ow * 3];
        for (ox, a) in acc.iter().enumerate() {
            drow[ox * 3] = (a[0] / n) as u8;
            drow[ox * 3 + 1] = (a[1] / n) as u8;
            drow[ox * 3 + 2] = (a[2] / n) as u8;
        }
    }
    let mut jpeg = Vec::new();
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, JPEG_QUALITY);
    enc.encode(&rgb, ow as u32, oh as u32, image::ExtendedColorType::Rgb8).ok()?;
    Some(jpeg)
}

/// Control handle for a GDI preview loop: stopping just flips the flag; the thread exits on its own
/// within one frame interval (no join, so tearing down many previews stays fast).
pub struct GdiPreviewCtl {
    stop: Arc<AtomicBool>,
}

/// A preview capture is WGC where borderless capture is supported (Win11), else a GDI PrintWindow
/// loop — Win10's WGC unconditionally draws the yellow system border on the captured window, which
/// is wrong for previews (the app previews *every* window, so every window would get one).
pub enum PreviewCapture {
    Wgc(CaptureCtl),
    Gdi(GdiPreviewCtl),
}

impl PreviewCapture {
    pub fn stop(self) {
        match self {
            PreviewCapture::Wgc(c) => {
                let _ = c.stop();
            }
            PreviewCapture::Gdi(g) => g.stop.store(true, Ordering::Relaxed),
        }
    }
}

/// Start preview-profile capture of a window: same JPEG stream as `start_window_capture`, but never
/// draws the system capture border on the window. Play sessions keep WGC (on Win10 its border marks
/// exactly the agent-driven windows).
pub fn start_window_capture_preview(
    hwnd: isize,
) -> Result<(PreviewCapture, mpsc::Receiver<CapturedFrame>), String> {
    if GraphicsCaptureApi::is_border_settings_supported().unwrap_or(false) {
        let (capture, frame_rx) = start_window_capture(hwnd)?; // WGC WithoutBorder
        return Ok((PreviewCapture::Wgc(capture), frame_rx));
    }
    unsafe {
        use windows::Win32::UI::WindowsAndMessaging::IsWindow;
        if !IsWindow(HWND(hwnd as *mut c_void)).as_bool() {
            return Err("no such window".to_string());
        }
    }
    let (frame_tx, frame_rx) = mpsc::channel::<CapturedFrame>(3);
    let stop = Arc::new(AtomicBool::new(false));
    let stop_task = stop.clone();
    std::thread::spawn(move || gdi_preview_loop(hwnd, frame_tx, stop_task));
    Ok((PreviewCapture::Gdi(GdiPreviewCtl { stop }), frame_rx))
}

/// GDI previews run at 1 fps (they are wallpaper, not inference input) and are downscaled to this
/// width before JPEG — the pixel convert / encode / base64 / webview-decode pipeline all shrink with
/// it. Full-res frames stay exclusive to the WGC play path (the cradle needs them). PrintWindow
/// forces the target app to render synchronously, so across a desktop's worth of previews the poke
/// rate is what janks the window manager — keep it low, and stagger the loops so the pokes spread
/// out instead of landing in one burst.
const GDI_FRAME_INTERVAL: Duration = Duration::from_millis(1000);
const PREVIEW_MAX_W: i32 = 512;

/// A reusable DIB + memory DC (recreated only when the window resizes — allocating a full-window
/// 32bpp bitmap per frame was multi-GB/s of churn across a desktop's worth of previews).
struct Dib {
    dc: windows::Win32::Graphics::Gdi::HDC,
    bmp: windows::Win32::Graphics::Gdi::HBITMAP,
    bits: *mut c_void,
    w: i32,
    h: i32,
}

unsafe fn make_dib(screen: windows::Win32::Graphics::Gdi::HDC, w: i32, h: i32) -> Option<Dib> {
    use windows::Win32::Graphics::Gdi::{
        CreateCompatibleDC, CreateDIBSection, SelectObject, SetBrushOrgEx, SetStretchBltMode,
        BITMAPINFO, BITMAPINFOHEADER, BI_RGB, DIB_RGB_COLORS, HALFTONE,
    };
    let dc = CreateCompatibleDC(screen);
    let mut bmi = BITMAPINFO::default();
    bmi.bmiHeader = BITMAPINFOHEADER {
        biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
        biWidth: w,
        biHeight: -h, // top-down
        biPlanes: 1,
        biBitCount: 32,
        biCompression: BI_RGB.0,
        ..Default::default()
    };
    let mut bits: *mut c_void = std::ptr::null_mut();
    let bmp = CreateDIBSection(screen, &bmi, DIB_RGB_COLORS, &mut bits, None, 0).ok()?;
    SelectObject(dc, bmp);
    SetStretchBltMode(dc, HALFTONE); // quality downscale when this DC is a StretchBlt target
    let _ = SetBrushOrgEx(dc, 0, 0, None);
    Some(Dib { dc, bmp, bits, w, h })
}

unsafe fn free_dib(d: Dib) {
    use windows::Win32::Graphics::Gdi::{DeleteDC, DeleteObject};
    let _ = DeleteObject(d.bmp);
    let _ = DeleteDC(d.dc);
}

/// 2.5 fps PrintWindow(PW_RENDERFULLCONTENT) → downscale → JPEG loop. Unchanged frames (sparse
/// pixel hash) are skipped, so static windows cost a grab but no encode/emit; uniform frames are
/// dropped so windows that render nothing in the background still count as "no picture" for the
/// Hide-empty filter; minimized windows are skipped entirely (PrintWindow yields junk for them).
fn gdi_preview_loop(hwnd: isize, tx: mpsc::Sender<CapturedFrame>, stop: Arc<AtomicBool>) {
    use windows::Win32::Foundation::RECT;
    use windows::Win32::Graphics::Gdi::{GdiFlush, GetDC, ReleaseDC, SRCCOPY, StretchBlt};
    use windows::Win32::Storage::Xps::{PrintWindow, PRINT_WINDOW_FLAGS};
    use windows::Win32::UI::WindowsAndMessaging::{GetWindowRect, IsIconic, IsWindow};
    const PW_RENDERFULLCONTENT: PRINT_WINDOW_FLAGS = PRINT_WINDOW_FLAGS(2); // DWM content, undocumented but long-stable

    let hwnd_val = hwnd;
    let hwnd = HWND(hwnd as *mut c_void);
    // Stagger: spread the per-window loops across the interval so ~N windows means a steady
    // trickle of PrintWindow pokes rather than N-at-once bursts every tick.
    std::thread::sleep(Duration::from_millis((hwnd_val as u64).wrapping_mul(2654435761) % 1000));
    let mut next_frame_id = 0u64;
    unsafe {
        let screen = GetDC(None);
        if screen.is_invalid() {
            return;
        }
        let mut full: Option<Dib> = None;
        let mut small: Option<Dib> = None;
        let mut last_hash: Option<u64> = None;
        while !stop.load(Ordering::Relaxed) && !tx.is_closed() {
            let t0 = Instant::now();
            if !IsWindow(hwnd).as_bool() {
                break;
            }
            // Cloaked windows (other virtual desktops, suspended UWP ghosts) enumerate as visible
            // but render stale junk — skip them entirely; no frames also keeps Hide-empty honest.
            let mut cloaked: u32 = 0;
            let _ = windows::Win32::Graphics::Dwm::DwmGetWindowAttribute(
                hwnd,
                windows::Win32::Graphics::Dwm::DWMWA_CLOAKED,
                &mut cloaked as *mut u32 as *mut c_void,
                4,
            );
            let mut rc = RECT::default();
            if cloaked == 0 && !IsIconic(hwnd).as_bool() && GetWindowRect(hwnd, &mut rc).is_ok() {
                let (w, h) = (rc.right - rc.left, rc.bottom - rc.top);
                if w > 0 && h > 0 {
                    if full.as_ref().map(|d| (d.w, d.h)) != Some((w, h)) {
                        if let Some(d) = full.take() {
                            free_dib(d);
                        }
                        full = make_dib(screen, w, h);
                        last_hash = None;
                    }
                    let scale = (PREVIEW_MAX_W as f32 / w as f32).min(1.0);
                    let (sw, sh) = (((w as f32 * scale) as i32).max(1), ((h as f32 * scale) as i32).max(1));
                    if small.as_ref().map(|d| (d.w, d.h)) != Some((sw, sh)) {
                        if let Some(d) = small.take() {
                            free_dib(d);
                        }
                        small = make_dib(screen, sw, sh);
                        last_hash = None;
                    }
                    if let (Some(f), Some(s)) = (&full, &small) {
                        if PrintWindow(hwnd, f.dc, PW_RENDERFULLCONTENT).as_bool() {
                            let _ = StretchBlt(s.dc, 0, 0, sw, sh, f.dc, 0, 0, w, h, SRCCOPY);
                            let _ = GdiFlush();
                            let raw = std::slice::from_raw_parts(s.bits as *const u8, (sw * sh * 4) as usize);
                            let hash = sample_hash(raw);
                            if last_hash != Some(hash) && !is_uniform_bgra(raw) {
                                last_hash = Some(hash);
                                if let Some(jpeg) = encode_bgra_jpeg(raw, sw as usize * 4, sw as u32, sh as u32) {
                                    let frame = CapturedFrame { id: next_frame_id, captured_t_us: agentpc_core::now_us(), jpeg };
                                    next_frame_id = next_frame_id.wrapping_add(1);
                                    let _ = tx.try_send(frame);
                                }
                            }
                        }
                    }
                }
            }
            let el = t0.elapsed();
            if el < GDI_FRAME_INTERVAL {
                std::thread::sleep(GDI_FRAME_INTERVAL - el);
            }
        }
        if let Some(d) = full.take() {
            free_dib(d);
        }
        if let Some(d) = small.take() {
            free_dib(d);
        }
        ReleaseDC(None, screen);
    }
}

/// Sparse FNV-1a over every 61st pixel — cheap change detection for the skip-unchanged gate.
fn sample_hash(raw: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for px in raw.chunks_exact(4).step_by(61) {
        for &b in &px[..3] {
            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// Sparse sample: does every checked pixel match the first one (ignoring alpha)?
fn is_uniform_bgra(raw: &[u8]) -> bool {
    if raw.len() < 4 {
        return true;
    }
    let (b0, g0, r0) = (raw[0], raw[1], raw[2]);
    raw.chunks_exact(4)
        .step_by(97)
        .all(|p| p[0] == b0 && p[1] == g0 && p[2] == r0)
}

extern "system" fn enum_proc(hwnd: HWND, lparam: LPARAM) -> BOOL {
    unsafe {
        if IsWindowVisible(hwnd).as_bool() {
            let len = GetWindowTextLengthW(hwnd);
            if len > 0 {
                let mut buf = vec![0u16; (len + 1) as usize];
                let n = GetWindowTextW(hwnd, &mut buf);
                if n > 0 {
                    let title = String::from_utf16_lossy(&buf[..n as usize]);
                    let v = &mut *(lparam.0 as *mut Vec<WinInfo>);
                    v.push(WinInfo {
                        hwnd: (hwnd.0 as isize).to_string(),
                        title,
                    });
                }
            }
        }
        BOOL(1)
    }
}

/// Enumerate visible, titled top-level windows for the box-binding picker.
pub fn list_windows() -> Vec<WinInfo> {
    let mut v: Vec<WinInfo> = Vec::new();
    unsafe {
        let _ = EnumWindows(Some(enum_proc), LPARAM(&mut v as *mut _ as isize));
    }
    v.sort_by(|a, b| a.title.to_lowercase().cmp(&b.title.to_lowercase()));
    v
}
