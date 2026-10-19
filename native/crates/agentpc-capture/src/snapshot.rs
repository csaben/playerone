//! On-demand, full-resolution snapshot of one app's windows composited as the user would see them —
//! main window, dialogs, menus and tooltips — even while they are covered by other windows.
//!
//! Each layer is rendered with `PrintWindow(PW_RENDERFULLCONTENT)` (DWM content, so D3D/flip-model
//! surfaces like OBS's preview come through) and copied, bottom to top, into a *stage*: a fixed
//! screen rectangle, normally the main window's visible frame. Pixels on the stage map 1:1 to
//! screen pixels at `stage.left + x, stage.top + y`, which is what lets a click at a screenshot
//! coordinate be routed back to the right window. Minimized layers are skipped (they don't render).

use std::ffi::c_void;

use windows::Win32::Foundation::{HWND, RECT};
use windows::Win32::Graphics::Gdi::{BitBlt, GdiFlush, GetDC, ReleaseDC, SRCCOPY};
use windows::Win32::Storage::Xps::{PrintWindow, PRINT_WINDOW_FLAGS};
use windows::Win32::UI::WindowsAndMessaging::{GetWindowRect, IsIconic};

use super::{free_dib, make_dib};

const PW_RENDERFULLCONTENT: PRINT_WINDOW_FLAGS = PRINT_WINDOW_FLAGS(2);

/// A screen rectangle, `right`/`bottom` exclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct ScreenRect {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

impl ScreenRect {
    pub fn width(&self) -> i32 {
        self.right - self.left
    }
    pub fn height(&self) -> i32 {
        self.bottom - self.top
    }
    pub fn contains(&self, x: i32, y: i32) -> bool {
        x >= self.left && x < self.right && y >= self.top && y < self.bottom
    }
    pub fn intersect(&self, o: &ScreenRect) -> Option<ScreenRect> {
        let r = ScreenRect {
            left: self.left.max(o.left),
            top: self.top.max(o.top),
            right: self.right.min(o.right),
            bottom: self.bottom.min(o.bottom),
        };
        (r.width() > 0 && r.height() > 0).then_some(r)
    }
}

/// One window to draw: its handle and the part of it that is actually visible on screen (the DWM
/// extended frame bounds — `GetWindowRect` also counts the invisible resize border).
#[derive(Debug, Clone, Copy)]
pub struct Layer {
    pub hwnd: isize,
    pub visible: ScreenRect,
}

/// A composited frame: tightly packed top-down BGRA, `width * 4` bytes per row.
pub struct Snapshot {
    pub width: u32,
    pub height: u32,
    pub bgra: Vec<u8>,
}

impl Snapshot {
    /// Encode as an opaque RGB PNG (fast compression — this is on the agent's critical path).
    pub fn to_png(&self) -> Result<Vec<u8>, String> {
        use image::codecs::png::{CompressionType, FilterType, PngEncoder};
        use image::ImageEncoder;
        let mut rgb = Vec::with_capacity((self.width * self.height * 3) as usize);
        for px in self.bgra.chunks_exact(4) {
            rgb.extend_from_slice(&[px[2], px[1], px[0]]);
        }
        let mut out = Vec::new();
        PngEncoder::new_with_quality(&mut out, CompressionType::Fast, FilterType::Sub)
            .write_image(&rgb, self.width, self.height, image::ExtendedColorType::Rgb8)
            .map_err(|e| format!("png encode failed: {e}"))?;
        Ok(out)
    }
}

/// Composite `layers` (ordered **bottom to top**) onto a black stage covering `stage`.
pub fn compose(stage: ScreenRect, layers: &[Layer]) -> Result<Snapshot, String> {
    let (sw, sh) = (stage.width(), stage.height());
    if sw <= 0 || sh <= 0 {
        return Err("stage is empty (window minimized or zero-sized?)".into());
    }
    unsafe {
        let screen = GetDC(None);
        if screen.is_invalid() {
            return Err("GetDC failed".into());
        }
        let result = (|| {
            let canvas = make_dib(screen, sw, sh).ok_or("stage allocation failed")?;
            for layer in layers {
                let h = HWND(layer.hwnd as *mut c_void);
                if IsIconic(h).as_bool() {
                    continue;
                }
                let Some(vis) = layer.visible.intersect(&stage) else { continue };
                let mut wr = RECT::default();
                if GetWindowRect(h, &mut wr).is_err() {
                    continue;
                }
                let (ww, wh) = (wr.right - wr.left, wr.bottom - wr.top);
                if ww <= 0 || wh <= 0 {
                    continue;
                }
                let Some(dib) = make_dib(screen, ww, wh) else { continue };
                if PrintWindow(h, dib.dc, PW_RENDERFULLCONTENT).as_bool() {
                    let _ = BitBlt(
                        canvas.dc,
                        vis.left - stage.left,
                        vis.top - stage.top,
                        vis.width(),
                        vis.height(),
                        dib.dc,
                        vis.left - wr.left,
                        vis.top - wr.top,
                        SRCCOPY,
                    );
                }
                free_dib(dib);
            }
            let _ = GdiFlush();
            let len = (sw * sh * 4) as usize;
            let mut bgra = std::slice::from_raw_parts(canvas.bits as *const u8, len).to_vec();
            free_dib(canvas);
            for px in bgra.chunks_exact_mut(4) {
                px[3] = 255;
            }
            Ok(Snapshot { width: sw as u32, height: sh as u32, bgra })
        })();
        ReleaseDC(None, screen);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_intersection_and_containment() {
        let a = ScreenRect { left: 0, top: 0, right: 100, bottom: 50 };
        let b = ScreenRect { left: 90, top: 40, right: 200, bottom: 200 };
        assert_eq!(a.intersect(&b), Some(ScreenRect { left: 90, top: 40, right: 100, bottom: 50 }));
        assert!(a.contains(0, 0) && !a.contains(100, 10));
        let far = ScreenRect { left: 500, top: 500, right: 600, bottom: 600 };
        assert_eq!(a.intersect(&far), None);
    }
}
