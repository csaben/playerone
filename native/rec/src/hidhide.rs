//! HidHide (Nefarius): hides your real controller from every app but this one, so the passthrough
//! pad (pad.rs) is the only controller games see. With the passthrough reporting the real pad's
//! vendor/product id, an app bound to your controller takes the passthrough without being set up.
//!
//! Driven through HidHideCLI, which needs no elevation. Only XInput pads are hidden: the
//! passthrough reads real pads through XInput, so hiding any other controller would lose it.

use std::path::PathBuf;
use std::os::windows::process::CommandExt;
use std::process::Command;

use serde_json::Value;
use windows::core::PCWSTR;
use windows::Win32::Devices::DeviceAndDriverInstallation::{
    CM_Get_DevNode_Registry_PropertyW, CM_Get_Device_ID_ListW, CM_Get_Device_ID_List_SizeW, CM_Get_Parent, CM_Locate_DevNodeW,
    CM_DRP_SERVICE, CM_GETIDLIST_FILTER_ENUMERATOR, CM_GETIDLIST_FILTER_PRESENT, CM_LOCATE_DEVNODE_NORMAL,
    CR_SUCCESS,
};

/// A real (not virtual) XInput controller's HID interface.
pub struct RealPad {
    /// device instance path, what HidHide hides
    pub instance: String,
    pub vendor: u16,
    pub product: u16,
}

fn cli() -> Option<PathBuf> {
    let base = std::env::var_os("ProgramFiles").map(PathBuf::from).unwrap_or_else(|| "C:\\Program Files".into());
    let p = base.join("Nefarius Software Solutions\\HidHide\\x64\\HidHideCLI.exe");
    p.exists().then_some(p)
}

pub fn installed() -> bool {
    cli().is_some()
}

fn run(args: &[&str]) -> Result<String, String> {
    let exe = cli().ok_or("HidHide isn't installed")?;
    // CREATE_NO_WINDOW: no console flashing up
    let out = Command::new(exe).args(args).creation_flags(0x0800_0000).output().map_err(|e| format!("HidHideCLI: {e}"))?;
    if !out.status.success() {
        return Err(format!("HidHideCLI {args:?}: {}", String::from_utf8_lossy(&out.stderr).trim()));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// The quoted paths a listing (`--app-list`, `--dev-list`) prints, one `--x "path"` per line.
fn listed(out: &str) -> Vec<String> {
    out.lines().filter_map(|l| Some(l.split_once('"')?.1.trim_end().trim_end_matches('"').to_string())).collect()
}

/// The present XInput controllers that aren't ViGEm's (virtual pads sit on the ViGEm bus).
pub fn real_pads() -> Vec<RealPad> {
    let Ok(out) = run(&["--dev-gaming"]) else { return Vec::new() };
    let Ok(groups) = serde_json::from_str::<Value>(&out) else { return Vec::new() };
    let mut pads = Vec::new();
    for d in groups.as_array().into_iter().flatten().flat_map(|g| g["devices"].as_array().into_iter().flatten()) {
        let (Some(instance), Some(container)) = (d["deviceInstancePath"].as_str(), d["baseContainerDeviceInstancePath"].as_str()) else { continue };
        let upper = instance.to_ascii_uppercase();
        if !d["present"].as_bool().unwrap_or(false) || !upper.contains("&IG_") || on_vigem(container) {
            continue;
        }
        if let Some((vendor, product)) = ids(&upper) {
            pads.push(RealPad { instance: instance.to_string(), vendor, product });
        }
    }
    pads
}

/// Vendor and product id from a HID instance path, over USB (`VID_045E&PID_02FF&IG_00`), Bluetooth
/// (`{00001124-…}_VID&0002045E_PID&02FD&IG_00`) or Bluetooth LE (`{00001812-…}_DEV_VID&02045E_PID&0B13&REV…`):
/// the hex after VID/PID, of which the id is the last four digits (Bluetooth prefixes the vendor
/// with where the id comes from).
fn ids(upper: &str) -> Option<(u16, u16)> {
    let field = |key: &str| {
        let rest = upper.split(key).nth(1)?.strip_prefix(['_', '&'])?;
        let hex: String = rest.chars().take_while(|c| c.is_ascii_hexdigit()).collect();
        u16::from_str_radix(hex.get(hex.len().checked_sub(4)?..)?, 16).ok()
    };
    Some((field("VID")?, field("PID")?))
}

/// The id the passthrough takes on for this pad. A Microsoft pad over Bluetooth reports a
/// model-specific product id, but over USB every Xbox One/Series pad is 045E:02FF to apps: the
/// passthrough is always that, so switching between cable and Bluetooth changes nothing an app
/// bound to the pad can see.
pub fn identity(p: &RealPad) -> (u16, u16) {
    let bluetooth = p.instance.starts_with("HID\\{");
    if bluetooth && p.vendor == 0x045E {
        (0x045E, 0x02FF)
    } else {
        (p.vendor, p.product)
    }
}

fn on_vigem(instance: &str) -> bool {
    under_vigem(instance, 1)
}

/// Whether the device sits on the ViGEm bus within `levels` parents.
fn under_vigem(instance: &str, levels: u32) -> bool {
    let id: Vec<u16> = instance.encode_utf16().chain([0]).collect();
    unsafe {
        let mut dev = 0u32;
        if CM_Locate_DevNodeW(&mut dev, PCWSTR(id.as_ptr()), CM_LOCATE_DEVNODE_NORMAL) != CR_SUCCESS {
            return false;
        }
        for _ in 0..levels {
            let mut parent = 0u32;
            if CM_Get_Parent(&mut parent, dev, 0) != CR_SUCCESS {
                return false;
            }
            let mut buf = [0u16; 64];
            let mut len = (buf.len() * 2) as u32;
            if CM_Get_DevNode_Registry_PropertyW(parent, CM_DRP_SERVICE, None, Some(buf.as_mut_ptr().cast()), &mut len, 0) == CR_SUCCESS
                && String::from_utf16_lossy(&buf[..(len as usize / 2)]).trim_end_matches('\0').eq_ignore_ascii_case("ViGEmBus")
            {
                return true;
            }
            dev = parent;
        }
        false
    }
}

/// The HID interfaces (IG_xx) of present ViGEm pads with this vendor/product id.
///
/// A virtual pad's HID interface number is handed out when it's plugged in. Interface 00 under a
/// real Xbox One pad's id (045E:02FF&IG_00) is exactly what Windows' xinputhid driver claims, and
/// on a virtual pad that driver fails: XInput still works but the HID device apps like Eden read
/// (SDL raw input) is broken.
pub fn vigem_interfaces(vendor: u16, product: u16) -> Vec<String> {
    let prefix = format!("HID\\VID_{vendor:04X}&PID_{product:04X}&IG_");
    let filter: Vec<u16> = "HID".encode_utf16().chain([0]).collect();
    let flags = CM_GETIDLIST_FILTER_ENUMERATOR | CM_GETIDLIST_FILTER_PRESENT;
    unsafe {
        let mut len = 0u32;
        if CM_Get_Device_ID_List_SizeW(&mut len, PCWSTR(filter.as_ptr()), flags) != CR_SUCCESS {
            return Vec::new();
        }
        let mut buf = vec![0u16; len as usize];
        if CM_Get_Device_ID_ListW(PCWSTR(filter.as_ptr()), &mut buf, flags) != CR_SUCCESS {
            return Vec::new();
        }
        String::from_utf16_lossy(&buf)
            .split('\0')
            .filter(|id| id.to_ascii_uppercase().starts_with(&prefix) && under_vigem(id, 4))
            .map(|id| id.to_ascii_uppercase()[prefix.len()..].chars().take(2).collect())
            .collect()
    }
}

/// Lets this process see hidden devices, hides these pads, and turns hiding on.
pub fn hide(pads: &[RealPad]) -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?.to_string_lossy().into_owned();
    let apps = listed(&run(&["--app-list"])?);
    let devs = listed(&run(&["--dev-list"])?);
    let mut args: Vec<&str> = Vec::new();
    if !apps.iter().any(|a| a.eq_ignore_ascii_case(&exe)) {
        args.extend(["--app-reg", &exe]);
    }
    for p in pads {
        if !devs.iter().any(|d| d.eq_ignore_ascii_case(&p.instance)) {
            args.extend(["--dev-hide", &p.instance]);
        }
    }
    args.push("--cloak-on");
    run(&args).map(|_| ())
}

/// Shows hidden devices to every app again (the hidden list is kept for next time).
pub fn unhide() {
    if installed() {
        let _ = run(&["--cloak-off"]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn listing_paths() {
        let out = "--app-reg \"C:\\a b\\x.exe\"\n--app-reg \"D:\\y.exe\"\n";
        assert_eq!(listed(out), vec!["C:\\a b\\x.exe", "D:\\y.exe"]);
    }

    #[test]
    fn ids_over_usb_and_bluetooth() {
        assert_eq!(ids("HID\\VID_045E&PID_02FF&IG_00\\8&2C475C17&3&0000"), Some((0x045E, 0x02FF)));
        assert_eq!(ids("HID\\{00001124-0000-1000-8000-00805F9B34FB}_VID&0002045E_PID&02FD&IG_00\\9&1&0000"), Some((0x045E, 0x02FD)));
        assert_eq!(ids("HID\\{00001812-0000-1000-8000-00805F9B34FB}_DEV_VID&02045E_PID&0B13_REV&0509&IG_00\\A&1&0000"), Some((0x045E, 0x0B13)));
        let bt = RealPad { instance: "HID\\{00001124-0000-1000-8000-00805F9B34FB}_VID&0002045E_PID&02FD&IG_00\\9".into(), vendor: 0x045E, product: 0x02FD };
        assert_eq!(identity(&bt), (0x045E, 0x02FF));
    }

    /// The virtual pads' HID interfaces under the Xbox One id: `cargo test -p player-one-rec vigem_ifaces -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn vigem_ifaces() {
        println!("interfaces {:?}", vigem_interfaces(0x045E, 0x02FF));
    }

    /// Lists the real XInput pads (needs HidHide): `cargo test -p player-one-rec real_pads -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn real_pads_skips_virtual() {
        for p in real_pads() {
            println!("{} {:04x}:{:04x}", p.instance, p.vendor, p.product);
        }
    }
}
