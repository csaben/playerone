//! Shared input-token vocabulary.
//!
//! Every sink speaks the same level-state token strings — letters/digits, `SPACE`, the arrows,
//! `LSHIFT`/`LCTRL`/… — established by the forward maps `agentpc_inject::sink::vk_for` (token → VK)
//! and `agentpc_client::cdp::key_def` (token → CDP key). The Windows human RawInput tap needs the
//! **reverse** direction (a VKey off `RAWKEYBOARD` → a token it can put in `Action::keys_held`), and
//! keeping that reverse map here — beside nothing, as the single canonical VK↔token authority —
//! means the tap can never disagree with what the sinks will later re-inject when replaying a shard.

const LETTERS: [&str; 26] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y", "Z",
];
const DIGITS: [&str; 10] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

/// Map a Windows virtual-key code to the sink token vocabulary, or `None` for keys agentpc doesn't
/// drive (which the tap then skips — never recording a key no sink could reproduce). This is the
/// exact reverse of the tokens `agentpc_inject::sink::vk_for` accepts. The generic modifier VKs
/// (`VK_SHIFT`/`VK_CONTROL`/`VK_MENU`) that raw input often reports instead of the L/R variants map
/// to the left token, which every sink accepts.
pub fn token_for_vk(vk: u32) -> Option<&'static str> {
    // A–Z: VK 0x41..=0x5A equal ASCII uppercase.
    if (0x41..=0x5A).contains(&vk) {
        return Some(LETTERS[(vk - 0x41) as usize]);
    }
    // 0–9: VK 0x30..=0x39.
    if (0x30..=0x39).contains(&vk) {
        return Some(DIGITS[(vk - 0x30) as usize]);
    }
    Some(match vk {
        0x20 => "SPACE",
        0x0D => "ENTER",
        0x1B => "ESC",
        0x09 => "TAB",
        0x08 => "BACKSPACE",
        0x25 => "LEFT",
        0x26 => "UP",
        0x27 => "RIGHT",
        0x28 => "DOWN",
        0xA0 | 0x10 => "LSHIFT", // VK_LSHIFT, or generic VK_SHIFT
        0xA1 => "RSHIFT",
        0xA2 | 0x11 => "LCTRL", // VK_LCONTROL, or generic VK_CONTROL
        0xA3 => "RCTRL",
        0xA4 | 0x12 => "LALT", // VK_LMENU, or generic VK_MENU
        0xA5 => "RALT",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letters_and_digits_round_trip_to_tokens() {
        assert_eq!(token_for_vk(0x57), Some("W"));
        assert_eq!(token_for_vk(0x41), Some("A"));
        assert_eq!(token_for_vk(0x30), Some("0"));
        assert_eq!(token_for_vk(0x39), Some("9"));
    }

    #[test]
    fn named_keys_and_modifiers_match_the_sink_vocab() {
        assert_eq!(token_for_vk(0x20), Some("SPACE"));
        assert_eq!(token_for_vk(0x26), Some("UP"));
        // Both the specific and the generic modifier VK land on a token every sink accepts.
        assert_eq!(token_for_vk(0xA0), Some("LSHIFT"));
        assert_eq!(token_for_vk(0x10), Some("LSHIFT"));
        assert_eq!(token_for_vk(0x11), Some("LCTRL"));
    }

    #[test]
    fn undrivable_keys_are_skipped() {
        assert_eq!(token_for_vk(0x7C), None); // F13
        assert_eq!(token_for_vk(0xFF), None); // raw-input "fake" key
        assert_eq!(token_for_vk(0), None);
    }
}
