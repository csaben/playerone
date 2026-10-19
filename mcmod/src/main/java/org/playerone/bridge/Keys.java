package org.playerone.bridge;

import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Physical-key tokens -> input ids. 26.x uses SDL3: keyboard keys are SDL scancodes (physical positions, layout
 * independent), mouse buttons are SDL buttons (left=1, middle=2, right=3). Values mirror InputConstants.KEY_* /
 * MOUSE_BUTTON_*; kept as plain ints so the network thread never touches game classes.
 * Ids >= MOUSE_BASE are mouse buttons (id - MOUSE_BASE = SDL button).
 */
public final class Keys {
	public static final int MOUSE_BASE = 100_000;
	public static final int MOUSE_LEFT = MOUSE_BASE + 1; // InputConstants.MOUSE_BUTTON_LEFT
	public static final int MOUSE_RIGHT = MOUSE_BASE + 3; // InputConstants.MOUSE_BUTTON_RIGHT

	private static final Map<String, Integer> TOKENS = new HashMap<>();

	static {
		for (char c = 'A'; c <= 'Z'; c++) {
			TOKENS.put(String.valueOf(c), 4 + (c - 'A')); // KEY_A = 4 .. KEY_Z = 29
		}
		for (int d = 1; d <= 9; d++) {
			TOKENS.put(String.valueOf(d), 30 + d - 1); // KEY_1 = 30 .. KEY_9 = 38
		}
		TOKENS.put("0", 39); // KEY_0
		TOKENS.put("ENTER", 40); // KEY_RETURN
		TOKENS.put("ESC", 41); // KEY_ESCAPE
		TOKENS.put("BACKSPACE", 42);
		TOKENS.put("TAB", 43);
		TOKENS.put("SPACE", 44);
		TOKENS.put("RIGHT", 79);
		TOKENS.put("LEFT", 80);
		TOKENS.put("DOWN", 81);
		TOKENS.put("UP", 82);
		TOKENS.put("LCTRL", 224); // KEY_LCONTROL
		TOKENS.put("LSHIFT", 225);
		TOKENS.put("LALT", 226);
		TOKENS.put("RCTRL", 228); // KEY_RCONTROL
		TOKENS.put("RSHIFT", 229);
		TOKENS.put("RALT", 230);
	}

	public static final int ESC = 41;

	private Keys() {
	}

	/** @return the input id, or -1 for an unknown token. */
	public static int token(String token) {
		Integer v = TOKENS.get(token.toUpperCase(Locale.ROOT));
		return v == null ? -1 : v;
	}

	public static boolean isMouse(int id) {
		return id >= MOUSE_BASE;
	}
}
