package org.playerone.bridge;

import com.google.gson.JsonNull;
import com.google.gson.JsonObject;
import com.mojang.blaze3d.platform.InputConstants;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import net.minecraft.client.KeyMapping;
import net.minecraft.client.Minecraft;
import net.minecraft.client.ToggleKeyMapping;
import net.minecraft.client.gui.screens.PauseScreen;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.client.gui.screens.inventory.AbstractContainerScreen;
import net.minecraft.client.player.LocalPlayer;
import org.playerone.bridge.mixin.KeyMappingAccessor;
import org.playerone.bridge.mixin.ToggleKeyMappingAccessor;

/** Client-thread side: turns the bridge state into KeyMapping state and player rotation. */
public final class InputApplier {
	public static final BridgeState STATE = new BridgeState();

	// client thread only
	private static Set<Integer> appliedHeld = new HashSet<>();
	private static Set<Integer> tapRelease = new HashSet<>();

	// published by the client thread for pong (read on connection threads)
	private static volatile boolean focused;
	private static volatile boolean paused;
	private static volatile String screen;

	private InputApplier() {
	}

	public static boolean isDriving() {
		return STATE.isActive();
	}

	/** True while the bridge controls the game and the attack mapping is down. */
	public static boolean isHoldingAttack(Minecraft mc) {
		return STATE.isActive() && mc.options.keyAttack.isDown();
	}

	public static JsonObject pong(String mcVersion, String bridgeVersion) {
		JsonObject o = new JsonObject();
		o.addProperty("pong", 1);
		o.addProperty("pid", ProcessHandle.current().pid());
		o.addProperty("mc", mcVersion);
		o.addProperty("focused", focused);
		o.addProperty("paused", paused);
		String s = screen;
		if (s == null) {
			o.add("screen", JsonNull.INSTANCE);
		} else {
			o.addProperty("screen", s);
		}
		o.addProperty("bridge", bridgeVersion);
		return o;
	}

	// ---------------------------------------------------------------- tick

	public static void onClientTick(Minecraft mc) {
		STATE.checkDeadMan();
		BridgeState.Drain d = STATE.drain();

		if (d.release()) {
			Set<Integer> all = new HashSet<>(appliedHeld);
			all.addAll(tapRelease);
			for (int id : all) {
				for (KeyMapping km : bound(mc, id)) {
					if (km instanceof ToggleKeyMapping) {
						((ToggleKeyMappingAccessor) km).playerone$reset();
					} else {
						km.setDown(false);
					}
				}
			}
			appliedHeld = new HashSet<>();
			tapRelease = new HashSet<>();
		}
		if (!d.active()) {
			return;
		}
		// keep the unfocused/AFK frame limiter from throttling the game while the agent plays
		mc.getFramerateLimitTracker().onInputReceived();

		Screen current = mc.gui.screen();
		if (d.becameActive() && current instanceof PauseScreen && !mc.getWindow().isFocused()) {
			// The game paused on focus loss before the agent took control: resume. The window is unfocused, so
			// MouseHandler.grabMouse() is a no-op and the real cursor is left alone.
			mc.gui.setScreen(null);
			current = mc.gui.screen();
		}

		Set<Integer> held = d.held();

		// taps from last tick (press and release arrived between ticks) go up now
		for (int id : tapRelease) {
			if (!held.contains(id)) {
				forEachHoldMapping(mc, id, false);
			}
		}
		tapRelease = new HashSet<>();

		if (current != null || mc.player == null) {
			handleScreenPresses(mc, current, d.presses());
			// vanilla releases every mapping when a screen opens; held keys are re-applied after it closes
			for (int id : appliedHeld) {
				if (!held.contains(id)) {
					forEachHoldMapping(mc, id, false);
				}
			}
			appliedHeld = held;
			return;
		}

		// press edges: click count (edge-triggered binds fire once); toggle-mode mappings flip once
		for (int id : d.presses()) {
			if (id == Keys.ESC) {
				mc.pauseGame(false); // vanilla: ESC in game opens the pause menu
				continue;
			}
			for (KeyMapping km : bound(mc, id)) {
				KeyMappingAccessor acc = (KeyMappingAccessor) km;
				acc.playerone$setClickCount(acc.playerone$getClickCount() + 1);
				if (isToggleMode(km)) {
					km.setDown(true);
				} else if (!held.contains(id)) {
					km.setDown(true); // pressed and released before this tick: hold it for one tick
					tapRelease.add(id);
				}
			}
		}
		// held state, re-applied every tick so vanilla focus/screen handling cannot drop it
		for (int id : held) {
			forEachHoldMapping(mc, id, true);
		}
		for (int id : appliedHeld) {
			if (!held.contains(id) && !tapRelease.contains(id)) {
				forEachHoldMapping(mc, id, false);
			}
		}
		appliedHeld = held;
	}

	private static void handleScreenPresses(Minecraft mc, Screen current, List<Integer> presses) {
		if (current == null) {
			return;
		}
		for (int id : presses) {
			if (id == Keys.ESC) {
				if (current.shouldCloseOnEsc()) {
					current.onClose();
					return;
				}
			} else if (current instanceof AbstractContainerScreen<?> && !Keys.isMouse(id)
				&& mc.options.keyInventory.matches(key(id))) {
				current.onClose(); // inventory key closes container screens, as in vanilla keyPressed
				return;
			}
		}
	}

	/** Sets held state on mappings in hold mode; toggle-mode mappings only change on press edges. */
	private static void forEachHoldMapping(Minecraft mc, int id, boolean down) {
		for (KeyMapping km : bound(mc, id)) {
			if (!isToggleMode(km) && km.isDown() != down) {
				km.setDown(down);
			}
		}
	}

	private static boolean isToggleMode(KeyMapping km) {
		return km instanceof ToggleKeyMapping && ((ToggleKeyMappingAccessor) km).playerone$needsToggle().getAsBoolean();
	}

	private static InputConstants.Key key(int id) {
		return Keys.isMouse(id)
			? InputConstants.Type.MOUSE.getOrCreate(id - Keys.MOUSE_BASE)
			: InputConstants.Type.KEYBOARD.getOrCreate(id);
	}

	/** Every KeyMapping the user currently has bound to this physical key or button. */
	private static List<KeyMapping> bound(Minecraft mc, int id) {
		InputConstants.Key key = key(id);
		List<KeyMapping> out = new ArrayList<>(2);
		for (KeyMapping km : mc.options.keyMappings) {
			if (km.matches(key)) {
				out.add(km);
			}
		}
		return out;
	}

	// ---------------------------------------------------------------- frame

	public static void onFrame(Minecraft mc) {
		focused = mc.getWindow().isFocused();
		paused = mc.isPaused();
		Screen s = mc.gui.screen();
		screen = s == null ? null : s.getClass().getName();

		LocalPlayer player = mc.player;
		boolean discard = s != null || player == null || mc.isPaused();
		double[] cam = STATE.takeCam(discard);
		if (cam == null || player == null) {
			return;
		}
		// Same scaling as MouseHandler.turnPlayer (non-smooth, non-scoping); Entity.turn applies the 0.15 factor.
		double f = mc.options.sensitivity().get() * 0.6 + 0.2;
		double e = f * f * f * 8.0;
		double xo = cam[0] * e;
		double yo = cam[1] * e;
		player.turn(mc.options.invertMouseX().get() ? -xo : xo, mc.options.invertMouseY().get() ? -yo : yo);
	}
}
