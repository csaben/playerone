package org.playerone.bridge;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Shared state between connection threads and the client thread. All access is synchronized on the instance.
 * Only the controller (the connection that most recently sent an act) can change held state.
 */
public final class BridgeState {
	public static final long DEAD_MAN_NANOS = 1_000_000_000L;
	public static final long CAM_SPREAD_NANOS = 50_000_000L;

	private Object controller;
	private long controllerLastMsg;
	private Set<Integer> held = new HashSet<>();
	private final List<Integer> pendingPresses = new ArrayList<>();
	private boolean releaseRequested;
	private boolean becameActive;

	private double camX;
	private double camY;
	private long camDeadline;
	private long lastFrame;

	/** Snapshot handed to the client thread each tick. */
	public record Drain(boolean active, Set<Integer> held, List<Integer> presses, boolean release, boolean becameActive) {
	}

	public synchronized void onAct(Object conn, Set<Integer> newHeld, double dx, double dy) {
		long now = System.nanoTime();
		if (controller != conn) {
			if (controller == null) {
				becameActive = true;
				BridgeLog.info("controller active");
			} else {
				BridgeLog.info("controller switched to newest act sender");
			}
			controller = conn;
		}
		controllerLastMsg = now;
		for (Integer id : newHeld) {
			if (!held.contains(id)) {
				pendingPresses.add(id);
			}
		}
		held = new HashSet<>(newHeld);
		if (dx != 0 || dy != 0) {
			camX += dx;
			camY += dy;
			camDeadline = now + CAM_SPREAD_NANOS;
		}
	}

	/** Any message (act, ping, release) from the controller feeds the dead-man switch. */
	public synchronized void onMessage(Object conn) {
		if (conn == controller) {
			controllerLastMsg = System.nanoTime();
		}
	}

	public synchronized void onRelease(Object conn) {
		if (conn == controller) {
			releaseLocked("release requested");
		}
	}

	public synchronized void onDisconnect(Object conn) {
		if (conn == controller) {
			releaseLocked("controller disconnected");
		}
	}

	public synchronized void checkDeadMan() {
		if (controller != null && System.nanoTime() - controllerLastMsg > DEAD_MAN_NANOS) {
			releaseLocked("dead-man: no message from controller for 1 s");
		}
	}

	private void releaseLocked(String why) {
		BridgeLog.info("release: " + why);
		controller = null;
		held = new HashSet<>();
		pendingPresses.clear();
		camX = camY = 0;
		releaseRequested = true;
		becameActive = false;
	}

	public synchronized boolean isActive() {
		return controller != null;
	}

	public synchronized boolean isHeld(int id) {
		return controller != null && held.contains(id);
	}

	public synchronized Drain drain() {
		Drain d = new Drain(controller != null, new HashSet<>(held), new ArrayList<>(pendingPresses), releaseRequested, becameActive);
		pendingPresses.clear();
		releaseRequested = false;
		becameActive = false;
		return d;
	}

	/**
	 * Takes the share of pending camera counts due this frame: counts are spread linearly from the last frame to the
	 * deadline (50 ms after the newest act), so 20 Hz acts turn into smooth per-frame rotation.
	 * @return {dx, dy} raw counts to apply now.
	 */
	public synchronized double[] takeCam(boolean discard) {
		long now = System.nanoTime();
		long prev = lastFrame;
		lastFrame = now;
		if (discard || (camX == 0 && camY == 0)) {
			camX = camY = 0;
			return null;
		}
		double frac;
		if (now >= camDeadline || camDeadline <= prev) {
			frac = 1.0;
		} else {
			frac = Math.min(1.0, (double) (now - prev) / (double) (camDeadline - prev));
		}
		double x = camX * frac;
		double y = camY * frac;
		camX -= x;
		camY -= y;
		return new double[] {x, y};
	}
}
