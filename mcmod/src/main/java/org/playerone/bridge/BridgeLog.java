package org.playerone.bridge;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class BridgeLog {
	private static final Logger LOG = LoggerFactory.getLogger("playerone-bridge");

	private BridgeLog() {
	}

	public static void info(String msg) {
		LOG.info("[playerone-bridge] {}", msg);
	}

	public static void warn(String msg) {
		LOG.warn("[playerone-bridge] {}", msg);
	}

	public static void error(String msg, Throwable t) {
		LOG.error("[playerone-bridge] " + msg, t);
	}
}
