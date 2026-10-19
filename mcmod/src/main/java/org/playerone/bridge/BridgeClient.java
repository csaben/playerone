package org.playerone.bridge;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.loader.api.FabricLoader;

public final class BridgeClient implements ClientModInitializer {
	public static final int DEFAULT_PORT = 8970;

	@Override
	public void onInitializeClient() {
		int port = Integer.getInteger("playerone.bridge.port", DEFAULT_PORT);
		String mcVersion = version("minecraft");
		String bridgeVersion = version("playerone-bridge");
		BridgeLog.info("starting bridge " + bridgeVersion + " for Minecraft " + mcVersion);
		new BridgeServer(port, InputApplier.STATE, () -> InputApplier.pong(mcVersion, bridgeVersion)).start();
	}

	private static String version(String modId) {
		return FabricLoader.getInstance().getModContainer(modId)
			.map(c -> c.getMetadata().getVersion().getFriendlyString())
			.orElse("unknown");
	}
}
