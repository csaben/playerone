package org.playerone.bridge;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * 127.0.0.1 TCP server, newline-delimited JSON. Any number of concurrent connections (the dock's bare connect/close
 * probes are harmless); the connection that most recently sent an act is the controller.
 */
public final class BridgeServer {
	public interface Status {
		JsonObject pong();
	}

	private static final int MAX_LINE = 64 * 1024;
	private static final AtomicInteger IDS = new AtomicInteger();

	private final int port;
	private final BridgeState state;
	private final Status status;

	public BridgeServer(int port, BridgeState state, Status status) {
		this.port = port;
		this.state = state;
		this.status = status;
	}

	/**
	 * Binds synchronously and starts the accept thread.
	 * @return the bound port, or -1 if binding failed (logged; the game keeps running without the bridge).
	 */
	public int start() {
		ServerSocket server;
		try {
			server = new ServerSocket();
			server.bind(new InetSocketAddress(InetAddress.getLoopbackAddress(), port));
		} catch (IOException e) {
			BridgeLog.error("cannot listen on 127.0.0.1:" + port + " (port in use?)", e);
			return -1;
		}
		BridgeLog.info("listening on 127.0.0.1:" + server.getLocalPort());
		Thread t = new Thread(() -> acceptLoop(server), "playerone-bridge-accept");
		t.setDaemon(true);
		t.start();
		return server.getLocalPort();
	}

	private void acceptLoop(ServerSocket server) {
		try (server) {
			while (true) {
				Socket s = server.accept();
				new Conn(s, IDS.incrementAndGet()).start();
			}
		} catch (IOException e) {
			BridgeLog.error("server stopped", e);
		}
	}

	private final class Conn {
		private final Socket socket;
		private final int id;
		/** Replies are queued and written by a separate thread; when the queue is full they are dropped, never blocking. */
		private final BlockingQueue<String> out = new ArrayBlockingQueue<>(64);
		private volatile boolean closed;
		private boolean spoke;

		Conn(Socket socket, int id) {
			this.socket = socket;
			this.id = id;
		}

		void start() {
			Thread r = new Thread(this::readLoop, "playerone-bridge-conn-" + id);
			r.setDaemon(true);
			r.start();
			Thread w = new Thread(this::writeLoop, "playerone-bridge-write-" + id);
			w.setDaemon(true);
			w.start();
		}

		private void send(String line) {
			out.offer(line); // drop on overflow rather than stall
		}

		private void writeLoop() {
			try {
				OutputStream os = socket.getOutputStream();
				while (!closed) {
					String line = out.poll(200, TimeUnit.MILLISECONDS);
					if (line == null) {
						continue;
					}
					os.write((line + "\n").getBytes(StandardCharsets.UTF_8));
					os.flush();
				}
			} catch (IOException | InterruptedException ignored) {
				// reader side handles disconnect
			}
		}

		private void readLoop() {
			try (socket; BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream(), StandardCharsets.UTF_8))) {
				socket.setTcpNoDelay(true);
				StringBuilder sb = new StringBuilder();
				int c;
				while ((c = in.read()) != -1) {
					if (c == '\n') {
						String line = sb.toString().trim();
						sb.setLength(0);
						if (!line.isEmpty()) {
							handle(line);
						}
					} else if (sb.length() < MAX_LINE) {
						sb.append((char) c);
					}
				}
			} catch (IOException e) {
				// connection reset etc.
			} finally {
				closed = true;
				state.onDisconnect(this);
				if (spoke) {
					BridgeLog.info("client #" + id + " disconnected");
				}
			}
		}

		private void handle(String line) {
			if (!spoke) {
				spoke = true;
				BridgeLog.info("client #" + id + " connected from " + socket.getRemoteSocketAddress());
			}
			JsonObject msg;
			try {
				msg = JsonParser.parseString(line).getAsJsonObject();
			} catch (RuntimeException e) {
				send(error("bad json"));
				return;
			}
			state.onMessage(this);
			try {
				if (msg.has("ping")) {
					send(status.pong().toString());
				} else if (msg.has("release")) {
					state.onRelease(this);
					send("{\"ok\":true}");
				} else if (msg.has("act")) {
					String err = act(msg.getAsJsonObject("act"));
					send(err == null ? "{\"ok\":true}" : error(err));
				} else {
					send(error("unknown message"));
				}
			} catch (RuntimeException e) {
				BridgeLog.warn("client #" + id + " bad message: " + e);
				send(error("bad message: " + e.getMessage()));
			}
		}

		/** Applies whatever is valid; returns an error string for unknown tokens (known ones still apply). */
		private String act(JsonObject act) {
			Set<Integer> held = new HashSet<>();
			String err = null;
			if (act.has("keys") && act.get("keys").isJsonArray()) {
				for (JsonElement k : act.getAsJsonArray("keys")) {
					int v = Keys.token(k.getAsString());
					if (v < 0) {
						err = "unknown key: " + k.getAsString();
					} else {
						held.add(v);
					}
				}
			}
			if (act.has("mb") && act.get("mb").isJsonArray()) {
				JsonArray mb = act.getAsJsonArray("mb");
				if (mb.size() > 0 && truthy(mb.get(0))) {
					held.add(Keys.MOUSE_LEFT);
				}
				if (mb.size() > 1 && truthy(mb.get(1))) {
					held.add(Keys.MOUSE_RIGHT);
				}
			}
			double dx = 0;
			double dy = 0;
			if (act.has("cam") && act.get("cam").isJsonArray()) {
				JsonArray cam = act.getAsJsonArray("cam");
				if (cam.size() >= 2) {
					dx = cam.get(0).getAsDouble();
					dy = cam.get(1).getAsDouble();
					if (!Double.isFinite(dx) || !Double.isFinite(dy)) {
						dx = dy = 0;
						err = "cam not finite";
					}
				}
			}
			state.onAct(this, held, dx, dy);
			return err;
		}

		private boolean truthy(JsonElement e) {
			if (e.isJsonPrimitive() && e.getAsJsonPrimitive().isBoolean()) {
				return e.getAsBoolean();
			}
			return e.isJsonPrimitive() && e.getAsJsonPrimitive().isNumber() && e.getAsDouble() != 0;
		}

		private String error(String why) {
			JsonObject o = new JsonObject();
			o.addProperty("error", why);
			return o.toString();
		}
	}
}
