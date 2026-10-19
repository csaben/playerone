package org.playerone.bridge;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.gson.JsonNull;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Set;
import java.util.function.BooleanSupplier;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/** Exercises the network/controller logic without the game (the client-thread side is not covered here). */
class BridgeServerTest {
	private BridgeState state;
	private int port;

	@BeforeEach
	void setUp() {
		state = new BridgeState();
		port = new BridgeServer(0, state, () -> {
			JsonObject o = new JsonObject();
			o.addProperty("pong", 1);
			o.addProperty("pid", ProcessHandle.current().pid());
			o.addProperty("mc", "26.3");
			o.addProperty("focused", false);
			o.addProperty("paused", false);
			o.add("screen", JsonNull.INSTANCE);
			o.addProperty("bridge", "test");
			return o;
		}).start();
		assertTrue(port > 0);
	}

	private static final class Client implements AutoCloseable {
		final Socket s;
		final OutputStream out;
		final BufferedReader in;

		Client(int port) throws Exception {
			s = new Socket("127.0.0.1", port);
			s.setSoTimeout(3000);
			out = s.getOutputStream();
			in = new BufferedReader(new InputStreamReader(s.getInputStream(), StandardCharsets.UTF_8));
		}

		void send(String json) throws Exception {
			out.write((json + "\n").getBytes(StandardCharsets.UTF_8));
			out.flush();
		}

		JsonObject read() throws Exception {
			return JsonParser.parseString(in.readLine()).getAsJsonObject();
		}

		JsonObject call(String json) throws Exception {
			send(json);
			return read();
		}

		@Override
		public void close() throws Exception {
			s.close();
		}
	}

	private static void await(BooleanSupplier cond) throws InterruptedException {
		long end = System.currentTimeMillis() + 3000;
		while (!cond.getAsBoolean()) {
			if (System.currentTimeMillis() > end) {
				throw new AssertionError("timed out");
			}
			Thread.sleep(5);
		}
	}

	@Test
	void pongHasPid() throws Exception {
		try (Client c = new Client(port)) {
			JsonObject p = c.call("{\"ping\":1}");
			assertEquals(1, p.get("pong").getAsInt());
			assertEquals(ProcessHandle.current().pid(), p.get("pid").getAsLong());
			assertTrue(p.get("screen").isJsonNull());
			assertFalse(state.isActive(), "ping alone must not take control");
		}
	}

	@Test
	void actHoldsKeysAndRecordsPressEdges() throws Exception {
		try (Client c = new Client(port)) {
			assertEquals(true, c.call("{\"act\":{\"keys\":[\"W\",\"SPACE\",\"1\"],\"mb\":[1,0],\"cam\":[0,0]}}").get("ok").getAsBoolean());
			BridgeState.Drain d = state.drain();
			assertTrue(d.active());
			assertEquals(Set.of(26, 44, 30, Keys.MOUSE_LEFT), d.held());
			assertEquals(4, d.presses().size());
			// same held set again: no new edges
			c.call("{\"act\":{\"keys\":[\"W\",\"SPACE\",\"1\"],\"mb\":[1,0],\"cam\":[0,0]}}");
			assertEquals(List.of(), state.drain().presses());
			// unknown token: error reply, known keys still apply
			JsonObject r = c.call("{\"act\":{\"keys\":[\"W\",\"NOPE\"],\"mb\":[0,1]}}");
			assertNotNull(r.get("error"));
			assertEquals(Set.of(26, Keys.MOUSE_RIGHT), state.drain().held());
		}
	}

	@Test
	void probeDoesNotDisturbController() throws Exception {
		try (Client c = new Client(port)) {
			c.call("{\"act\":{\"keys\":[\"W\"],\"mb\":[0,0],\"cam\":[0,0]}}");
			state.drain();
			for (int i = 0; i < 3; i++) {
				new Socket("127.0.0.1", port).close(); // dock probe: connect + close, no bytes
			}
			Thread.sleep(100);
			BridgeState.Drain d = state.drain();
			assertTrue(d.active());
			assertFalse(d.release());
			assertEquals(Set.of(26), d.held());
		}
	}

	@Test
	void newestActSenderWinsAndOnlyControllerDisconnectReleases() throws Exception {
		try (Client a = new Client(port); Client b = new Client(port)) {
			a.call("{\"act\":{\"keys\":[\"W\"]}}");
			b.call("{\"act\":{\"keys\":[\"S\"]}}");
			assertEquals(Set.of(22), state.drain().held());
			a.close();
			Thread.sleep(150);
			assertTrue(state.isActive(), "old controller leaving must not release");
			assertFalse(state.drain().release());
			b.close();
			await(() -> !state.isActive());
			assertTrue(state.drain().release());
		}
	}

	@Test
	void releaseOnlyFromController() throws Exception {
		try (Client a = new Client(port); Client b = new Client(port)) {
			a.call("{\"act\":{\"keys\":[\"W\"]}}");
			assertTrue(b.call("{\"release\":true}").get("ok").getAsBoolean());
			assertTrue(state.isActive());
			a.call("{\"release\":true}");
			assertFalse(state.isActive());
			assertTrue(state.drain().release());
		}
	}

	@Test
	void deadManReleasesOnceThenIdlePingsDoNothing() throws Exception {
		try (Client c = new Client(port)) {
			c.call("{\"act\":{\"keys\":[\"W\"]}}");
			state.drain();
			Thread.sleep(1100);
			state.checkDeadMan();
			assertFalse(state.isActive());
			assertTrue(state.drain().release());
			for (int i = 0; i < 4; i++) {
				c.call("{\"ping\":1}");
				state.checkDeadMan();
				assertFalse(state.drain().release(), "idle pings must not release again");
			}
			Thread.sleep(1100);
			state.checkDeadMan();
			assertFalse(state.drain().release());
		}
	}

	@Test
	void pingsFromControllerFeedDeadMan() throws Exception {
		try (Client c = new Client(port)) {
			c.call("{\"act\":{\"keys\":[\"W\"]}}");
			for (int i = 0; i < 4; i++) {
				Thread.sleep(400);
				c.call("{\"ping\":1}");
				state.checkDeadMan();
			}
			assertTrue(state.isActive());
		}
	}

	@Test
	void neverBlocksOnUnreadReplies() throws Exception {
		try (Client c = new Client(port)) {
			// never read the {"ok":true} replies; ~1.5 MB of replies would fill any socket buffer
			StringBuilder sb = new StringBuilder();
			for (int i = 0; i < 100_000; i++) {
				sb.append("{\"act\":{\"keys\":[\"W\"],\"cam\":[1,0]}}\n");
			}
			sb.append("{\"act\":{\"keys\":[\"D\"],\"cam\":[0,0]}}\n");
			c.out.write(sb.toString().getBytes(StandardCharsets.UTF_8));
			c.out.flush();
			await(() -> state.isHeld(7)); // D
		}
	}

	@Test
	void camAccumulatesAndSpreadsUntilDeadline() throws Exception {
		try (Client c = new Client(port)) {
			state.takeCam(false); // start the frame clock
			c.call("{\"act\":{\"keys\":[],\"cam\":[30,-10]}}");
			c.call("{\"act\":{\"keys\":[\"W\"],\"cam\":[0,0]}}"); // key-change act adds nothing, keeps the pending cam
			c.call("{\"act\":{\"keys\":[\"W\"],\"cam\":[10,0]}}");
			double sx = 0;
			double sy = 0;
			int frames = 0;
			long end = System.nanoTime() + 200_000_000L;
			while (System.nanoTime() < end) {
				double[] d = state.takeCam(false);
				if (d != null) {
					sx += d[0];
					sy += d[1];
					frames++;
				}
				Thread.sleep(5);
			}
			assertEquals(40.0, sx, 1e-9);
			assertEquals(-10.0, sy, 1e-9);
			assertTrue(frames > 1, "camera should be spread over several frames");
			assertEquals(null, state.takeCam(true));
		}
	}
}
