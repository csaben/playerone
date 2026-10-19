# playerone-bridge

Fabric client mod for Minecraft Java 26.3. It gives the player-one agent an in-game input path so the agent can play while Minecraft is unfocused or behind another window. The mod never touches the real mouse or keyboard, never moves the OS cursor and never takes focus.

- It listens on `127.0.0.1:8970`. Override the port with `-Dplayerone.bridge.port=NNNN` in the profile's JVM arguments.
- It only needs Fabric Loader. Fabric API is not required.
- Log lines start with `[playerone-bridge]` and go to the game log (`logs/latest.log` in the game directory).

## Protocol

The connection is TCP and carries newline-delimited JSON objects, one per line.

### Connections and the controller

- Any number of connections can be open at once. A bare connect and close (the dock's probe) has no effect.
- A connection becomes the **controller** when it sends an `act`. The most recent sender of an `act` wins.
- Held state is released only in these cases:
  - the controller disconnects
  - the controller sends `release`
  - the controller goes **1 s** without sending any message (dead-man switch). Pings count as messages.
- After a release, the bridge stays idle until the next `act`. An idle client that keeps pinging causes no further releases.
- Connections that send nothing, or only `ping`, never change held state.

### Client to mod

| Message | Meaning |
|---|---|
| `{"act":{"keys":["W","SPACE","1"],"mb":[l,r],"cam":[dx,dy]}}` | `keys` is the full set of physical keys held right now. `mb` is `[left,right]`, each 0/1 or a boolean. `cam` is raw mouse counts. All three fields are optional. |
| `{"release":true}` | Release everything. This is ignored unless it comes from the controller. |
| `{"ping":1}` | The mod replies with a pong (see the next section). |

Key tokens: `A`-`Z`, `0`-`9`, `SPACE ENTER ESC TAB BACKSPACE LEFT UP RIGHT DOWN LSHIFT RSHIFT LCTRL RCTRL LALT RALT`. They are mapped to SDL scancodes, which identify physical key positions, so the keyboard layout does not matter.

### Mod to client

- Pong: `{"pong":1,"pid":<java pid>,"mc":"26.3","focused":bool,"paused":bool,"screen":"<screen class>"|null,"bridge":"<mod version>"}`
- Reply to `act` or `release`: `{"ok":true}`, or `{"error":"..."}`.
  - An unknown key token gets an error reply, but the valid keys in that act still apply.
  - Replies are queued and written by a separate thread. If the client never reads them, they are dropped and reading is never blocked. Clients can ignore everything except pongs.

### What happens in game

These run on the client thread.

- **Keys and buttons drive whatever `KeyMapping`s are bound to that physical key or button**, so the user's keybinds are respected. The left and right mouse buttons are SDL buttons 1 and 3 (attack and use by default).
  - **On a press:** the click count goes up by one, so edge-triggered binds (inventory, hotbar, drop, swap) fire once.
  - **While held:** `setDown(true)` is applied again at the start of every client tick.
  - **On release:** the binding goes up.
  - **Tap between ticks:** if a press and release both arrive between two ticks, the key is held for one tick.
  - **Toggle mode** (toggle sprint, sneak, attack or use in the user's options) behaves like the physical key: each press flips the mapping.
- **Camera:** `cam` counts from all acts add up. They are spread linearly over render frames until 50 ms after the newest act that had non-zero `cam`.
  - Scaling matches vanilla `MouseHandler.turnPlayer`: `f = sens*0.6+0.2`, `e = f^3*8`, then `player.turn(dx*e, dy*e)`, honoring invert X/Y. `Entity.turn` applies the 0.15 factor.
  - `cam` is discarded while a screen is open or the game is paused.
- **Screens:** while a screen is open, held keys are not applied (vanilla releases all mappings when a screen opens). Two presses still work:
  - `ESC` closes screens that allow it.
  - The inventory key closes container screens.
  - In game, `ESC` opens the pause menu, the same as vanilla.
- **Unfocused play:** while a controller is active:
  - Vanilla's pause-on-focus-loss is skipped through a mixin. The options file is not changed.
  - If the pause menu is already open when control starts and the window is unfocused, it is closed.
  - Mining continues even though the mouse is not grabbed.
  - The idle frame limiter is kept awake.

## Build

```bash
./gradlew build      # -> build/libs/playerone-bridge-0.1.0.jar (also runs unit tests)
```

The Gradle daemon and the compile toolchain are Java 25. Gradle downloads them automatically through foojay (`gradle/gradle-daemon-jvm.properties`), so the Java 17 on PATH is enough to start the wrapper.

Versions: Minecraft 26.3, which is unobfuscated, so there is no `mappings` line. Fabric Loader 0.19.5 (latest stable, and the mod also accepts 0.20.x). Loom 1.18-SNAPSHOT (resolves to 1.18.4). Gradle 9.7.1.

## Install (Windows)

The setup below uses a separate game directory, so your vanilla saves, options and mods stay untouched.

1. Get the Fabric installer (needs Java 8 or newer; the Java 17 on PATH works):

   ```powershell
   curl.exe -L -o fabric-installer.jar https://maven.fabricmc.net/net/fabricmc/fabric-installer/1.1.2/fabric-installer-1.1.2.jar
   ```

2. Install Fabric for 26.3. This adds a `fabric-loader-0.19.5-26.3` version and a "fabric-loader-26.3" profile to the launcher. Vanilla files are not modified.

   ```powershell
   java -jar fabric-installer.jar client -mcversion 26.3 -loader 0.19.5 -dir "$env:APPDATA\.minecraft"
   ```

3. Create a game directory for player-one and copy the mod into it:

   ```powershell
   New-Item -ItemType Directory -Force "$env:APPDATA\.minecraft-player-one\mods"
   Copy-Item D:\Code\player-one\mcmod\build\libs\playerone-bridge-0.1.0.jar "$env:APPDATA\.minecraft-player-one\mods\"
   ```

4. In the Minecraft Launcher, go to Installations and edit "fabric-loader-26.3":
   - Rename it to "player-one".
   - Set **Game directory** to `%APPDATA%\.minecraft-player-one`.
   - Optionally, add `-Dplayerone.bridge.port=8970` under More options, then JVM arguments.

5. Launch the "player-one" profile and enter a world. The log should show `[playerone-bridge] listening on 127.0.0.1:8970`.

## Smoke test

Run this with the game in a world. Minecraft can stay unfocused behind another window.

```bash
uv run --no-project python tools/bridge_client.py          # ping, hold W 1 s, turn right, release, ping
uv run --no-project python tools/bridge_client.py --port 8971
```

## Mixin targets (verified against 26.3 bytecode)

- `Minecraft#tick()V` at HEAD: applies held state and press edges.
- `Minecraft#runTick(Z)V`, after `MouseHandler#handleAccumulatedMovement()V`: applies per-frame camera and publishes focused, paused and screen status.
- `Minecraft#pauseIfInactive()V` at HEAD: cancelled while a controller is active.
- `Minecraft#handleKeybinds()V`, redirect of `MouseHandler#isMouseGrabbed()Z`: the only call there is the `continueAttack(...)` condition. It returns true while the bridge holds attack.
- Accessors: `KeyMapping.clickCount`, `ToggleKeyMapping.needsToggle`, and an invoker for `ToggleKeyMapping#reset()V`.
