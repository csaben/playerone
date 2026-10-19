# Copied from agentpc

`agentpc-core`, `agentpc-capture` and `agentpc-record` are copies of `../agentpc/crates/*` at commit
`8344d18`, so player-one writes the same shards (SCHEMA.md in agentpc) that agentpc's trainers read.

Changes from upstream:

- `agentpc-capture/src/human_tap.rs`: a device mask (`set_devices`, `DEV_PADS`, `DEV_KEYBOARD`,
  `DEV_MOUSE`) and `connected_pads()`, so player-one can choose which controllers, and whether the
  keyboard and mouse, are recorded. Upstream reads all four XInput slots plus keyboard and mouse.

When agentpc's schema changes, copy the crates again and reapply that change.
