// What runs when you attach to a window. A macro agent gets the window and a `type` function
// that sends keys to it. A policy agent instead plays the window until you detach: the recorder
// streams frames to a policy server (config "policy", train/serve.py) and presses what it returns,
// backing off whenever you touch the keyboard, mouse or pad.

const AGENTS = {
  hello: {
    label: 'hello world',
    async run(win, { type }) {
      return type('hello world');
    },
  },
  bc: {
    label: 'behavior clone',
    policy: true,
  },
};

const DEFAULT = 'hello';

module.exports = { AGENTS, DEFAULT };
