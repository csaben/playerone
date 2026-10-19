const test = require('node:test');
const assert = require('node:assert');
const { filter } = require('../src/ui/search');

const wins = [
  { app: 'WindowsTerminal', title: 'psmux' },
  { app: 'Code', title: 'main.js - player-one - Visual Studio Code' },
  { app: 'chrome', title: 'Inbox - Gmail - Google Chrome' },
  { app: 'notepad', title: 'Untitled - Notepad' },
];

test('empty query keeps the order', () => {
  assert.deepStrictEqual(filter('', wins), wins);
});

test('prefix of the app name wins', () => {
  assert.strictEqual(filter('note', wins)[0].app, 'notepad');
});

test('words from the title find the window', () => {
  assert.strictEqual(filter('gmail', wins)[0].app, 'chrome');
  assert.strictEqual(filter('player', wins)[0].app, 'Code');
});

test('letters in order match', () => {
  assert.strictEqual(filter('vsc', wins)[0].app, 'Code');
});

test('every word must match', () => {
  assert.deepStrictEqual(filter('notepad gmail', wins), []);
});
