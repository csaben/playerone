// Fuzzy search over windows for the panel's "/" filter. Every word of the query must match
// somewhere: a prefix beats a substring, which beats letters in order ("vsc" finds
// "Visual Studio Code"). The app name counts a little more than the title.
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.POSearch = factory();
})(this, () => {
  const FIELDS = [['app', 3], ['title', 2.5]];

  // letters of `q` in order in `s`; tighter runs score higher, 0 if they aren't all there
  function subseq(q, s) {
    let at = -1;
    let gaps = 0;
    for (const ch of q) {
      const i = s.indexOf(ch, at + 1);
      if (i < 0) return 0;
      if (at >= 0) gaps += i - at - 1;
      at = i;
    }
    return 1 / (1 + gaps * 0.3);
  }

  function word(w, x) {
    let best = 0;
    for (const [f, weight] of FIELDS) {
      const s = String(x[f] || '').toLowerCase();
      if (!s) continue;
      let sc = 0;
      if (s === w) sc = 12;
      else if (s.startsWith(w) || s.split(/[\s._/\\|:—–-]+/).some((part) => part.startsWith(w))) sc = 9;
      else if (s.includes(w)) sc = 6;
      else if (w.length >= 2) sc = 4 * subseq(w, s);
      best = Math.max(best, sc * weight);
    }
    return best;
  }

  function score(query, x) {
    const words = String(query).toLowerCase().split(/\s+/).filter(Boolean);
    let total = 0;
    for (const w of words) {
      const s = word(w, x);
      if (!s) return 0;
      total += s;
    }
    return total;
  }

  // items that match, best first; ties keep the given order
  function filter(query, items) {
    if (!String(query).trim()) return items.slice();
    return items.map((x, i) => ({ x, i, s: score(query, x) }))
      .filter((r) => r.s > 0)
      .sort((a, b) => b.s - a.s || a.i - b.i)
      .map((r) => r.x);
  }

  return { score, filter };
});
