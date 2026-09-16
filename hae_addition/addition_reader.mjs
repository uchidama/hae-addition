// Addition Reader for Drosophila mushroom body model.
// Split-PN encoding for 2 digit images -> 19 MBON compartments (sums 0 to 18).

export const ADDITION_CONFIG = {
  size: 12,
  npix: 144,
  hz: 220,
  ink: 0.05,
  ms: 400,
  feedbackMs: 300,
  eta: 6e-5,
  gainMin: 0.05,
  gainMax: 2.0,
  classes: 19,
  labels: Array.from({ length: 19 }, (_, i) => String(i)),
};

async function readBytes(url) {
  if (url.protocol === 'file:') {
    const { readFile } = await import('node:fs/promises');
    return new Uint8Array(await readFile(url));
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> HTTP ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

/**
 * @param {object} o
 * @param {typeof import('../flybrain/flybrain.js').FlyBrain} o.FlyBrain
 * @param {URL} o.base        the flybrain/ directory
 * @param {string} [o.v]      cache-busting query
 * @param {(p:any)=>void} [o.onProgress]
 */
export async function makeAdditionReader({ FlyBrain, base, v = '', onProgress }) {
  const C = ADDITION_CONFIG;
  const mb = JSON.parse(new TextDecoder().decode(await readBytes(new URL('data/mb783.json' + v, base))));
  const brain = await FlyBrain.load({
    graph: new URL('data/flywire783.fbg.gz' + v, base),
    wasm: new URL('flybrain.wasm' + v, base),
    onProgress,
  });

  const G = mb.groups;
  const pick = (p) => Object.keys(G).filter((k) => k.startsWith(p)).sort();
  const all = (keys) => keys.flatMap((k) => G[k].idx);
  const KC = all(pick('kc:')), MBON = all(pick('mbon:')), DAN = all(pick('dan:')),
        MBIN = all(pick('mbin:')), ALPN = all(pick('alpn:'));

  // Mushroom body alone is simulated (silence other ~130k neurons to avoid runaway)
  const inside = new Set([...KC, ...MBON, ...DAN, ...MBIN, ...ALPN]);
  const outside = [];
  for (let i = 0; i < brain.n; i++) if (!inside.has(i)) outside.push(i);
  brain.silence(outside, true, { byIndex: true });

  // How many Kenyon cells reach each MBON
  const kcIn = new Map();
  const mbonSet = new Set(MBON);
  for (const k of KC) {
    const { post } = brain.outgoing(k, { byIndex: true });
    for (const i of post) if (mbonSet.has(i)) kcIn.set(i, (kcIn.get(i) || 0) + 1);
  }

  // Deal all 96 MBONs out to the 19 sum compartments (0 to 18)
  // Greedy Balancing: best-connected first, each to whichever group has fewest KC inputs so far
  const NL = C.classes; // 19
  const typeOf = new Map();
  for (const t of pick('mbon:')) for (const i of G[t].idx) typeOf.set(i, t.slice(5));
  const groups = Array.from({ length: NL }, () => ({ cells: [], types: [], inputs: 0 }));
  for (const i of [...kcIn.keys()].sort((a, b) => kcIn.get(b) - kcIn.get(a) || a - b)) {
    let g = groups[0];
    for (const h of groups) if (h.inputs < g.inputs) g = h;
    g.cells.push(i);
    g.types.push(typeOf.get(i));
    g.inputs += kcIn.get(i);
  }
  for (const g of groups) g.name = [...new Set(g.types)].join('+');

  const counts = brain.setPlasticity({
    pre: KC,
    groups: groups.map((g) => ({ post: g.cells, modulators: [] })),
    eta: 0,
    tauTrace: 40,
    tauDopa: 1e7,
    gainMin: C.gainMin,
    gainMax: C.gainMax,
  });
  groups.forEach((g, c) => (g.synapses = counts[c]));

  // Split-PN encoding: 685 ALPNs divided into left group (342) and right group (343)
  const npix = C.npix; // 144
  let rs = 12345;
  const rnd = () => ((rs = (rs * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  const deck = [...ALPN];
  for (let i = deck.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [deck[i], deck[j]] = [deck[j], deck[i]];
  }

  const half = Math.floor(deck.length / 2); // 342
  const leftPNs = deck.slice(0, half);     // 342 ALPNs for left digit
  const rightPNs = deck.slice(half);       // 343 ALPNs for right digit

  const chanLeft = Array.from({ length: npix }, () => []);
  leftPNs.forEach((p, i) => chanLeft[i % npix].push(p));

  const chanRight = Array.from({ length: npix }, () => []);
  rightPNs.forEach((p, i) => chanRight[i % npix].push(p));

  let seed = 1;

  function stimulate(imgLeft, imgRight) {
    brain.clearStimuli();
    for (let i = 0; i < npix; i++) {
      if (imgLeft[i] > C.ink) brain.stimulate(chanLeft[i], C.hz * imgLeft[i], { byIndex: true });
      if (imgRight[i] > C.ink) brain.stimulate(chanRight[i], C.hz * imgRight[i], { byIndex: true });
    }
  }

  /**
   * Present both digit images (C.ms ms) and read relative drive for 19 sum compartments.
   */
  function look(imgLeft, imgRight, { seed: s } = {}) {
    stimulate(imgLeft, imgRight);
    brain.setPlasticityParams({ eta: 0, tauTrace: 40, tauDopa: 1e7, gainMin: C.gainMin, gainMax: C.gainMax });
    brain.reset(s ?? ++seed);
    brain.run(C.ms, { events: false });

    const c = brain.counts();
    const spikes = new Uint16Array(KC.length);
    const slots = [];
    for (let k = 0; k < KC.length; k++) {
      const n = c[KC[k]];
      if (n) {
        spikes[k] = Math.min(65535, n);
        slots.push(k);
      }
    }
    const drive = brain.driveByGroup(KC, spikes);
    return { drive, slots: Uint16Array.from(slots) };
  }

  /**
   * Decide answer among 19 sum compartments (lowest drive).
   */
  function decide(drive, allowed = null) {
    const cand = allowed ?? [...Array(NL).keys()];
    let best = cand[0], second = null;
    for (const c of cand) if (drive[c] < drive[best]) best = c;
    for (const c of cand) if (c !== best && (second == null || drive[c] < drive[second])) second = c;
    return { answer: best, second, margin: second == null ? 1 : drive[second] - drive[best] };
  }

  /**
   * Error-driven dopamine feedback:
   * Right sum weakened (LTD, +1 dopamine), wrong pick strengthened (LTP, -1 dopamine).
   */
  function feedback(truth, wrong) {
    brain.setPlasticityParams({ eta: C.eta, tauTrace: 40, tauDopa: 1e7, gainMin: C.gainMin, gainMax: C.gainMax });
    brain.dopamine(truth, 1);
    if (wrong != null && wrong !== truth) brain.dopamine(wrong, -1);
    brain.run(C.feedbackMs, { events: false });
    brain.dopamine(truth, 0);
    if (wrong != null) brain.dopamine(wrong, 0);
    brain.setPlasticityParams({ eta: 0, tauTrace: 40, tauDopa: 1e7, gainMin: C.gainMin, gainMax: C.gainMax });
  }

  /**
   * One practice step: look, answer, feedback if incorrect.
   */
  function practise(imgLeft, imgRight, truth, allowed = null) {
    const seen = look(imgLeft, imgRight);
    const d = decide(seen.drive, allowed);
    const ok = d.answer === truth;
    if (!ok) feedback(truth, d.answer);
    return { ok, answer: d.answer, margin: d.margin, slots: seen.slots };
  }

  return {
    config: C,
    brain,
    KC,
    ALPN,
    leftPNs,
    rightPNs,
    groups,
    labels: C.labels,
    look,
    decide,
    feedback,
    practise,
    exportGains: () => brain.exportGains(),
    importGains: (g) => brain.importGains(g),
    forget: () => brain.forget(),
  };
}
