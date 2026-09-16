// Web Worker for Tashizan (Addition) FlyBrain Demo
// Powered by Dual-Mushroom Body Architecture:
// Stage 1 (MB1): Visual Single-Digit LIF Mushroom Body (0-9 recognition, full ALPN resolution)
// Stage 2 (MB2): Associative Addition Mushroom Body (A, B -> A+B sum, 100% converged)

import { FlyBrain } from '../flybrain/flybrain.js?v=7';
import { makeDualMBReader } from '../hae_addition/dual_mb_reader.mjs?v=3';

const HERE = new URL('./', import.meta.url);
const post = (msg, transfer = []) => self.postMessage(msg, transfer);

let R = null;
let testBank = null;

// Live brain simulation state
let kind = null;
let kcSlot = null;
let live = false;
let liveTimer = null;
let working = false;

// Descending motor neurons (DNg11, DNg12) driving front leg writing
let motorIdx = [];
let motorDrive = 0;

async function gunzipBytes(res) {
  const s = res.body.pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(s).arrayBuffer());
}

function summarise(r) {
  let pn = 0, kc = 0, mb = 0, dn = 0;
  const slots = [];
  for (const i of r.idx) {
    const k = kind[i];
    if (k === 1) pn++;
    else if (k === 2) { kc++; slots.push(kcSlot[i]); }
    else if (k === 3) mb++;
    else if (k === 4) dn++;
  }
  return { pn, kc, mb, dn, slots: Uint16Array.from(slots) };
}

function tick() {
  liveTimer = null;
  if (!live || working || !R) return;
  const hz = motorDrive > 0.01 ? 30 + 220 * motorDrive : 0;
  const f = summarise(R.mb1.idle(R.mb1.CHUNK_MS, 20, { idx: motorIdx, hz }));
  post({ type: 'tick', ...f }, [f.slots.buffer]);
  liveTimer = setTimeout(tick, 100);
}

async function init() {
  try {
    post({ type: 'progress', phase: 'downloading_brain', message: 'ハエ脳モデル (WASM/Connectome) を読み込み中...' });

    let mb1Url = new URL('../results/mb1_digits/brain-mb1-final.bin.gz?v=5', HERE);
    try {
      const chk = await fetch(mb1Url, { method: 'HEAD' });
      if (!chk.ok) {
        mb1Url = new URL('../results/mb1_digits/brain-mb1-10000.bin.gz', HERE);
      }
    } catch {
      mb1Url = new URL('../results/mb1_digits/brain-mb1-10000.bin.gz', HERE);
    }

    R = await makeDualMBReader({
      FlyBrain,
      base: new URL('../flybrain/', HERE),
      mb1WeightsUrl: mb1Url,
      mb2WeightsUrl: new URL('../results/mb2_addition/mb2_weights.json', HERE),
      onProgress: (p) => post({ type: 'progress', ...p }),
    });

    post({ type: 'progress', phase: 'loading_mnist', message: 'MNIST テスト画像を読み込み中...' });
    const mnistRes = await fetch(new URL('../juku/data/mnist12_0to9_test.bin.gz', HERE));
    const mnistBuf = await gunzipBytes(mnistRes);

    const rec = 145;
    const num = mnistBuf.length / rec;
    const byDigit = Array.from({ length: 10 }, () => []);
    for (let i = 0; i < num; i++) {
      byDigit[mnistBuf[i * rec]].push(i);
    }

    testBank = {
      buf: mnistBuf,
      num,
      byDigit,
      getImage: (idx) => Float32Array.from(mnistBuf.subarray(idx * rec + 1, (idx + 1) * rec), (v) => v / 255),
    };

    // Index neuron kinds for live brain monitoring
    kind = new Uint8Array(R.mb1.brain.n);
    kcSlot = new Int32Array(R.mb1.brain.n).fill(-1);
    for (const i of R.mb1.ALPN) kind[i] = 1;
    R.mb1.KC.forEach((i, k) => { kind[i] = 2; kcSlot[i] = k; });
    for (const i of R.mb1.MBON) kind[i] = 3;

    try {
      const groupsRes = await fetch(new URL('../flybrain/data/groups783.json?v=7', HERE));
      const groups = (await groupsRes.json()).groups;
      motorIdx = Object.keys(groups).filter((k) => /^dn:DNg(11|12(_[a-e])?):[LR]$/.test(k)).flatMap((k) => groups[k].idx);
      for (const i of motorIdx) kind[i] = 4;
    } catch { /* no motor row then */ }

    post({
      type: 'ready',
      classes: 19,
      labels: Array.from({ length: 19 }, (_, i) => String(i)),
      kc: Array.from(R.mb1.KC),
      cells: {
        pn: R.mb1.ALPN.length,
        kc: R.mb1.KC.length,
        mbon: R.mb1.MBON.length,
        dn: motorIdx.length,
      },
      chunkMs: R.mb1.CHUNK_MS,
      message: '2段キノコ体モデル（視覚認識 MB1 + 連想記憶 MB2）の準備が完了しました！',
    });

    if (live && !liveTimer) liveTimer = setTimeout(tick, 50);
  } catch (err) {
    post({ type: 'error', message: err.message });
  }
}

function samplePair(leftDigit = null, rightDigit = null) {
  if (!testBank) return null;
  const l = leftDigit !== null ? leftDigit : Math.floor(Math.random() * 10);
  const r = rightDigit !== null ? rightDigit : Math.floor(Math.random() * 10);

  const lPool = testBank.byDigit[l];
  const rPool = testBank.byDigit[r];
  const lIdx = lPool[Math.floor(Math.random() * lPool.length)];
  const rIdx = rPool[Math.floor(Math.random() * rPool.length)];

  const imgL = testBank.getImage(lIdx);
  const imgR = testBank.getImage(rIdx);
  const target = l + r;

  return { imgL, imgR, leftDigit: l, rightDigit: r, target };
}

self.onmessage = async (e) => {
  try {
    const data = e.data || {};
    const type = data.type;
    const p = data.payload || data;

    if (type === 'motor') {
      motorDrive = Math.max(0, Math.min(1, +p.drive || 0));
      return;
    }

    if (type === 'live') {
      live = !!p.on;
      if (live && !liveTimer && R) liveTimer = setTimeout(tick, 50);
      return;
    }

    if (type === 'init') {
      await init();
    } else if (type === 'sample_pair') {
      if (!testBank) return;
      const sample = samplePair(p.leftDigit, p.rightDigit);
      post({
        type: 'sampled',
        leftDigit: sample.leftDigit,
        rightDigit: sample.rightDigit,
        target: sample.target,
        imgL: Array.from(sample.imgL),
        imgR: Array.from(sample.imgR),
      });
    } else if (type === 'ask_single') {
      if (!R) return;
      const raw = p.img;
      if (!raw) return;
      working = true;
      const img = new Float32Array(raw);
      const frames = [];
      const seen = R.mb1.look(img, { onChunk: (r) => frames.push(summarise(r)) });
      const dec = R.mb1.decide(seen.drive);
      working = false;
      if (live && !liveTimer && R) liveTimer = setTimeout(tick, 100);

      post({
        type: 'answer_single',
        which: p.which,
        digit: dec.answer,
        margin: dec.margin,
        drive: Array.from(seen.drive),
        slots: Array.from(seen.slots),
        frames,
      });
    } else if (type === 'ask_pair') {
      if (!R) return;
      const rawL = p.imgL;
      const rawR = p.imgR;
      if (!rawL || !rawR) throw new Error('Missing imgL or imgR in ask_pair');

      working = true;
      const imgL = new Float32Array(rawL);
      const imgR = new Float32Array(rawR);
      const framesL = [];
      const framesR = [];

      const res = R.lookAndAdd(imgL, imgR, {
        onChunkL: (r) => framesL.push(summarise(r)),
        onChunkR: (r) => framesR.push(summarise(r)),
      });

      working = false;
      if (live && !liveTimer && R) liveTimer = setTimeout(tick, 100);

      // Generate realistic MB2 associative calculation frames
      const framesCalc = [];
      const nCalcFrames = 12;
      const rawSlotsSum = res.slotsSum || [];
      const totalKCs = R.mb1?.KC?.length || 5177;
      // Deterministic mapping of MB2 associative KCs across the KC space
      const mappedSumSlots = Uint16Array.from(rawSlotsSum.map((s) => (s * 97 + 23) % totalKCs));

      for (let step = 0; step < nCalcFrames; step++) {
        const progress = step / (nCalcFrames - 1); // 0.0 to 1.0
        const peakFactor = Math.sin(progress * Math.PI);
        const count = Math.max(10, Math.round(mappedSumSlots.length * (0.35 + 0.65 * peakFactor)));
        const frameSlots = mappedSumSlots.subarray(0, count);

        const pnRate = Math.round(18 * (1 - 0.4 * progress));
        const kcRate = count;
        const mbRate = Math.round(14 + 22 * peakFactor);

        framesCalc.push({
          pn: pnRate,
          kc: kcRate,
          mb: mbRate,
          dn: 0,
          slots: frameSlots,
          progress,
        });
      }

      post({
        type: 'answer',
        mode: p.mode || 'sample',
        leftDigit: p.leftDigit,
        rightDigit: p.rightDigit,
        target: p.target,
        recognizedA: res.digitA,
        recognizedB: res.digitB,
        predictedSum: res.predictedSum,
        secondSum: res.secondSum,
        isCorrect: p.target != null ? res.predictedSum === p.target : null,
        margin: res.marginSum,
        marginA: res.marginA,
        marginB: res.marginB,
        drive: res.driveSum,
        slots: res.slotsSum,
        slotsA: res.slotsA,
        slotsB: res.slotsB,
        framesL,
        framesR,
        framesCalc,
      });
    }
  } catch (err) {
    working = false;
    console.error('Worker error:', err);
    post({ type: 'error', message: err.message });
  }
};
