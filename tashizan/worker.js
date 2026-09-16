// Web Worker for Tashizan (Addition) FlyBrain Demo
// Powered by Dual-Mushroom Body Architecture:
// Stage 1 (MB1): Visual Single-Digit LIF Mushroom Body (0-9 recognition, full ALPN resolution)
// Stage 2 (MB2): Associative Addition Mushroom Body (A, B -> A+B sum, 100% converged)

import { FlyBrain } from '../flybrain/flybrain.js?v=7';
import { makeDualMBReader } from '../hae_addition/dual_mb_reader.mjs?v=2';

const HERE = new URL('./', import.meta.url);
const post = (msg) => self.postMessage(msg);

let R = null;
let testBank = null;

async function gunzipBytes(res) {
  const s = res.body.pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(s).arrayBuffer());
}

async function init() {
  try {
    post({ type: 'progress', phase: 'downloading_brain', message: 'ハエ脳モデル (WASM/Connectome) を読み込み中...' });

    let mb1Url = new URL('../results/mb1_digits/brain-mb1-final.bin.gz', HERE);
    try {
      const chk = await fetch(mb1Url, { method: 'HEAD' });
      if (!chk.ok) {
        mb1Url = new URL('../results/mb1_digits/brain-mb1-1500.bin.gz', HERE);
      }
    } catch {
      mb1Url = new URL('../results/mb1_digits/brain-mb1-1500.bin.gz', HERE);
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

    post({
      type: 'ready',
      classes: 19,
      labels: Array.from({ length: 19 }, (_, i) => String(i)),
      kc: Array.from(R.mb1.KC),
      message: '2段キノコ体モデル（視覚認識 MB1 + 連想記憶 MB2）の準備が完了しました！',
    });
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
      const img = new Float32Array(raw);
      const seen = R.mb1.look(img);
      const dec = R.mb1.decide(seen.drive);
      post({
        type: 'answer_single',
        which: p.which,
        digit: dec.answer,
        margin: dec.margin,
        drive: Array.from(seen.drive),
        slots: Array.from(seen.slots),
      });
    } else if (type === 'ask_pair') {
      if (!R) return;
      const rawL = p.imgL;
      const rawR = p.imgR;
      if (!rawL || !rawR) throw new Error('Missing imgL or imgR in ask_pair');
      const imgL = new Float32Array(rawL);
      const imgR = new Float32Array(rawR);

      const res = R.lookAndAdd(imgL, imgR);

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
        slots: res.slotsA,
        slotsA: res.slotsA,
        slotsB: res.slotsB,
      });
    }
  } catch (err) {
    console.error('Worker error:', err);
    post({ type: 'error', message: err.message });
  }
};
