// Web Worker for Tashizan (Addition) FlyBrain Demo

import { FlyBrain } from '../flybrain/flybrain.js?v=6';
import { makeAdditionReader } from '../hae_addition/addition_reader.mjs?v=1';

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

    R = await makeAdditionReader({
      FlyBrain,
      base: new URL('../flybrain/', HERE),
      onProgress: (p) => post({ type: 'progress', ...p }),
    });

    post({ type: 'progress', phase: 'loading_weights', message: '足し算の学習済み重みを読み込み中...' });
    const weightRes = await fetch(new URL('../results/addition_all_pairs/brain-addition-5000.bin.gz', HERE));
    if (!weightRes.ok) throw new Error(`Weights HTTP ${weightRes.status}`);
    const weightBuf = await gunzipBytes(weightRes);
    const gains = new Float32Array(weightBuf.buffer, weightBuf.byteOffset, weightBuf.byteLength / 4);
    R.importGains(gains);

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
      classes: R.labels.length,
      labels: R.labels,
      message: 'ハエ脳の準備が完了しました！',
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
  const { type, payload } = e.data;

  if (type === 'init') {
    await init();
  } else if (type === 'sample_and_ask') {
    if (!R || !testBank) return;
    const sample = samplePair(payload?.leftDigit, payload?.rightDigit);
    const seen = R.look(sample.imgL, sample.imgR);
    const decision = R.decide(seen.drive);

    post({
      type: 'answer',
      mode: 'sample',
      leftDigit: sample.leftDigit,
      rightDigit: sample.rightDigit,
      target: sample.target,
      imgL: Array.from(sample.imgL),
      imgR: Array.from(sample.imgR),
      predictedSum: decision.answer,
      isCorrect: decision.answer === sample.target,
      margin: decision.margin,
      drive: Array.from(seen.drive),
    });
  } else if (type === 'ask') {
    if (!R) return;
    const imgL = new Float32Array(payload.imgL);
    const imgR = new Float32Array(payload.imgR);
    const seen = R.look(imgL, imgR);
    const decision = R.decide(seen.drive);

    post({
      type: 'answer',
      mode: 'custom',
      imgL: payload.imgL,
      imgR: payload.imgR,
      predictedSum: decision.answer,
      margin: decision.margin,
      drive: Array.from(seen.drive),
    });
  }
};
