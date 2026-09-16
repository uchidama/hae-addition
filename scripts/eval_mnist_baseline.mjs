import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';
import { FlyBrain } from '../flybrain/flybrain.js';
import { makeReader } from '../juku/reader.mjs';

const ROOT = new URL('../', import.meta.url);
const RESULTS_DIR = new URL('results/mnist_baseline/', ROOT);
await mkdir(RESULTS_DIR, { recursive: true });

console.log('Loading flybrain reader for suji...');
const R = await makeReader({ FlyBrain, base: new URL('flybrain/', ROOT), course: 'suji' });
const NL = R.labels.length;

// Load learned weights
const brainGz = await readFile(new URL('juku/state/suji/brain-110217.bin.gz', ROOT));
const gains = new Float32Array(new Uint8Array(gunzipSync(brainGz)).buffer);
R.importGains(gains);
console.log('Imported gains:', gains.length, 'synapses');

// Load test data
const testBuf = new Uint8Array(gunzipSync(await readFile(new URL('juku/data/mnist12_test.bin.gz', ROOT))));
const rec = 145;
const nTotal = testBuf.length / rec;
const img = (buf, n) => Float32Array.from(buf.subarray(n * rec + 1, (n + 1) * rec), (v) => v / 255);

console.log(`Loaded test set: ${nTotal} images`);

// 1. Evaluate on standard test set (450 samples: 50 per digit, matching trainer.mjs)
console.log('Evaluating standard 450-sample test set (50 per digit)...');
const stdPer = new Array(9).fill(0);
const stdSet = [];
for (let n = 0; n < nTotal && stdSet.length < 450; n++) {
  const d = testBuf[n * rec] - 1;
  if (stdPer[d] < 50) {
    stdPer[d]++;
    stdSet.push({ img: img(testBuf, n), truth: d });
  }
}

let stdOk = 0;
const stdPerOk = new Array(NL).fill(0);
const stdConfusion = Array.from({ length: NL }, () => new Array(NL).fill(0));

const t0 = Date.now();
stdSet.forEach((q, i) => {
  const seen = R.look(q.img, { seed: 1000 + i });
  const d = R.decide(seen.drive);
  stdConfusion[q.truth][d.answer]++;
  if (d.answer === q.truth) {
    stdOk++;
    stdPerOk[q.truth]++;
  }
});
const stdTime = Date.now() - t0;
const stdAccuracy = stdOk / stdSet.length;
console.log(`Standard 450-sample test accuracy: ${(stdAccuracy * 100).toFixed(2)}% (${stdOk}/${stdSet.length}) in ${stdTime} ms`);
console.log('Per digit:', R.labels.map((l, i) => `${l}:${(stdPerOk[i] / 50 * 100).toFixed(1)}%`).join(' '));

// 2. Evaluate on full test set (9020 samples)
console.log(`Evaluating full ${nTotal}-sample test set...`);
let fullOk = 0;
const fullPer = new Array(NL).fill(0);
const fullPerOk = new Array(NL).fill(0);
const fullConfusion = Array.from({ length: NL }, () => new Array(NL).fill(0));

const t1 = Date.now();
for (let n = 0; n < nTotal; n++) {
  const truth = testBuf[n * rec] - 1;
  const image = img(testBuf, n);
  const seen = R.look(image, { seed: 2000 + n });
  const d = R.decide(seen.drive);
  fullPer[truth]++;
  fullConfusion[truth][d.answer]++;
  if (d.answer === truth) {
    fullOk++;
    fullPerOk[truth]++;
  }
  if ((n + 1) % 1000 === 0 || n + 1 === nTotal) {
    const elapsed = Date.now() - t1;
    console.log(`  Processed ${n + 1}/${nTotal} (${((n + 1) / nTotal * 100).toFixed(1)}%) - running acc: ${(fullOk / (n + 1) * 100).toFixed(2)}% [${elapsed} ms]`);
  }
}
const fullTime = Date.now() - t1;
const fullAccuracy = fullOk / nTotal;
console.log(`Full test accuracy: ${(fullAccuracy * 100).toFixed(2)}% (${fullOk}/${nTotal}) in ${(fullTime / 1000).toFixed(1)} s`);

const metrics = {
  model: 'brain-110217.bin.gz',
  course: 'suji',
  labels: R.labels,
  standard_test: {
    samples: stdSet.length,
    accuracy: stdAccuracy,
    correct: stdOk,
    time_ms: stdTime,
    per_class_accuracy: Object.fromEntries(R.labels.map((l, i) => [l, stdPerOk[i] / stdPer[i]])),
    confusion_matrix: stdConfusion,
  },
  full_test: {
    samples: nTotal,
    accuracy: fullAccuracy,
    correct: fullOk,
    time_ms: fullTime,
    per_class_accuracy: Object.fromEntries(R.labels.map((l, i) => [l, fullPerOk[i] / fullPer[i]])),
    confusion_matrix: fullConfusion,
  },
};

await writeFile(new URL('metrics.json', RESULTS_DIR), JSON.stringify(metrics, null, 2));
console.log('Saved metrics to results/mnist_baseline/metrics.json');
