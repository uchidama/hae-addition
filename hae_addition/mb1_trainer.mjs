import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { gzipSync, gunzipSync } from 'node:zlib';
import { FlyBrain } from '../flybrain/flybrain.js';
import { makeReader } from '../juku/reader.mjs';
import * as K from '../hiragana/kana.mjs';

// Parse arguments
const args = process.argv.slice(2);
let maxSteps = 3000;
let evalEvery = 500;
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--steps' && args[i + 1]) maxSteps = parseInt(args[++i], 10);
  if (args[i] === '--eval-every' && args[i + 1]) evalEvery = parseInt(args[++i], 10);
}

const ROOT = new URL('../', import.meta.url);
const OUT_DIR = new URL('results/mb1_digits/', ROOT);
await mkdir(OUT_DIR, { recursive: true });

console.log(`=== Training MB1 (10-Class Single-Digit Recognizer 0-9) ===`);
console.log(`Target steps: ${maxSteps}, Eval every: ${evalEvery}`);

const R = await makeReader({ FlyBrain, base: new URL('flybrain/', ROOT), course: 'suji10' });
const NL = R.labels.length;

// Load MNIST 0-9 data
const load = async (f) => new Uint8Array(gunzipSync(await readFile(new URL('juku/data/' + f, ROOT))));
const train = await load('mnist12_0to9_train.bin.gz');
const test = await load('mnist12_0to9_test.bin.gz');
const rec = 145;
const nTrain = train.length / rec;
const nTest = test.length / rec;
const img = (buf, n) => Float32Array.from(buf.subarray(n * rec + 1, (n + 1) * rec), (v) => v / 255);

console.log(`Loaded ${nTrain} train samples, ${nTest} test samples.`);

// Shuffled train order
const order = [...Array(nTrain).keys()];
const rnd = K.mulberry32(20260917);
for (let i = order.length - 1; i > 0; i--) {
  const j = Math.floor(rnd() * (i + 1));
  [order[i], order[j]] = [order[j], order[i]];
}

// Build balanced test set (40 samples per digit = 400 test samples)
const testSet = [];
const per = new Array(NL).fill(0);
for (let n = 0; n < nTest && testSet.length < 400; n++) {
  const d = test[n * rec];
  if (per[d] < 40) {
    per[d]++;
    testSet.push({ img: img(test, n), truth: d });
  }
}

function evaluate(step) {
  let ok = 0;
  const perClassOk = new Array(NL).fill(0);
  const perClassTotal = new Array(NL).fill(0);

  testSet.forEach((q, i) => {
    const seen = R.look(q.img, { seed: 8000 + i });
    const d = R.decide(seen.drive);
    perClassTotal[q.truth]++;
    if (d.answer === q.truth) {
      ok++;
      perClassOk[q.truth]++;
    }
  });

  const accuracy = ok / testSet.length;
  return { accuracy, ok, total: testSet.length, perClassOk, perClassTotal };
}

// Initial evaluation
console.log('Evaluating initial naive brain (Step 0)...');
const eval0 = evaluate(0);
console.log(`Step 0: Test Acc = ${(eval0.accuracy * 100).toFixed(2)}% (${eval0.ok}/${eval0.total})`);

const WINDOW = 200;
const recent = [];
const log = [{ step: 0, test_acc: eval0.accuracy, timestamp: Date.now() }];
const tStart = Date.now();

for (let step = 1; step <= maxSteps; step++) {
  const idx = order[(step - 1) % nTrain];
  const truth = train[idx * rec];
  const image = img(train, idx);

  const res = R.practise(image, truth);
  recent.push(res.ok ? 1 : 0);
  if (recent.length > WINDOW) recent.shift();

  if (step % evalEvery === 0 || step === maxSteps) {
    const t0 = Date.now();
    const ev = evaluate(step);
    const runAcc = recent.reduce((a, b) => a + b, 0) / recent.length;
    const speed = (step / ((Date.now() - tStart) / 1000)).toFixed(1);

    console.log(
      `Step ${step}/${maxSteps} (${(step / maxSteps * 100).toFixed(0)}%): ` +
      `Test Acc = ${(ev.accuracy * 100).toFixed(2)}% ` +
      `Recent Run = ${(runAcc * 100).toFixed(2)}% ` +
      `[${speed} steps/s, eval took ${Date.now() - t0}ms]`
    );

    log.push({
      step,
      test_acc: ev.accuracy,
      run_acc: runAcc,
      timestamp: Date.now(),
      per_digit: ev.perClassOk.map((c, i) => Math.round(100 * c / ev.perClassTotal[i])),
    });

    // Save checkpoint
    const weights = R.exportGains();
    const buf = gzipSync(Buffer.from(weights.buffer, weights.byteOffset, weights.byteLength));
    await writeFile(new URL(`brain-mb1-${step}.bin.gz`, OUT_DIR), buf);
  }
}

// Save final model & log
const finalWeights = R.exportGains();
const finalBuf = gzipSync(Buffer.from(finalWeights.buffer, finalWeights.byteOffset, finalWeights.byteLength));
await writeFile(new URL('brain-mb1-final.bin.gz', OUT_DIR), finalBuf);
await writeFile(new URL('training_log.json', OUT_DIR), JSON.stringify(log, null, 2));

console.log(`Training complete! Final model saved to results/mb1_digits/brain-mb1-final.bin.gz`);
