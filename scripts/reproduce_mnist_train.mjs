import { readFile, writeFile, mkdir, rename } from 'node:fs/promises';
import { gzipSync, gunzipSync } from 'node:zlib';
import { FlyBrain } from '../flybrain/flybrain.js';
import { makeReader } from '../juku/reader.mjs';
import * as K from '../hiragana/kana.mjs';

// Parse arguments
const args = process.argv.slice(2);
let maxSteps = 1000;
let evalEvery = 200;
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--steps' && args[i + 1]) maxSteps = parseInt(args[++i], 10);
  if (args[i] === '--eval-every' && args[i + 1]) evalEvery = parseInt(args[++i], 10);
}

const ROOT = new URL('../', import.meta.url);
const STATE_DIR = new URL('results/mnist_baseline/scratch_train/', ROOT);
await mkdir(STATE_DIR, { recursive: true });

console.log(`[Scratch Training Test] Target steps: ${maxSteps}, Eval every: ${evalEvery}`);
console.log('Initializing naive FlyBrain reader for suji...');
const R = await makeReader({ FlyBrain, base: new URL('flybrain/', ROOT), course: 'suji' });
const NL = R.labels.length;

// Load MNIST data
const load = async (f) => new Uint8Array(gunzipSync(await readFile(new URL('juku/data/' + f, ROOT))));
const train = await load('mnist12_train.bin.gz');
const test = await load('mnist12_test.bin.gz');
const rec = 145;
const nTrain = train.length / rec;
const img = (buf, n) => Float32Array.from(buf.subarray(n * rec + 1, (n + 1) * rec), (v) => v / 255);

// Shuffled order
const order = [...Array(nTrain).keys()];
const rnd = K.mulberry32(20260913);
for (let i = order.length - 1; i > 0; i--) {
  const j = Math.floor(rnd() * (i + 1));
  [order[i], order[j]] = [order[j], order[i]];
}

// 100 test samples (matching 10-11 per digit for quick evaluation)
const testSet = [];
const per = new Array(NL).fill(0);
for (let n = 0; n * rec < test.length && testSet.length < 180; n++) {
  const d = test[n * rec] - 1;
  if (per[d] < 20) {
    per[d]++;
    testSet.push({ img: img(test, n), truth: d });
  }
}

function sitTest() {
  let ok = 0;
  testSet.forEach((q, i) => {
    const seen = R.look(q.img, { seed: 5000 + i });
    const d = R.decide(seen.drive);
    if (d.answer === q.truth) ok++;
  });
  return ok / testSet.length;
}

// Initial naive evaluation
const initialAcc = sitTest();
console.log(`Step 0 (Naive): Test Accuracy = ${(initialAcc * 100).toFixed(1)}% (Random guess baseline is ~${(100 / NL).toFixed(1)}%)`);

// Training loop
const logEntries = [{ step: 0, test_acc: initialAcc, timestamp: Date.now() }];
let cursor = 0;
let correctTrain = 0;
const t0 = Date.now();

for (let step = 1; step <= maxSteps; step++) {
  const n = order[cursor % nTrain];
  cursor++;
  const image = img(train, n);
  const truth = train[n * rec] - 1;

  const result = R.practise(image, truth);
  if (result.ok) correctTrain++;

  if (step % evalEvery === 0 || step === maxSteps) {
    const testAcc = sitTest();
    const trainAcc = correctTrain / step;
    const elapsed = Date.now() - t0;
    const rate = (step / (elapsed / 1000)).toFixed(1);
    console.log(`Step ${step}/${maxSteps}: Test Acc = ${(testAcc * 100).toFixed(1)}%, Train Acc = ${(trainAcc * 100).toFixed(1)}% (${rate} steps/s)`);
    logEntries.push({ step, test_acc: testAcc, train_acc: trainAcc, timestamp: Date.now() });
  }
}

// Save learned weights
const gains = R.exportGains();
const weightName = `brain-scratch-${maxSteps}.bin.gz`;
await writeFile(new URL(weightName, STATE_DIR), gzipSync(Buffer.from(gains.buffer), { level: 6 }));
await writeFile(new URL('scratch_log.json', STATE_DIR), JSON.stringify(logEntries, null, 2));

console.log(`[Scratch Training Test] Completed! Saved weights to ${weightName}`);
