import { readFile, writeFile, mkdir, rename, appendFile } from 'node:fs/promises';
import { gzipSync } from 'node:zlib';
import { FlyBrain } from '../flybrain/flybrain.js';
import { makeAdditionReader } from './addition_reader.mjs';
import { AdditionDataset } from './dataset.mjs';

// Parse CLI arguments
const args = process.argv.slice(2);
function getArg(flag, defaultValue) {
  const idx = args.indexOf(flag);
  return idx !== -1 && args[idx + 1] ? args[idx + 1] : defaultValue;
}

const mode = getArg('--mode', 'all-pairs');
const maxSteps = parseInt(getArg('--steps', '10000'), 10);
const evalEvery = parseInt(getArg('--eval-every', '1000'), 10);
const testPerPair = parseInt(getArg('--test-per-pair', '5'), 10);
const outDirName = getArg('--out-dir', 'results/addition_all_pairs');

const ROOT = new URL('../', import.meta.url);
const OUT_DIR = new URL(`${outDirName}/`, ROOT);
await mkdir(OUT_DIR, { recursive: true });

console.log(`=== FlyBrain Addition Training ===`);
console.log(`Mode: ${mode}, Steps: ${maxSteps}, Eval Every: ${evalEvery}, Test Per Pair: ${testPerPair}`);
console.log(`Output Directory: ${outDirName}`);

// 1. Initialize dataset & reader
console.log('Loading dataset...');
const ds = await AdditionDataset.create();

console.log('Initializing AdditionReader (Split-PN, 19 classes)...');
const R = await makeAdditionReader({ FlyBrain, base: new URL('flybrain/', ROOT) });
const NL = R.labels.length; // 19

// 2. Prepare test set
const testSet = ds.getFixedTestSet(testPerPair, null, 777);
console.log(`Prepared fixed test set: ${testSet.length} samples (${testPerPair} per pair, 100 pairs)`);

function evaluate(stepNumber) {
  let ok = 0;
  const perClassTotal = new Array(NL).fill(0);
  const perClassOk = new Array(NL).fill(0);
  const pairTotal = {};
  const pairOk = {};
  const confusion = Array.from({ length: NL }, () => new Array(NL).fill(0));

  for (const q of testSet) {
    const seen = R.look(q.leftImage, q.rightImage);
    const d = R.decide(seen.drive);

    perClassTotal[q.target]++;
    pairTotal[q.pairKey] = (pairTotal[q.pairKey] || 0) + 1;
    confusion[q.target][d.answer]++;

    if (d.answer === q.target) {
      ok++;
      perClassOk[q.target]++;
      pairOk[q.pairKey] = (pairOk[q.pairKey] || 0) + 1;
    }
  }

  const accuracy = ok / testSet.length;
  const perClassAcc = {};
  for (let c = 0; c < NL; c++) {
    perClassAcc[c] = perClassTotal[c] ? perClassOk[c] / perClassTotal[c] : 0;
  }
  const pairAcc = {};
  for (const k of Object.keys(pairTotal)) {
    pairAcc[k] = (pairOk[k] || 0) / pairTotal[k];
  }

  return {
    step: stepNumber,
    accuracy,
    correct: ok,
    total: testSet.length,
    perClassAcc,
    pairAcc,
    confusion,
  };
}

// 3. Evaluate initial naive model (Step 0)
console.log('Evaluating naive model (Step 0)...');
const tEval0 = Date.now();
const eval0 = evaluate(0);
console.log(`Step 0 (Naive): Test Accuracy = ${(eval0.accuracy * 100).toFixed(2)}% (Random baseline: ~${(100 / NL).toFixed(2)}%) in ${Date.now() - tEval0} ms`);

const config = {
  experiment: 'addition_all_pairs',
  mode,
  maxSteps,
  evalEvery,
  testPerPair,
  testSetSize: testSet.length,
  classes: NL,
  labels: R.labels,
  encoding: 'Split-PN (342 left ALPN, 343 right ALPN)',
  plasticity: 'Error-driven dopamine (eta=6e-5, gainMin=0.05, gainMax=2.0)',
  startTime: new Date().toISOString(),
};
await writeFile(new URL('config.json', OUT_DIR), JSON.stringify(config, null, 2));

const logFile = new URL('log.jsonl', OUT_DIR);
await writeFile(logFile, JSON.stringify({
  step: 0,
  test_acc: eval0.accuracy,
  run_acc: null,
  timestamp: Date.now(),
}) + '\n');

// 4. Training Loop
const WINDOW = 200;
const recentOk = [];
let totalTrainOk = 0;
let lastEvalResult = eval0;
const tStart = Date.now();

for (let step = 1; step <= maxSteps; step++) {
  const sample = ds.sample({ mode: 'train' });
  const result = R.practise(sample.leftImage, sample.rightImage, sample.target);

  if (result.ok) totalTrainOk++;
  recentOk.push(result.ok ? 1 : 0);
  if (recentOk.length > WINDOW) recentOk.shift();

  if (step % evalEvery === 0 || step === maxSteps) {
    const t0 = Date.now();
    const evalRes = evaluate(step);
    lastEvalResult = evalRes;
    const runAcc = recentOk.reduce((a, b) => a + b, 0) / recentOk.length;
    const elapsed = Date.now() - tStart;
    const speed = (step / (elapsed / 1000)).toFixed(1);

    console.log(
      `Step ${step}/${maxSteps} (${(step / maxSteps * 100).toFixed(1)}%): ` +
      `Test Acc = ${(evalRes.accuracy * 100).toFixed(2)}%, ` +
      `Recent Run Acc = ${(runAcc * 100).toFixed(2)}% ` +
      `[${speed} steps/s, eval took ${Date.now() - t0} ms]`
    );

    await appendFile(logFile, JSON.stringify({
      step,
      test_acc: evalRes.accuracy,
      run_acc: runAcc,
      timestamp: Date.now(),
    }) + '\n');

    // Save weights
    const gains = R.exportGains();
    const weightPath = new URL(`brain-addition-${step}.bin.gz`, OUT_DIR);
    await writeFile(weightPath, gzipSync(Buffer.from(gains.buffer), { level: 6 }));
  }
}

// 5. Finalize and Save Final Metrics
const durationSec = (Date.now() - tStart) / 1000;
console.log(`Training completed in ${durationSec.toFixed(1)} s!`);

const finalMetrics = {
  experiment: 'addition_all_pairs',
  final_step: maxSteps,
  final_test_accuracy: lastEvalResult.accuracy,
  final_test_correct: lastEvalResult.correct,
  total_test_samples: lastEvalResult.total,
  duration_seconds: durationSec,
  per_sum_accuracy: lastEvalResult.perClassAcc,
  pair_accuracies: lastEvalResult.pairAcc,
  confusion_matrix: lastEvalResult.confusion,
};

await writeFile(new URL('metrics.json', OUT_DIR), JSON.stringify(finalMetrics, null, 2));
console.log(`Saved final results to ${outDirName}/metrics.json`);
