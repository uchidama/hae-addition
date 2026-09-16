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

const mode = getArg('--mode', 'all-pairs'); // 'all-pairs' | 'held-out' | 'commutativity'
const maxSteps = parseInt(getArg('--steps', '5000'), 10);
const evalEvery = parseInt(getArg('--eval-every', '500'), 10);
const testPerPair = parseInt(getArg('--test-per-pair', '3'), 10);
const defaultOutDir = mode === 'held-out' ? 'results/addition_heldout'
                    : mode === 'commutativity' ? 'results/addition_commutativity'
                    : 'results/addition_all_pairs';
const outDirName = getArg('--out-dir', defaultOutDir);

const ROOT = new URL('../', import.meta.url);
const OUT_DIR = new URL(`${outDirName}/`, ROOT);
await mkdir(OUT_DIR, { recursive: true });

console.log(`=== FlyBrain Addition Training (${mode}) ===`);
console.log(`Steps: ${maxSteps}, Eval Every: ${evalEvery}, Test Per Pair: ${testPerPair}`);
console.log(`Output Directory: ${outDirName}`);

// 1. Initialize dataset & reader
console.log('Loading dataset...');
const ds = await AdditionDataset.create();

let splitInfo = {};
let allowedTrainPairs = null;

if (mode === 'held-out') {
  const held = ds.getHeldOutSplit(20, 42);
  allowedTrainPairs = held.seenPairs;
  splitInfo = {
    seenCount: held.seenPairs.size,
    unseenCount: held.unseenPairs.size,
    seenPairs: [...held.seenPairs].sort(),
    unseenPairs: [...held.unseenPairs].sort(),
  };
  console.log(`Held-out split: ${held.seenPairs.size} Seen pairs, ${held.unseenPairs.size} Unseen pairs`);
} else if (mode === 'commutativity') {
  const comm = ds.getCommutativeSplit(20, 42);
  allowedTrainPairs = comm.seenPairs;
  splitInfo = {
    seenCount: comm.seenPairs.size,
    heldOutRevCount: comm.heldOutReversed.size,
    forwardTrainedCount: comm.forwardTrained.size,
    seenPairs: [...comm.seenPairs].sort(),
    heldOutReversed: [...comm.heldOutReversed].sort(),
    forwardTrained: [...comm.forwardTrained].sort(),
  };
  console.log(`Commutative split: ${comm.seenPairs.size} Seen, ${comm.heldOutReversed.size} Held-out reversed, ${comm.forwardTrained.size} Forward trained`);
} else {
  allowedTrainPairs = new Set(ds.allPairs.map((p) => p.key));
  splitInfo = { mode: 'all-pairs', totalPairs: 100 };
}

console.log('Initializing AdditionReader (Split-PN, 19 classes)...');
const R = await makeAdditionReader({ FlyBrain, base: new URL('flybrain/', ROOT) });
const NL = R.labels.length; // 19

// 2. Prepare test set (always evaluate all 100 pairs)
const testSet = ds.getFixedTestSet(testPerPair, null, 777);
console.log(`Prepared fixed test set: ${testSet.length} samples (${testPerPair} per pair, 100 pairs)`);

function evaluate(stepNumber) {
  let ok = 0;
  let seenOk = 0, seenTotal = 0;
  let unseenOk = 0, unseenTotal = 0;
  let fwdOk = 0, fwdTotal = 0;
  let revOk = 0, revTotal = 0;

  const perClassTotal = new Array(NL).fill(0);
  const perClassOk = new Array(NL).fill(0);
  const pairTotal = {};
  const pairOk = {};
  const confusion = Array.from({ length: NL }, () => new Array(NL).fill(0));

  for (const q of testSet) {
    const seen = R.look(q.leftImage, q.rightImage);
    const d = R.decide(seen.drive);
    const isCorrect = d.answer === q.target;

    perClassTotal[q.target]++;
    pairTotal[q.pairKey] = (pairTotal[q.pairKey] || 0) + 1;
    confusion[q.target][d.answer]++;

    if (isCorrect) {
      ok++;
      perClassOk[q.target]++;
      pairOk[q.pairKey] = (pairOk[q.pairKey] || 0) + 1;
    }

    // Split breakdowns
    if (mode === 'held-out') {
      if (splitInfo.unseenPairs.includes(q.pairKey)) {
        unseenTotal++;
        if (isCorrect) unseenOk++;
      } else {
        seenTotal++;
        if (isCorrect) seenOk++;
      }
    } else if (mode === 'commutativity') {
      if (splitInfo.heldOutReversed.includes(q.pairKey)) {
        revTotal++;
        if (isCorrect) revOk++;
      }
      if (splitInfo.forwardTrained.includes(q.pairKey)) {
        fwdTotal++;
        if (isCorrect) fwdOk++;
      }
      if (splitInfo.seenPairs.includes(q.pairKey)) {
        seenTotal++;
        if (isCorrect) seenOk++;
      }
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

  const result = {
    step: stepNumber,
    accuracy,
    correct: ok,
    total: testSet.length,
    perClassAcc,
    pairAcc,
    confusion,
  };

  if (mode === 'held-out') {
    result.seenAccuracy = seenTotal ? seenOk / seenTotal : 0;
    result.unseenAccuracy = unseenTotal ? unseenOk / unseenTotal : 0;
    result.seenCorrect = seenOk;
    result.seenTotal = seenTotal;
    result.unseenCorrect = unseenOk;
    result.unseenTotal = unseenTotal;
  } else if (mode === 'commutativity') {
    result.seenAccuracy = seenTotal ? seenOk / seenTotal : 0;
    result.commutativeSourceAccuracy = fwdTotal ? fwdOk / fwdTotal : 0;
    result.commutativeGeneralizationAccuracy = revTotal ? revOk / revTotal : 0;
    result.fwdCorrect = fwdOk;
    result.fwdTotal = fwdTotal;
    result.revCorrect = revOk;
    result.revTotal = revTotal;
  }

  return result;
}

// 3. Initial evaluation (Step 0)
console.log('Evaluating naive model (Step 0)...');
const tEval0 = Date.now();
const eval0 = evaluate(0);
console.log(`Step 0 (Naive): Test Acc = ${(eval0.accuracy * 100).toFixed(2)}% in ${Date.now() - tEval0} ms`);

const config = {
  experiment: `addition_${mode}`,
  mode,
  maxSteps,
  evalEvery,
  testPerPair,
  testSetSize: testSet.length,
  classes: NL,
  labels: R.labels,
  encoding: 'Split-PN (342 left ALPN, 343 right ALPN)',
  plasticity: 'Error-driven dopamine (eta=6e-5, gainMin=0.05, gainMax=2.0)',
  splitInfo,
  startTime: new Date().toISOString(),
};
await writeFile(new URL('config.json', OUT_DIR), JSON.stringify(config, null, 2));

const logFile = new URL('log.jsonl', OUT_DIR);
await writeFile(logFile, JSON.stringify({
  step: 0,
  test_acc: eval0.accuracy,
  seen_acc: eval0.seenAccuracy ?? null,
  unseen_acc: eval0.unseenAccuracy ?? null,
  comm_source_acc: eval0.commutativeSourceAccuracy ?? null,
  comm_gen_acc: eval0.commutativeGeneralizationAccuracy ?? null,
  run_acc: null,
  timestamp: Date.now(),
}) + '\n');

// 4. Training loop
const WINDOW = 200;
const recentOk = [];
let totalTrainOk = 0;
let lastEvalResult = eval0;
const tStart = Date.now();

for (let step = 1; step <= maxSteps; step++) {
  const sample = ds.sample({ mode: 'train', allowedPairs: allowedTrainPairs });
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

    let extraMsg = '';
    if (mode === 'held-out') {
      extraMsg = `[Seen: ${(evalRes.seenAccuracy * 100).toFixed(2)}%, Unseen: ${(evalRes.unseenAccuracy * 100).toFixed(2)}%]`;
    } else if (mode === 'commutativity') {
      extraMsg = `[Fwd (A+B): ${(evalRes.commutativeSourceAccuracy * 100).toFixed(2)}%, Rev (B+A): ${(evalRes.commutativeGeneralizationAccuracy * 100).toFixed(2)}%]`;
    }

    console.log(
      `Step ${step}/${maxSteps} (${(step / maxSteps * 100).toFixed(1)}%): ` +
      `Overall Test = ${(evalRes.accuracy * 100).toFixed(2)}% ${extraMsg} ` +
      `Recent Run = ${(runAcc * 100).toFixed(2)}% ` +
      `[${speed} steps/s, eval took ${Date.now() - t0} ms]`
    );

    await appendFile(logFile, JSON.stringify({
      step,
      test_acc: evalRes.accuracy,
      seen_acc: evalRes.seenAccuracy ?? null,
      unseen_acc: evalRes.unseenAccuracy ?? null,
      comm_source_acc: evalRes.commutativeSourceAccuracy ?? null,
      comm_gen_acc: evalRes.commutativeGeneralizationAccuracy ?? null,
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
  experiment: `addition_${mode}`,
  mode,
  final_step: maxSteps,
  final_test_accuracy: lastEvalResult.accuracy,
  final_test_correct: lastEvalResult.correct,
  total_test_samples: lastEvalResult.total,
  duration_seconds: durationSec,
  per_sum_accuracy: lastEvalResult.perClassAcc,
  pair_accuracies: lastEvalResult.pairAcc,
  confusion_matrix: lastEvalResult.confusion,
};

if (mode === 'held-out') {
  finalMetrics.seen_accuracy = lastEvalResult.seenAccuracy;
  finalMetrics.unseen_accuracy = lastEvalResult.unseenAccuracy;
  finalMetrics.seen_correct = lastEvalResult.seenCorrect;
  finalMetrics.seen_total = lastEvalResult.seenTotal;
  finalMetrics.unseen_correct = lastEvalResult.unseenCorrect;
  finalMetrics.unseen_total = lastEvalResult.unseenTotal;
  finalMetrics.unseen_pairs = splitInfo.unseenPairs;
} else if (mode === 'commutativity') {
  finalMetrics.seen_accuracy = lastEvalResult.seenAccuracy;
  finalMetrics.commutative_source_accuracy = lastEvalResult.commutativeSourceAccuracy;
  finalMetrics.commutative_generalization_accuracy = lastEvalResult.commutativeGeneralizationAccuracy;
  finalMetrics.forward_trained_pairs = splitInfo.forwardTrained;
  finalMetrics.held_out_reversed_pairs = splitInfo.heldOutReversed;
}

await writeFile(new URL('metrics.json', OUT_DIR), JSON.stringify(finalMetrics, null, 2));
console.log(`Saved final results to ${outDirName}/metrics.json`);
