// Dual-Mushroom Body Addition Reader
// Integrates MB1 (10-Class Single-Digit Recognizer) with MB2 (Addition Associator)

import { FlyBrain } from '../flybrain/flybrain.js';
import { makeReader } from '../juku/reader.mjs';
import { MushroomBody2 } from './mb2_addition.mjs';

async function readBytes(url) {
  if (url.protocol === 'file:') {
    const { readFile } = await import('node:fs/promises');
    return new Uint8Array(await readFile(url));
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> HTTP ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

async function gunzipBytes(bytes) {
  const s = new Response(bytes).body.pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(s).arrayBuffer());
}

/**
 * Creates and initializes the Dual-Mushroom Body pipeline.
 * @param {object} opts
 * @param {URL} opts.base - base URL for flybrain/
 * @param {URL} opts.mb1WeightsUrl - URL to brain-mb1-final.bin.gz
 * @param {URL} opts.mb2WeightsUrl - URL to mb2_weights.json
 * @param {(p: object) => void} [opts.onProgress]
 */
export async function makeDualMBReader(opts = {}) {
  const base = opts.base || new URL('../flybrain/', import.meta.url);
  const onProgress = opts.onProgress || (() => {});

  // 1. Initialize MB1 (Visual Digit Recognizer, 10 classes 0-9)
  onProgress({ phase: 'mb1_init', message: '第1キノコ体（視覚認識 LIFモデル）を初期化中...' });
  const FB = opts.FlyBrain || FlyBrain;
  const mb1 = await makeReader({
    FlyBrain: FB,
    base,
    course: 'suji10',
    onProgress: (p) => onProgress({ ...p, phase: 'mb1_' + p.phase }),
  });

  if (opts.mb1WeightsUrl) {
    onProgress({ phase: 'mb1_weights', message: '第1キノコ体の学習済み重みを読み込み中...' });
    try {
      const raw = await readBytes(opts.mb1WeightsUrl);
      const buf = await gunzipBytes(raw);
      const gains = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
      mb1.importGains(gains);
    } catch (err) {
      console.warn('Could not load MB1 weights:', err);
    }
  }

  // 2. Initialize MB2 (Addition Associative Mushroom Body)
  onProgress({ phase: 'mb2_init', message: '第2キノコ体（足し算連合 連想記憶エンジン）を初期化中...' });
  const mb2 = new MushroomBody2();

  if (opts.mb2WeightsUrl) {
    try {
      const bytes = await readBytes(opts.mb2WeightsUrl);
      const json = JSON.parse(new TextDecoder().decode(bytes));
      mb2.importWeights(json);
    } catch {
      // Fallback: train on the fly (takes ~20ms)
      mb2.trainAllPairs();
    }
  } else {
    mb2.trainAllPairs();
  }

  onProgress({ phase: 'ready', message: '2段キノコ体モデルの準備が完了しました！' });

  /**
   * Run the end-to-end Dual-Mushroom Body addition inference.
   * @param {Float32Array} imgL - Left digit image (144 pixels)
   * @param {Float32Array} imgR - Right digit image (144 pixels)
   * @param {object} [opts]
   * @param {(r: object, t: number) => void} [opts.onChunkL]
   * @param {(r: object, t: number) => void} [opts.onChunkR]
   */
  function lookAndAdd(imgL, imgR, { onChunkL, onChunkR } = {}) {
    // Step 1: MB1 recognizes Digit A
    const seenL = mb1.look(imgL, { onChunk: onChunkL });
    const decL = mb1.decide(seenL.drive);
    const digitA = decL.answer;

    // Step 2: MB1 recognizes Digit B
    const seenR = mb1.look(imgR, { onChunk: onChunkR });
    const decR = mb1.decide(seenR.drive);
    const digitB = decR.answer;

    // Step 3: MB2 associates the pair (digitA, digitB) with the sum
    const vec = new Float32Array(20);
    vec[digitA] = 1.0;
    vec[10 + digitB] = 1.0;
    const resSum = mb2.predict(vec);
    const predictedSum = resSum.predicted;

    return {
      digitA,
      digitB,
      predictedSum,
      secondSum: resSum.second,
      marginA: decL.margin,
      marginB: decR.margin,
      marginSum: resSum.margin,
      driveA: Array.from(seenL.drive),
      driveB: Array.from(seenR.drive),
      driveSum: Array.from(resSum.drives),
      slotsA: Array.from(seenL.slots),
      slotsB: Array.from(seenR.slots),
      slotsSum: resSum.slots,
    };
  }

  return {
    mb1,
    mb2,
    lookAndAdd,
  };
}
