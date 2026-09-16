import { readFile } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';
import * as K from '../hiragana/kana.mjs';

const ROOT = new URL('../', import.meta.url);

export async function loadMnist0to9(file) {
  const buf = new Uint8Array(gunzipSync(await readFile(new URL(`juku/data/${file}`, ROOT))));
  const rec = 145;
  const num = buf.length / rec;
  const byDigit = Array.from({ length: 10 }, () => []);

  for (let i = 0; i < num; i++) {
    const label = buf[i * rec];
    byDigit[label].push(i);
  }

  const getImage = (idx) => Float32Array.from(buf.subarray(idx * rec + 1, (idx + 1) * rec), (v) => v / 255);

  return { buf, num, byDigit, getImage };
}

export class AdditionDataset {
  static async create() {
    const [train, test] = await Promise.all([
      loadMnist0to9('mnist12_0to9_train.bin.gz'),
      loadMnist0to9('mnist12_0to9_test.bin.gz'),
    ]);
    return new AdditionDataset(train, test);
  }

  constructor(train, test) {
    this.train = train;
    this.test = test;
    this.allPairs = [];
    for (let l = 0; l <= 9; l++) {
      for (let r = 0; r <= 9; r++) {
        this.allPairs.push({ left: l, right: r, key: `${l}+${r}`, target: l + r });
      }
    }
  }

  /**
   * Sample a single paired addition instance.
   * @param {object} [opts]
   * @param {'train'|'test'} [opts.mode='train']
   * @param {Array<string>|Set<string>} [opts.allowedPairs]  e.g. Set(["0+0", "1+2", ...])
   * @param {() => number} [opts.rng]
   */
  sample(opts = {}) {
    const mode = opts.mode || 'train';
    const split = mode === 'train' ? this.train : this.test;
    const rng = opts.rng || Math.random;

    let pair;
    if (opts.allowedPairs) {
      const allowed = Array.isArray(opts.allowedPairs) ? opts.allowedPairs : [...opts.allowedPairs];
      const chosenKey = allowed[Math.floor(rng() * allowed.length)];
      const [l, r] = chosenKey.split('+').map(Number);
      pair = { left: l, right: r, key: chosenKey, target: l + r };
    } else {
      pair = this.allPairs[Math.floor(rng() * this.allPairs.length)];
    }

    const leftPool = split.byDigit[pair.left];
    const rightPool = split.byDigit[pair.right];
    const leftIdx = leftPool[Math.floor(rng() * leftPool.length)];
    const rightIdx = rightPool[Math.floor(rng() * rightPool.length)];

    return {
      leftImage: split.getImage(leftIdx),
      rightImage: split.getImage(rightIdx),
      leftDigit: pair.left,
      rightDigit: pair.right,
      target: pair.target,
      pairKey: pair.key,
      mode,
    };
  }

  /**
   * Generate a deterministic fixed test set.
   * @param {number} [samplesPerPair=5]
   * @param {Array<string>|Set<string>} [allowedPairs]
   * @param {number} [seed=777]
   */
  getFixedTestSet(samplesPerPair = 5, allowedPairs = null, seed = 777) {
    const rng = K.mulberry32(seed);
    const pairs = allowedPairs
      ? (Array.isArray(allowedPairs) ? allowedPairs : [...allowedPairs]).map((k) => {
          const [l, r] = k.split('+').map(Number);
          return { left: l, right: r, key: k, target: l + r };
        })
      : this.allPairs;

    const out = [];
    for (const p of pairs) {
      const leftPool = this.test.byDigit[p.left];
      const rightPool = this.test.byDigit[p.right];
      for (let i = 0; i < samplesPerPair; i++) {
        const leftIdx = leftPool[Math.floor(rng() * leftPool.length)];
        const rightIdx = rightPool[Math.floor(rng() * rightPool.length)];
        out.push({
          leftImage: this.test.getImage(leftIdx),
          rightImage: this.test.getImage(rightIdx),
          leftDigit: p.left,
          rightDigit: p.right,
          target: p.target,
          pairKey: p.key,
          mode: 'test',
        });
      }
    }
    return out;
  }
}
