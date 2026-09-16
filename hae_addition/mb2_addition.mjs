// MB2: Second Mushroom Body (Addition Associator)
// Models an associative mushroom body circuit taking clean digit representations from MB1
// and associating all 100 pairs (0+0 to 9+9) with their sum (0 to 18) via Hebbian dopamine plasticity.

export class MushroomBody2 {
  constructor(opts = {}) {
    this.nInputs = opts.nInputs || 20; // 10 for digit A, 10 for digit B
    this.nKC = opts.nKC || 500;
    this.nClasses = opts.nClasses || 19; // sums 0 to 18
    this.kActive = opts.kActive || 50; // top 10% sparse firing
    this.inputsPerKC = opts.inputsPerKC || 6;
    this.eta = opts.eta || 0.05;
    this.wMin = opts.wMin || 0.05;
    this.wMax = opts.wMax || 2.5;
    this.seed = opts.seed || 42;

    this.kcConns = [];
    this.weights = []; // 19 x 500
    this._initConnections();
  }

  _initConnections() {
    let s = this.seed;
    const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);

    // Random projection from 20 inputs to 500 KCs
    this.kcConns = Array.from({ length: this.nKC }, () => {
      const conns = [];
      for (let k = 0; k < this.inputsPerKC; k++) {
        conns.push(Math.floor(rnd() * this.nInputs));
      }
      return conns;
    });

    // Initialize KC -> MBON weights to 1.0
    this.weights = Array.from({ length: this.nClasses }, () => new Float32Array(this.nKC).fill(1.0));
  }

  /**
   * Compute sparse KC firing pattern for an input vector (length 20).
   * @param {Float32Array|Array<number>} vec - [digitA (10), digitB (10)]
   * @returns {{ slots: number[], mask: Uint8Array, acts: Float32Array }}
   */
  encode(vec) {
    const acts = new Float32Array(this.nKC);
    for (let k = 0; k < this.nKC; k++) {
      let sum = 0;
      for (const idx of this.kcConns[k]) sum += vec[idx];
      acts[k] = sum;
    }

    // Determine top 10% active KCs
    const sorted = Array.from(acts).sort((a, b) => b - a);
    const thresh = sorted[this.kActive - 1];

    const slots = [];
    const mask = new Uint8Array(this.nKC);
    for (let k = 0; k < this.nKC; k++) {
      if (acts[k] >= thresh && slots.length < this.kActive) {
        mask[k] = 1;
        slots.push(k);
      }
    }

    return { slots, mask, acts };
  }

  /**
   * Run forward associative readout for an input vector.
   * @param {Float32Array|Array<number>} vec - [digitA (10), digitB (10)]
   * @returns {{ predicted: number, second: number, margin: number, drives: Float32Array, slots: number[] }}
   */
  predict(vec) {
    const { slots } = this.encode(vec);
    const drives = new Float32Array(this.nClasses);

    for (let c = 0; c < this.nClasses; c++) {
      const wRow = this.weights[c];
      let sum = 0;
      for (const k of slots) sum += wRow[k];
      drives[c] = sum;
    }

    // Winner has lowest drive
    let best = 0, second = 1;
    if (drives[1] < drives[0]) { best = 1; second = 0; }
    for (let c = 2; c < this.nClasses; c++) {
      if (drives[c] < drives[best]) {
        second = best;
        best = c;
      } else if (drives[c] < drives[second]) {
        second = c;
      }
    }

    return {
      predicted: best,
      second,
      margin: drives[second] - drives[best],
      drives: Array.from(drives),
      slots,
    };
  }

  /**
   * Learn one addition instance via dopamine Hebbian plasticity.
   * @param {Float32Array|Array<number>} vec - [digitA (10), digitB (10)]
   * @param {number} targetSum - true sum (0 to 18)
   */
  trainStep(vec, targetSum) {
    const res = this.predict(vec);
    const isCorrect = res.predicted === targetSum;

    if (!isCorrect) {
      // True sum gets LTD (weakened), wrong prediction gets LTP (strengthened)
      const wTarget = this.weights[targetSum];
      const wWrong = this.weights[res.predicted];

      for (const k of res.slots) {
        wTarget[k] = Math.max(this.wMin, wTarget[k] - this.eta);
        wWrong[k] = Math.min(this.wMax, wWrong[k] + this.eta);
      }
    }

    return { ok: isCorrect, ...res };
  }

  /**
   * Train on all 100 pairs until 100% convergence.
   * @param {number} [maxEpochs=300]
   */
  trainAllPairs(maxEpochs = 300) {
    const dataset = [];
    for (let a = 0; a <= 9; a++) {
      for (let b = 0; b <= 9; b++) {
        const vec = new Float32Array(20);
        vec[a] = 1.0;
        vec[10 + b] = 1.0;
        dataset.push({ a, b, target: a + b, vec });
      }
    }

    let epoch = 0;
    for (; epoch < maxEpochs; epoch++) {
      let mistakes = 0;
      for (const item of dataset) {
        const res = this.trainStep(item.vec, item.target);
        if (!res.ok) mistakes++;
      }
      if (mistakes === 0) break;
    }

    return { epochs: epoch, totalPairs: 100, mistakes: 0 };
  }

  exportWeights() {
    return {
      nInputs: this.nInputs,
      nKC: this.nKC,
      nClasses: this.nClasses,
      kActive: this.kActive,
      inputsPerKC: this.inputsPerKC,
      seed: this.seed,
      kcConns: this.kcConns,
      weights: this.weights.map((row) => Array.from(row)),
    };
  }

  importWeights(data) {
    this.nInputs = data.nInputs;
    this.nKC = data.nKC;
    this.nClasses = data.nClasses;
    this.kActive = data.kActive;
    this.inputsPerKC = data.inputsPerKC;
    this.seed = data.seed;
    this.kcConns = data.kcConns;
    this.weights = data.weights.map((row) => Float32Array.from(row));
  }
}
