// Page controller for Tashizan (Addition) FlyBrain Demo
// Features sequential 3D fly writing with front-leg IK and mushroom body addition judgment

import { FlagFly } from '../suji/flag.js?v=38';
import { Sound } from '../juku/sound.js?v=1';

const worker = new Worker(new URL('./worker.js?v=2', import.meta.url), { type: 'module' });
const sound = new Sound();

// UI Elements
const statusEl = document.getElementById('status');
const stagechip = document.getElementById('stagechip');
const soundBtn = document.getElementById('soundbtn');
const flyStateText = document.getElementById('flyStateText');
const flyCanvas = document.getElementById('fly');

const cardA = document.getElementById('cardA');
const cardB = document.getElementById('cardB');
const padLeft = document.getElementById('padLeft');
const padRight = document.getElementById('padRight');
const prevLeft = document.getElementById('prevLeft');
const prevRight = document.getElementById('prevRight');
const labelA = document.getElementById('labelA');
const labelB = document.getElementById('labelB');
const clearA = document.getElementById('clearA');
const clearB = document.getElementById('clearB');

const resultBox = document.getElementById('resultBox');
const predSum = document.getElementById('predSum');
const verdictMark = document.getElementById('verdictMark');
const resultSub = document.getElementById('resultSub');

const tallyPct = document.getElementById('tallyPct');
const tallyFrac = document.getElementById('tallyFrac');
const tallySub = document.getElementById('tallySub');
const histEl = document.getElementById('hist');

const autobtn = document.getElementById('autobtn');
const autobadge = document.getElementById('autobadge');
const btnSample = document.getElementById('btnSample');
const btnSolve = document.getElementById('btnSolve');
const btnClear = document.getElementById('btnClear');
const driveGrid = document.getElementById('driveGrid');

// State
let fly = null;
let isWorkerReady = false;
let isFlyReady = false;
let busy = true;
let auto = true;
let quizTimer = null;
let flyWaits = 0;
let asked = 0;
let right = 0;
const hist = [];
let currentSample = null;
let kcXY = null;
let isHandMode = false;

// 1. Initialize 19 Drive Items (Sum 0 to 18)
for (let i = 0; i <= 18; i++) {
  const div = document.createElement('div');
  div.className = 'drive-item';
  div.id = `drive-${i}`;
  div.innerHTML = `
    <div class="drive-header">
      <strong>和 ${i}</strong>
      <span class="val">–</span>
    </div>
    <div class="drive-bar-bg">
      <div class="drive-bar-fill" style="width:0%"></div>
    </div>
  `;
  driveGrid.appendChild(div);
}

function updateDriveMeters(drive, predicted) {
  if (!drive) return;
  const minDrive = Math.min(...drive);
  const maxDrive = Math.max(...drive);
  const range = maxDrive - minDrive || 1;

  for (let i = 0; i <= 18; i++) {
    const item = document.getElementById(`drive-${i}`);
    if (!item) continue;
    const isWinner = i === predicted;
    item.className = `drive-item ${isWinner ? 'winner' : ''}`;
    const valEl = item.querySelector('.val');
    valEl.textContent = drive[i].toFixed(2);

    // Invert bar: lower drive = stronger activation (longer bar)
    const normalizedStrength = Math.max(0, Math.min(100, ((maxDrive - drive[i]) / range) * 100));
    const fillEl = item.querySelector('.drive-bar-fill');
    fillEl.style.width = `${normalizedStrength.toFixed(0)}%`;
  }
}

// 2. Sound Toggle & Flight Hum
const drawSoundBtn = () => {
  if (soundBtn) {
    soundBtn.textContent = sound.on ? '🔊' : '🔇';
    soundBtn.setAttribute('aria-pressed', String(sound.on));
  }
};
soundBtn?.addEventListener('click', () => {
  sound.setOn(!sound.on);
  drawSoundBtn();
});
drawSoundBtn();

(function hum() {
  requestAnimationFrame(hum);
  if (fly && sound) sound.buzz(fly.flap || 0, (fly.alt || 0) * 3);
})();

// 3. Loading Bar Animation
function loadBar(f, text) {
  const el = document.getElementById('loadbar');
  const txt = document.getElementById('loadText');
  const fill = document.getElementById('lbFill');
  if (txt) txt.textContent = text;
  if (fill) fill.style.width = `${Math.round(f * 100)}%`;
  if (f >= 1) {
    setTimeout(() => {
      el.classList.add('done');
      setTimeout(() => { el.hidden = true; }, 600);
    }, 400);
  }
}

// 4. Initialize 3D Fly (NeuroMechFly)
new FlagFly(flyCanvas).load().then((f) => {
  fly = f;
  window.__fly = f;

  // Inject stroke data for digit '0' so the fly can physically write '0'
  if (!fly.strokes) fly.strokes = {};
  fly.strokes['0'] = [[
    0, -0.45, 0.067, -0.435, 0.13, -0.39, 0.184, -0.318, 0.225, -0.225, 0.251, -0.116,
    0.26, 0, 0.251, 0.116, 0.225, 0.225, 0.184, 0.318, 0.13, 0.39, 0.067, 0.435,
    0, 0.45, -0.067, 0.435, -0.13, 0.39, -0.184, 0.318, -0.225, 0.225, -0.251, 0.116,
    -0.26, 0, -0.251, -0.116, -0.225, -0.225, -0.184, -0.318, -0.13, -0.39, -0.067, -0.435,
    0, -0.45
  ]];

  // Adjust font size on the flag cloth for 2-digit sums (10-18)
  fly.setDigit = (d, sure = 1) => {
    const { cvs, tex } = fly.flag;
    const g = cvs.getContext('2d');
    g.fillStyle = sure > 0.45 ? '#f4f7fb' : '#ece3cc';
    g.fillRect(0, 0, 128, 128);
    g.strokeStyle = '#aeb8c4';
    g.lineWidth = 5;
    g.strokeRect(2.5, 2.5, 123, 123);
    g.fillStyle = '#141a21';
    const s = String(d);
    const fSize = s.length > 1 ? 58 : 86;
    g.font = `700 ${fSize}px "Hiragino Kaku Gothic ProN", "Noto Sans JP", "Yu Gothic", system-ui, sans-serif`;
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.fillText(s, 64, 68);
    tex.needsUpdate = true;
  };

  isFlyReady = true;
  flyStateText.textContent = 'ハエ準備完了';
  if (isWorkerReady) startIfReady();
}).catch((err) => {
  console.error('Failed to load 3D fly:', err);
  flyStateText.textContent = '（3Dモデル読み込み失敗）';
});

// 5. Kenyon Cells Soma Positions & Canvas
async function loadKCPositions(kcIdx) {
  try {
    const res = await fetch(new URL('../flybrain/data/pos783.bin.gz?v=1', import.meta.url));
    const buf = await new Response(res.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
    const dv = new DataView(buf);
    const n = dv.getUint32(4, true);
    const xs = new Uint16Array(buf, 8, n), ys = new Uint16Array(buf, 8 + 2 * n, n);
    const out = new Float32Array(2 * kcIdx.length);
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const i of kcIdx) {
      if (xs[i] < x0) x0 = xs[i]; if (xs[i] > x1) x1 = xs[i];
      if (ys[i] < y0) y0 = ys[i]; if (ys[i] > y1) y1 = ys[i];
    }
    const sx = x1 - x0 || 1, sy = y1 - y0 || 1, aim = 0.36;
    const k = Math.min(1 / sx, aim / sy), ox = (1 - sx * k) / 2, oy = (aim - sy * k) / 2;
    kcIdx.forEach((i, j) => {
      out[2 * j] = ox + (xs[i] - x0) * k;
      out[2 * j + 1] = (oy + (ys[i] - y0) * k) / aim;
    });
    return out;
  } catch (e) {
    console.warn('Could not load KC positions:', e);
    return null;
  }
}

function drawKC(activeSlots = null) {
  const cv = document.getElementById('kc');
  if (!cv || !kcXY) return;
  const w = cv.clientWidth || 720, h = Math.round(w * 0.36);
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  cv.width = w * dpr; cv.height = h * dpr; cv.style.height = h + 'px';
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const n = kcXY.length / 2;
  const r = Math.max(0.7, Math.min(2.2, w / 420));
  const pad = 10;
  const on = new Uint8Array(n);
  if (activeSlots) for (const k of activeSlots) on[k] = 1;

  for (const pass of [0, 1]) {
    ctx.fillStyle = pass ? '#59b7ff' : '#232c36';
    ctx.beginPath();
    for (let k = 0; k < n; k++) {
      if (on[k] !== pass) continue;
      const px = pad + kcXY[2 * k] * (w - 2 * pad);
      const py = pad + kcXY[2 * k + 1] * (h - 2 * pad);
      ctx.moveTo(px + r, py);
      ctx.arc(px, py, r, 0, Math.PI * 2);
    }
    ctx.fill();
  }

  const kcCountEl = document.getElementById('kcCount');
  if (activeSlots) {
    kcCountEl.textContent = `発火細胞: ${activeSlots.length.toLocaleString()} 個 (${(100 * activeSlots.length / n).toFixed(1)}%)`;
  }
}

window.addEventListener('resize', () => {
  drawKC();
  if (fly) fly.resize();
});

// 6. Drawing Pad Handler (Interactive Canvas)
function setupDrawPad(canvas, preview, clearBtn) {
  const ctx = canvas.getContext('2d');
  const prevCtx = preview.getContext('2d');
  let drawing = false;
  let hasContent = false;

  ctx.lineWidth = 14;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = '#ffffff';

  function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: (clientX - rect.left) * (canvas.width / rect.width),
      y: (clientY - rect.top) * (canvas.height / rect.height),
    };
  }

  function start(e) {
    drawing = true;
    hasContent = true;
    clearBtn.hidden = false;
    enterHandMode();
    const pos = getPos(e);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
    e.preventDefault();
  }

  function move(e) {
    if (!drawing) return;
    const pos = getPos(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    updatePreview();
    e.preventDefault();
  }

  function stop() {
    if (!drawing) return;
    drawing = false;
    updatePreview();
  }

  function updatePreview() {
    prevCtx.drawImage(canvas, 0, 0, 12, 12);
  }

  canvas.addEventListener('mousedown', start);
  canvas.addEventListener('mousemove', move);
  window.addEventListener('mouseup', stop);

  canvas.addEventListener('touchstart', start, { passive: false });
  canvas.addEventListener('touchmove', move, { passive: false });
  window.addEventListener('touchend', stop);

  clearBtn.addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    prevCtx.clearRect(0, 0, 12, 12);
    clearBtn.hidden = true;
    hasContent = false;
  });

  return {
    clear: () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      prevCtx.clearRect(0, 0, 12, 12);
      clearBtn.hidden = true;
      hasContent = false;
    },
    hasStrokes: () => hasContent,
    getImageFloatArray: () => {
      const imgData = prevCtx.getImageData(0, 0, 12, 12).data;
      const arr = new Float32Array(144);
      for (let i = 0; i < 144; i++) arr[i] = imgData[i * 4] / 255.0;
      return arr;
    },
    drawPixels: (pixels) => {
      hasContent = true;
      clearBtn.hidden = false;
      const imgData = prevCtx.createImageData(12, 12);
      for (let i = 0; i < 144; i++) {
        const v = Math.round(pixels[i] * 255);
        imgData.data[i * 4] = v;
        imgData.data[i * 4 + 1] = v;
        imgData.data[i * 4 + 2] = v;
        imgData.data[i * 4 + 3] = v > 0 ? 255 : 0;
      }
      prevCtx.putImageData(imgData, 0, 0);

      // Scaled to 132x132 pad
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(preview, 0, 0, canvas.width, canvas.height);
    },
  };
}

const padL = setupDrawPad(padLeft, prevLeft, clearA);
const padR = setupDrawPad(padRight, prevRight, clearB);

function enterHandMode() {
  if (isHandMode) return;
  isHandMode = true;
  setAuto(false);
  predSum.textContent = '?';
  verdictMark.textContent = '–';
  verdictMark.className = 'verdict-mark';
  resultBox.className = 'result-box';
  resultSub.textContent = '手書き入力中';
  labelA.textContent = '✍';
  labelB.textContent = '✍';
  statusEl.textContent = '手書きモード: 数字Aと数字Bを描いたら「🧠 手書きを解かせる」を押してください。';
}

// 7. Score Tally
function updateTally() {
  const pct = asked ? Math.round((100 * right) / asked) : 0;
  tallyPct.textContent = asked ? `${pct}%` : '–';
  tallyFrac.textContent = asked ? `(${right}/${asked})` : '';
  tallySub.textContent = asked ? `このページで ${right} / ${asked} 問正解` : '出題を準備中…';
  histEl.innerHTML = hist.slice(-32).map((ok) => `<i class="${ok ? 'ok' : 'no'}"></i>`).join('');
  stagechip.innerHTML = `<b>正答率 ${asked ? pct + '%' : '14%'}</b><small>足し算テスト (${asked}問)</small>`;
}

// 8. Auto Mode Toggle
function setAuto(on) {
  auto = on;
  autobtn.textContent = on ? '自動停止' : '自動出題';
  autobadge.hidden = !on;
  if (!on && quizTimer) {
    clearTimeout(quizTimer);
    quizTimer = null;
  }
  if (on && !busy) {
    isHandMode = false;
    nextQuestion(300);
  }
}
autobtn.addEventListener('click', () => setAuto(!auto));

// 9. Question Loop & 3D Fly Writing Choreography
function nextQuestion(delay = 0, awaitFly = false) {
  if (quizTimer) clearTimeout(quizTimer);
  quizTimer = setTimeout(() => {
    quizTimer = null;
    if (busy) {
      nextQuestion(400, awaitFly);
      return;
    }
    if (awaitFly && fly && fly.isBusy() && ++flyWaits < 120) {
      nextQuestion(250, true);
      return;
    }
    flyWaits = 0;
    if (awaitFly) {
      statusEl.textContent = 'ひと休み — いつものハエに戻っています…';
      nextQuestion(3200 + Math.random() * 2000, false);
      return;
    }
    busy = true;
    statusEl.textContent = 'MNISTから出題を準備中…';
    worker.postMessage({ type: 'sample_pair' });
  }, delay);
}

function startIfReady() {
  if (isFlyReady && isWorkerReady) {
    busy = false;
    statusEl.textContent = '準備完了。足し算の出題を開始します。';
    setAuto(true);
  }
}

function runWritingChoreography(data) {
  currentSample = data;
  padL.drawPixels(data.imgL);
  padR.drawPixels(data.imgR);
  labelA.textContent = data.leftDigit;
  labelB.textContent = data.rightDigit;
  predSum.textContent = '?';
  verdictMark.textContent = '–';
  verdictMark.className = 'verdict-mark';
  resultBox.className = 'result-box';
  resultSub.textContent = 'ハエの予測';

  // Step 1: Fly writes Digit A with front leg
  cardA.classList.add('writing');
  cardB.classList.remove('writing');
  statusEl.textContent = `🪰 ハエが1つ目の数字「${data.leftDigit}」を地面に書いています…`;

  if (fly) {
    fly.show(String(data.leftDigit), 1.0, false, true, () => {
      // Done writing digit A!
      cardA.classList.remove('writing');
      setTimeout(step2, 350);
    });
  } else {
    setTimeout(step2, 800);
  }

  // Step 2: Fly writes Digit B with front leg
  function step2() {
    cardB.classList.add('writing');
    statusEl.textContent = `🪰 ハエが2つ目の数字「${data.rightDigit}」を地面に書いています…`;

    if (fly) {
      fly.show(String(data.rightDigit), 1.0, false, true, () => {
        // Done writing digit B!
        cardB.classList.remove('writing');
        setTimeout(step3, 350);
      });
    } else {
      setTimeout(step3, 800);
    }
  }

  // Step 3: Mushroom body addition inference
  function step3() {
    statusEl.textContent = `🧠 キノコ体が足し算を計算中 (${data.leftDigit} + ${data.rightDigit} = ?)…`;
    worker.postMessage({
      type: 'ask_pair',
      imgL: data.imgL,
      imgR: data.imgR,
      leftDigit: data.leftDigit,
      rightDigit: data.rightDigit,
      target: data.target,
      mode: 'sample',
    });
  }
}

function revealAnswer(m) {
  const { predictedSum, target, isCorrect, leftDigit, rightDigit, margin, drive, slots } = m;
  predSum.textContent = predictedSum;

  if (m.mode === 'sample') {
    resultBox.className = `result-box ${isCorrect ? 'correct' : 'incorrect'}`;
    verdictMark.textContent = isCorrect ? '○' : '×';
    verdictMark.className = `verdict-mark ${isCorrect ? 'ok' : 'no'}`;
    resultSub.textContent = `正解: ${leftDigit} + ${rightDigit} = ${target}`;

    asked++;
    if (isCorrect) right++;
    hist.push(isCorrect);
    updateTally();

    if (isCorrect) sound.ok();
    else sound.ng();

    if (isCorrect) {
      statusEl.textContent = `「${predictedSum}」— 正解！餌が出ます。ハエが砂糖水を飲んでいます。`;
    } else {
      statusEl.textContent = `「${predictedSum}」— 不正解（正解は ${target}）。ハエが首をかしげています。`;
    }
  } else {
    // Custom handwritten answer
    resultBox.className = 'result-box';
    verdictMark.textContent = '';
    resultSub.textContent = `ハエの予測: 和 ${predictedSum}`;
    statusEl.textContent = `ハエの判定: 「${predictedSum}」（第2候補: 和 ${m.secondSum}）`;
  }

  // Update MBON drive bars & Kenyon cell map
  updateDriveMeters(drive, predictedSum);
  drawKC(slots);

  // 3D Fly Flag & Feeding Behavior
  if (fly) {
    fly.reward = !!isCorrect;
    fly.setDigit(predictedSum, Math.min(1, margin / 0.03));
    fly.raiseFlag();
    if (m.mode === 'sample' && !isCorrect) {
      setTimeout(() => { if (fly) fly.puzzle(); }, 1200);
    }
  }

  busy = false;
  if (auto) nextQuestion(600, true);
}

// 10. Worker Message Handling
worker.onmessage = async (e) => {
  const m = e.data;

  if (m.type === 'progress') {
    if (m.phase === 'downloading_brain') loadBar(0.3, '脳モデルを読み込み中…');
    else if (m.phase === 'loading_weights') loadBar(0.7, '足し算の学習済み重みを読み込み中…');
    else if (m.phase === 'loading_mnist') loadBar(0.9, 'MNISTテスト画像を読み込み中…');
  } else if (m.type === 'ready') {
    loadBar(1.0, '準備完了');
    isWorkerReady = true;
    kcXY = await loadKCPositions(m.kc);
    drawKC();
    if (isFlyReady) startIfReady();
  } else if (m.type === 'sampled') {
    runWritingChoreography(m);
  } else if (m.type === 'answer') {
    revealAnswer(m);
  } else if (m.type === 'error') {
    statusEl.textContent = `⚠️ エラー: ${m.message}`;
    busy = false;
  }
};

// Start Worker Initialization
loadBar(0.1, 'ハエ脳モデルの初期化を開始…');
worker.postMessage({ type: 'init' });

// 11. Manual User Controls
btnSample.addEventListener('click', () => {
  if (busy) return;
  isHandMode = false;
  setAuto(false);
  busy = true;
  statusEl.textContent = 'MNISTからランダム出題中…';
  worker.postMessage({ type: 'sample_pair' });
});

btnSolve.addEventListener('click', () => {
  if (busy) return;
  const imgL = padL.getImageFloatArray();
  const imgR = padR.getImageFloatArray();

  setAuto(false);
  busy = true;
  statusEl.textContent = '手書きの数字をハエが読んでいます…';

  // Fly writes / reads user's hand and judges
  cardA.classList.add('writing');
  setTimeout(() => {
    cardA.classList.remove('writing');
    cardB.classList.add('writing');
    setTimeout(() => {
      cardB.classList.remove('writing');
      worker.postMessage({
        type: 'ask_pair',
        imgL: Array.from(imgL),
        imgR: Array.from(imgR),
        leftDigit: '✍',
        rightDigit: '✍',
        target: null,
        mode: 'custom',
      });
    }, 500);
  }, 500);
});

btnClear.addEventListener('click', () => {
  padL.clear();
  padR.clear();
  predSum.textContent = '?';
  verdictMark.textContent = '–';
  verdictMark.className = 'verdict-mark';
  resultBox.className = 'result-box';
  resultSub.textContent = '手書きを入力してください';
  labelA.textContent = '–';
  labelB.textContent = '–';
  for (let i = 0; i <= 18; i++) {
    const item = document.getElementById(`drive-${i}`);
    item.className = 'drive-item';
    item.querySelector('.val').textContent = '–';
    item.querySelector('.drive-bar-fill').style.width = '0%';
  }
  drawKC(null);
  statusEl.textContent = '消去しました。数字を描くか「ランダム出題」を押してください。';
});
