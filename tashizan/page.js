// Page controller for Tashizan (Addition) FlyBrain Demo
// Features sequential 3D fly writing with front-leg IK and mushroom body addition judgment

import { FlagFly } from '../suji/flag.js?v=38';
import { Sound } from '../juku/sound.js?v=1';

const worker = new Worker(new URL('./worker.js?v=5', import.meta.url), { type: 'module' });
const sound = new Sound();

// UI Elements
const statusEl = document.getElementById('status');
const stagechip = document.getElementById('stagechip');
const soundBtn = document.getElementById('soundbtn');
const flyStateText = document.getElementById('flyStateText');
const flyCanvas = document.getElementById('fly');

// Floating Brain Window Elements
const brainBtn = document.getElementById('brainbtn');
const brainWin = document.getElementById('brainwin');
const bwHead = document.getElementById('bwhead');
const bwMode = document.getElementById('bwmode');
const bwMin = document.getElementById('bwmin');
const bwClose = document.getElementById('bwclose');
const bwState = document.getElementById('bwstate');
const kcMini = document.getElementById('kcmini');
const kcMiniPct = document.getElementById('kcminipct');
const bpBars = document.getElementById('bpbars');

const cardA = document.getElementById('cardA');
const cardB = document.getElementById('cardB');
const padLeft = document.getElementById('padLeft');
const padRight = document.getElementById('padRight');
const prevLeft = document.getElementById('prevLeft');
const prevRight = document.getElementById('prevRight');
const labelA = document.getElementById('labelA');
const labelB = document.getElementById('labelB');
const recogA = document.getElementById('recogA');
const recogB = document.getElementById('recogB');
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

// Brain window state
let brainOn = true;
try { brainOn = localStorage.getItem('hae-brain-open') !== '0'; } catch { /* no storage */ }
const SERIES = 90;
const series = [];
let heat = null;
let heatR = null;
let heatG = null;
let heatB = null;
let heatT = 0;
let rafOn = false;
let replaying = false;
let currentBwMode = 'idle';
let metaCells = { pn: 685, kc: 5177, mbon: 96, dn: 2 };
let chunkMs = 25;
let lastSlots = null;
let lastSlotsA = null;
let lastSlotsB = null;
let lastDrive = null;
let lastPredicted = null;

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
  let saved = null;
  try { saved = localStorage.getItem('hae-brain-pos'); } catch { /* no storage */ }
  if (!saved && brainWin && !brainWin.hidden) placeWindow();
}).catch((err) => {
  console.error('Failed to load 3D fly:', err);
  flyStateText.textContent = '（3Dモデル読み込み失敗）';
});

// 5. Floating Brain Window Controller & Live Activity Simulation
const MODE_TEXT = {
  idle: 'ふだん（背景入力だけ）',
  writeA: '数字Aを書いている（前脚の運動指令）',
  writeB: '数字Bを書いている（前脚の運動指令）',
  readA: '数字Aを見ている（キノコ体 MB1）',
  readB: '数字Bを見ている（キノコ体 MB1）',
  calc: '足し算を計算中（和 0〜18）',
  eat: '餌を食べている',
  fly: '飛行中',
};

const MODE_SHORT = {
  idle: 'ふだん',
  writeA: 'Aを書く',
  writeB: 'Bを書く',
  readA: 'A見る',
  readB: 'B見る',
  calc: '足し算計算',
  eat: '食べる',
  fly: '飛行',
};

function setMode(m) {
  currentBwMode = m;
  if (!bwMode) return;
  bwMode.textContent = window.innerWidth <= 640 ? (MODE_SHORT[m] || m) : (MODE_TEXT[m] || m);
  bwMode.className = 'bwmode ' + m;
}

function feed(f, mode) {
  series.push({ pn: f.pn, kc: f.kc, mb: f.mb, dn: f.dn || 0, mode });
  if (series.length > SERIES) series.shift();
  if (heat && f.slots) {
    let r = 100, g = 175, b = 240; // Default idle noise: cool cyan
    if (mode === 'readA') {
      r = 255; g = 130; b = 25; // 識別A: 鮮やかなオレンジ (#ff8219)
    } else if (mode === 'readB') {
      r = 255; g = 205; b = 30; // 識別B: 鮮やかなイエロー・琥珀 (#ffcd1e)
    } else if (mode === 'calc') {
      r = 255; g = 50; b = 150; // 足し算の計算: 鮮烈なビビッドピンク (#ff3296)
    }
    for (let i = 0; i < f.slots.length; i++) {
      const s = f.slots[i];
      heat[s] = 1.0;
      if (heatR) {
        heatR[s] = r;
        heatG[s] = g;
        heatB[s] = b;
      }
    }
  }
}

function setBrain(on) {
  brainOn = on;
  try { localStorage.setItem('hae-brain-open', on ? '1' : '0'); } catch { /* no storage */ }
  if (brainBtn) {
    brainBtn.setAttribute('aria-pressed', String(on));
  }
  if (brainWin) {
    brainWin.hidden = !on;
  }
  if (on) {
    placeWindow();
    drawBrainBars();
    if (!rafOn) {
      rafOn = true;
      requestAnimationFrame(drawBrainLive);
    }
  }
  worker.postMessage({ type: 'live', on });
}

function placeWindow(pos) {
  if (!brainWin) return;
  const w = brainWin.offsetWidth || 300;
  const h = brainWin.offsetHeight || 320;
  let p = pos;
  if (!p) {
    try { p = JSON.parse(localStorage.getItem('hae-brain-pos') || 'null'); } catch { p = null; }
  }
  if (!p) {
    const stage = flyCanvas ? flyCanvas.getBoundingClientRect() : document.querySelector('.flywrap').getBoundingClientRect();
    p = { x: Math.min(stage.right, window.innerWidth) - w - 8, y: stage.bottom - h - 8 };
  }
  const x = Math.min(Math.max(4, p.x), window.innerWidth - w - 4);
  const y = Math.min(Math.max(4, p.y), window.innerHeight - 40);
  brainWin.style.left = x + 'px';
  brainWin.style.top = y + 'px';
  return { x, y };
}

// Window Dragging & Compact mode toggling
(() => {
  if (!bwHead || !brainWin) return;
  let drag = null;
  bwHead.addEventListener('pointerdown', (e) => {
    if (e.target.closest('button')) return;
    bwHead.setPointerCapture(e.pointerId);
    const r = brainWin.getBoundingClientRect();
    drag = { dx: e.clientX - r.left, dy: e.clientY - r.top };
    brainWin.classList.add('dragging');
  });
  bwHead.addEventListener('pointermove', (e) => {
    if (drag) placeWindow({ x: e.clientX - drag.dx, y: e.clientY - drag.dy });
  });
  const end = () => {
    if (!drag) return;
    drag = null;
    brainWin.classList.remove('dragging');
    try {
      localStorage.setItem('hae-brain-pos', JSON.stringify({
        x: parseFloat(brainWin.style.left),
        y: parseFloat(brainWin.style.top),
      }));
    } catch { /* no storage */ }
  };
  bwHead.addEventListener('pointerup', end);
  bwHead.addEventListener('pointercancel', end);

  bwClose?.addEventListener('click', () => setBrain(false));

  let compact = window.innerWidth <= 640;
  try {
    const c = localStorage.getItem('hae-brain-compact');
    if (c != null) compact = c === '1';
  } catch { /* no storage */ }

  const setCompact = (on) => {
    compact = on;
    brainWin.classList.toggle('compact', on);
    if (bwMin) {
      bwMin.textContent = on ? '▢' : '－';
      bwMin.title = on ? '広げる' : 'コンパクトにする';
    }
    try { localStorage.setItem('hae-brain-compact', on ? '1' : '0'); } catch { /* no storage */ }
    if (brainOn) placeWindow({ x: parseFloat(brainWin.style.left) || 0, y: parseFloat(brainWin.style.top) || 0 });
  };
  bwMin?.addEventListener('click', () => setCompact(!compact));
  setCompact(compact);

  window.addEventListener('resize', () => {
    if (brainOn) placeWindow({ x: parseFloat(brainWin.style.left), y: parseFloat(brainWin.style.top) });
  });
})();

brainBtn?.addEventListener('click', () => setBrain(!brainOn));

// Writing on the ground is a movement: tell the brain how hard the front leg is working
let motorSent = 0;
setInterval(() => {
  if (!brainOn || !fly) return;
  const writing = fly.phase === 'writing' && fly.pen;
  const drive = writing ? (fly.pen.down ? 1 : 0.35) : 0;
  if (Math.abs(drive - motorSent) > 0.01) {
    motorSent = drive;
    worker.postMessage({ type: 'motor', drive });
  }
}, 100);

function drawCompactState() {
  if (!bwState) return;
  let key = 'idle';
  if (currentBwMode.startsWith('read') || currentBwMode === 'calc') key = currentBwMode;
  else if (fly && fly.fl) key = 'fly';
  else if (fly && fly.phase === 'writing') key = 'write';
  else if (fly && (fly.phase === 'toFood' || fly.phase === 'feeding' || fly.phase === 'retract')) key = 'eat';
  if (bwState.dataset.key === key) return;
  bwState.dataset.key = key;
  bwState.textContent = {
    idle: '通常',
    write: '字を書く',
    readA: '数字A見る',
    readB: '数字B見る',
    calc: '足し算計算',
    eat: '食べる',
    fly: '飛行',
  }[key] || key;
  bwState.className = 'bwstate ' + key;
}

// 6. Kenyon Cells Soma Positions & Canvas
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

function drawKC(activeSlots = null, ignite = false) {
  if (!kcXY) return;
  const n = kcXY.length / 2;
  if (!heat) {
    heat = new Float32Array(n);
    heatR = new Uint8Array(n);
    heatG = new Uint8Array(n);
    heatB = new Uint8Array(n);
  }
  if (activeSlots && ignite) {
    for (let i = 0; i < activeSlots.length; i++) {
      const s = activeSlots[i];
      heat[s] = 1.0;
      if (heatR) {
        heatR[s] = 255; heatG[s] = 150; heatB[s] = 30;
      }
    }
  }
  const kcCountEl = document.getElementById('kcCount');
  if (kcCountEl && activeSlots) {
    const total = activeSlots.length;
    const nA = lastSlotsA ? lastSlotsA.length : 0;
    const nB = lastSlotsB ? lastSlotsB.length : 0;
    const sub = (nA && nB) ? ` [A: ${nA}個, B: ${nB}個]` : '';
    kcCountEl.textContent = `発火細胞: 2文字計 ${total.toLocaleString()} 個 (${(100 * total / n).toFixed(1)}%)${sub}`;
  }
  if (!brainOn) {
    drawStaticKC(activeSlots);
  }
}

function drawStaticKC(activeSlots) {
  const bigKc = document.getElementById('kc');
  if (!bigKc || !kcXY) return;
  const w = bigKc.clientWidth || 720, h = Math.round(w * 0.36);
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  if (bigKc.width !== w * dpr) {
    bigKc.width = w * dpr;
    bigKc.height = h * dpr;
    bigKc.style.height = h + 'px';
  }
  const ctx = bigKc.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const n = kcXY.length / 2;
  const pad = 10, r = Math.max(0.7, Math.min(2.2, w / 420));
  ctx.fillStyle = '#232c36';
  ctx.beginPath();
  for (let k = 0; k < n; k++) {
    const px = pad + kcXY[2 * k] * (w - 2 * pad);
    const py = pad + kcXY[2 * k + 1] * (h - 2 * pad);
    ctx.moveTo(px + r, py);
    ctx.arc(px, py, r, 0, 6.2832);
  }
  ctx.fill();

  if (activeSlots && activeSlots.length) {
    const setA = new Set(lastSlotsA || []);
    const setB = new Set(lastSlotsB || []);
    for (let i = 0; i < activeSlots.length; i++) {
      const s = activeSlots[i];
      const px = pad + kcXY[2 * s] * (w - 2 * pad);
      const py = pad + kcXY[2 * s + 1] * (h - 2 * pad);
      if (setA.has(s) && !setB.has(s)) {
        ctx.fillStyle = '#ff8219'; // 数字A (オレンジ)
      } else if (setB.has(s) && !setA.has(s)) {
        ctx.fillStyle = '#ffcd1e'; // 数字B (イエロー・琥珀)
      } else {
        ctx.fillStyle = '#ff3296'; // 両方・足し算計算 (ビビッドピンク)
      }
      ctx.beginPath();
      ctx.arc(px, py, r * 1.5, 0, 6.2832);
      ctx.fill();
    }
  }
}

function drawBrainBars(drive = lastDrive, predicted = lastPredicted) {
  if (!brainOn) return;
  if (lastSlots && kcXY) {
    const total = lastSlots.length;
    const totalPct = (100 * total / (kcXY.length / 2)).toFixed(1);
    const nA = lastSlotsA ? lastSlotsA.length : 0;
    const nB = lastSlotsB ? lastSlotsB.length : 0;
    if (nA && nB) {
      kcMiniPct.textContent = `2文字計 ${total.toLocaleString()} 個 (${totalPct}%) [A: ${nA}個, B: ${nB}個]`;
    } else {
      kcMiniPct.textContent = `${total.toLocaleString()} 個 (${totalPct}%)`;
    }
  }
  if (!drive || !bpBars) return;

  const cand = Array.from({ length: 19 }, (_, i) => [drive[i], i])
    .sort((a, b) => a[0] - b[0])
    .slice(0, 4);

  const lo = cand[0][0];
  const hi = Math.max(...drive);
  const span = (hi - lo) || 1;

  bpBars.innerHTML = cand.map(([v, c]) => `
    <div class="bprow">
      <b>和 ${c}</b>
      <span class="bpbar"><i class="${c === predicted ? 'win' : ''}" style="width:${(100 * (hi - v) / span).toFixed(0)}%"></i></span>
      <em>${v.toFixed(2)}</em>
    </div>
  `).join('');
}

function drawBrainLive(now) {
  if (!brainOn) {
    rafOn = false;
    return;
  }
  requestAnimationFrame(drawBrainLive);
  drawCompactState();

  if (!kcXY) return;
  const n = kcXY.length / 2;
  if (!heat) {
    heat = new Float32Array(n);
    heatR = new Uint8Array(n);
    heatG = new Uint8Array(n);
    heatB = new Uint8Array(n);
  }

  const dt = Math.min(0.1, (now - (heatT || now)) / 1000);
  heatT = now;
  const decay = Math.exp(-dt / 0.35);

  const dpr = Math.min(2, window.devicePixelRatio || 1);

  // 1. Render #kcmini in the floating brain window
  if (kcMini) {
    const w = kcMini.clientWidth || 280, h = Math.round(w * 0.36);
    if (kcMini.width !== w * dpr) {
      kcMini.width = w * dpr;
      kcMini.height = h * dpr;
      kcMini.style.height = h + 'px';
    }
    const x = kcMini.getContext('2d');
    x.setTransform(dpr, 0, 0, dpr, 0, 0);
    x.fillStyle = '#070a0d';
    x.fillRect(0, 0, w, h);

    const pad = 3, r = Math.max(0.6, w / 420);

    // Dark quiescent KCs
    x.fillStyle = '#242c35';
    x.beginPath();
    for (let k = 0; k < n; k++) {
      const px = pad + kcXY[2 * k] * (w - 2 * pad);
      const py = pad + kcXY[2 * k + 1] * (h - 2 * pad);
      x.moveTo(px + r, py);
      x.arc(px, py, r, 0, 6.2832);
    }
    x.fill();

    // Active glowing KCs (orange/amber for digits, cyan for idle noise)
    for (let k = 0; k < n; k++) {
      const v = heat[k];
      if (v < 0.03) continue;
      heat[k] = v * decay;
      const px = pad + kcXY[2 * k] * (w - 2 * pad);
      const py = pad + kcXY[2 * k + 1] * (h - 2 * pad);
      const red = heatR ? heatR[k] : 100;
      const grn = heatG ? heatG[k] : 175;
      const blu = heatB ? heatB[k] : 240;
      x.fillStyle = `rgba(${red},${grn},${blu},${v.toFixed(3)})`;
      x.beginPath();
      x.arc(px, py, r * (1 + 1.2 * v), 0, 6.2832);
      x.fill();
    }
  }

  // 2. ALSO render sparkling glow onto the big canvas #kc on the page!
  const bigKc = document.getElementById('kc');
  if (bigKc) {
    const w = bigKc.clientWidth || 720, h = Math.round(w * 0.36);
    if (bigKc.width !== w * dpr) {
      bigKc.width = w * dpr;
      bigKc.height = h * dpr;
      bigKc.style.height = h + 'px';
    }
    const ctx = bigKc.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const pad = 10, r = Math.max(0.7, Math.min(2.2, w / 420));
    ctx.fillStyle = '#232c36';
    ctx.beginPath();
    for (let k = 0; k < n; k++) {
      const px = pad + kcXY[2 * k] * (w - 2 * pad);
      const py = pad + kcXY[2 * k + 1] * (h - 2 * pad);
      ctx.moveTo(px + r, py);
      ctx.arc(px, py, r, 0, 6.2832);
    }
    ctx.fill();

    for (let k = 0; k < n; k++) {
      const v = heat[k];
      if (v < 0.03) continue;
      const px = pad + kcXY[2 * k] * (w - 2 * pad);
      const py = pad + kcXY[2 * k + 1] * (h - 2 * pad);
      const red = heatR ? heatR[k] : 100;
      const grn = heatG ? heatG[k] : 175;
      const blu = heatB ? heatB[k] : 240;
      ctx.fillStyle = `rgba(${red},${grn},${blu},${v.toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(px, py, r * (1 + 1.3 * v), 0, 6.2832);
      ctx.fill();
    }
  }

  // 3. Render 4 scrolling spike rate traces: PN, KC, MBON, DN
  const sec = (chunkMs || 25) / 1000;
  const traces = [
    ['pn', metaCells.pn || 685],
    ['kc', metaCells.kc || 5177],
    ['mb', metaCells.mbon || 96],
    ['dn', metaCells.dn || 2],
  ];

  for (const [key, cells] of traces) {
    const cv = document.getElementById('sp-' + key);
    if (!cv) continue;
    const cw = cv.clientWidth || 150, ch = cv.clientHeight || 22;
    if (cv.width !== cw * dpr) {
      cv.width = cw * dpr;
      cv.height = ch * dpr;
    }
    const g = cv.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, cw, ch);

    const hz = series.map((f) => (f[key] || 0) / cells / sec);
    const top = Math.max(key === 'pn' ? 30 : key === 'kc' ? 4 : key === 'dn' ? 60 : 10, ...hz);
    const step = cw / (SERIES - 1);
    const off = SERIES - series.length;

    // Background color shading according to mode
    series.forEach((f, i) => {
      if (f.mode === 'readA') {
        g.fillStyle = 'rgba(255,130,25,.28)';
        g.fillRect((off + i - 0.5) * step, 0, step + 0.5, ch);
      } else if (f.mode === 'readB') {
        g.fillStyle = 'rgba(255,205,30,.28)';
        g.fillRect((off + i - 0.5) * step, 0, step + 0.5, ch);
      } else if (f.mode === 'calc') {
        g.fillStyle = 'rgba(255,50,150,.32)'; // ビビッドピンク帯
        g.fillRect((off + i - 0.5) * step, 0, step + 0.5, ch);
      } else if (f.mode === 'ask') {
        g.fillStyle = 'rgba(255,130,25,.24)';
        g.fillRect((off + i - 0.5) * step, 0, step + 0.5, ch);
      } else if (f.mode?.startsWith('write')) {
        g.fillStyle = 'rgba(84,217,140,.18)';
        g.fillRect((off + i - 0.5) * step, 0, step + 0.5, ch);
      }
    });

    // Waveform line
    g.strokeStyle = '#59b7ff';
    g.lineWidth = 1.5;
    g.lineJoin = 'round';
    g.beginPath();
    hz.forEach((v, i) => {
      const px = (off + i) * step;
      const py = ch - 1.5 - (v / top) * (ch - 4);
      if (i) g.lineTo(px, py);
      else g.moveTo(px, py);
    });
    g.stroke();

    const hzEl = document.getElementById('hz-' + key);
    if (hzEl) {
      hzEl.textContent = hz.length
        ? `${hz[hz.length - 1] < 10 ? hz[hz.length - 1].toFixed(1) : Math.round(hz[hz.length - 1])} Hz`
        : '–';
    }
  }
}

function replay(frames, mode, done, onStep = null) {
  if (!brainOn || !frames?.length) {
    done();
    return;
  }
  replaying = true;
  setMode(mode);
  let i = 0;
  const next = () => {
    if (i < frames.length && brainOn) {
      const idx = i++;
      feed(frames[idx], mode);
      if (onStep) onStep(frames[idx], idx, frames.length);
      setTimeout(next, 50);
      return;
    }
    replaying = false;
    setMode('idle');
    done();
  };
  next();
}

function animateDriveProgress(drive, predicted, progress) {
  if (!drive || !bpBars) return;
  const narrow = window.innerWidth < 520;
  const count = narrow ? 3 : 4;

  const cand = Array.from({ length: 19 }, (_, i) => [drive[i], i])
    .sort((a, b) => a[0] - b[0])
    .slice(0, count);

  const lo = cand[0][0];
  const hi = Math.max(...drive);
  const span = (hi - lo) || 1;

  const isFinalizing = progress >= 0.75;
  bpBars.innerHTML = cand.map(([v, c], idx) => {
    const targetW = (100 * (hi - v) / span);
    const ease = Math.min(1, progress * (1.1 + 0.15 * (count - idx)));
    const currentW = Math.max(8, Math.round(targetW * ease));
    const isWin = isFinalizing && c === predicted;
    return `
      <div class="bprow">
        <b>和 ${c}</b>
        <span class="bpbar"><i class="${isWin ? 'win' : ''}" style="width:${currentW}%"></i></span>
        <em>${isFinalizing ? v.toFixed(2) : '探索中…'}</em>
      </div>
    `;
  }).join('');

  // Also animate driveGrid items on the page
  for (let i = 0; i <= 18; i++) {
    const item = document.getElementById(`drive-${i}`);
    if (!item) continue;
    const targetStrength = Math.max(0, Math.min(100, ((hi - drive[i]) / span) * 100));
    const fillEl = item.querySelector('.drive-bar-fill');
    if (fillEl) {
      fillEl.style.width = `${Math.round(targetStrength * Math.min(1, progress * 1.2))}%`;
    }
    const valEl = item.querySelector('.val');
    if (valEl) {
      valEl.textContent = isFinalizing ? drive[i].toFixed(2) : (progress > 0.4 ? (drive[i] + (1 - progress) * 2).toFixed(1) : '–');
    }
    if (isFinalizing && i === predicted) {
      item.className = 'drive-item winner';
    } else {
      item.className = 'drive-item';
    }
  }
}

function runCalculation(m, done) {
  const opPlus = document.getElementById('opPlus');
  const opEqual = document.getElementById('opEqual');
  if (opPlus) opPlus.classList.add('active');

  const nameA = m.recognizedA !== undefined ? m.recognizedA : (m.leftDigit || '?');
  const nameB = m.recognizedB !== undefined ? m.recognizedB : (m.rightDigit || '?');
  statusEl.textContent = `🧠 第2キノコ体 (MB2) が和を連想計算中: 「${nameA}」＋「${nameB}」…`;
  setMode('calc');

  const fullDrive = m.drive || [];
  const frames = m.framesCalc || [];

  if (!brainOn || !frames.length) {
    updateDriveMeters(fullDrive, m.predictedSum);
    drawBrainBars(fullDrive, m.predictedSum);
    setTimeout(() => {
      if (opPlus) opPlus.classList.remove('active');
      done();
    }, 450);
    return;
  }

  animateDriveProgress(fullDrive, m.predictedSum, 0.1);

  replay(frames, 'calc', () => {
    if (opPlus) opPlus.classList.remove('active');
    if (opEqual) opEqual.classList.add('active');

    updateDriveMeters(fullDrive, m.predictedSum);
    drawBrainBars(fullDrive, m.predictedSum);

    setTimeout(() => {
      if (opEqual) opEqual.classList.remove('active');
      done();
    }, 280);
  }, (frame, idx, total) => {
    const progress = (idx + 1) / total;
    animateDriveProgress(fullDrive, m.predictedSum, progress);
  });
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
  if (recogA) { recogA.textContent = '?'; recogA.parentElement.className = 'recog-label'; }
  if (recogB) { recogB.textContent = '?'; recogB.parentElement.className = 'recog-label'; }
  statusEl.textContent = '手書きモード: 数字Aと数字Bを描いたら「🧠 手書きを解かせる」を押してください。';
}

// 7. Score Tally
function updateTally() {
  const pct = asked ? Math.round((100 * right) / asked) : 0;
  tallyPct.textContent = asked ? `${pct}%` : '–';
  tallyFrac.textContent = asked ? `(${right}/${asked})` : '';
  tallySub.textContent = asked ? `このページで ${right} / ${asked} 問正解` : '出題を準備中…';
  histEl.innerHTML = hist.slice(-32).map((ok) => `<i class="${ok ? 'ok' : 'no'}"></i>`).join('');
  stagechip.innerHTML = `<b>正答率 ${asked ? pct + '%' : '75%'}</b><small>2段キノコ体 10k steps (${asked}問)</small>`;
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
    statusEl.textContent = '準備完了。10,000ステップ学習済みのハエ脳（正答率 ~75%）で足し算を開始します。';
    setAuto(true);
  }
}

function runWritingChoreography(data) {
  currentSample = data;
  padL.drawPixels(data.imgL);
  padR.drawPixels(data.imgR);
  labelA.textContent = data.leftDigit;
  labelB.textContent = data.rightDigit;
  if (recogA) { recogA.textContent = '?'; recogA.parentElement.className = 'recog-label'; }
  if (recogB) { recogB.textContent = '?'; recogB.parentElement.className = 'recog-label'; }
  predSum.textContent = '?';
  verdictMark.textContent = '–';
  verdictMark.className = 'verdict-mark';
  resultBox.className = 'result-box';
  resultSub.textContent = 'ハエの予測';

  // Step 1: Fly writes Digit A with front leg
  cardA.classList.add('writing');
  cardB.classList.remove('writing');
  setMode('writeA');
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
    setMode('writeB');
    statusEl.textContent = `🪰 ハエが2つ目の数字「${data.rightDigit}」を地面に書いています…`;

    if (fly) {
      fly.show(String(data.rightDigit), 1.0, false, false, () => {
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
    setMode('calc');
    statusEl.textContent = `🧠 キノコ体が足し算を計算中 (${data.leftDigit} + ${data.rightDigit} = ?)…`;
    const payload = {
      imgL: data.imgL,
      imgR: data.imgR,
      leftDigit: data.leftDigit,
      rightDigit: data.rightDigit,
      target: data.target,
      mode: 'sample',
    };
    worker.postMessage({
      type: 'ask_pair',
      payload,
      ...payload,
    });
  }
}

function revealAnswer(m) {
  const { predictedSum, target, isCorrect, leftDigit, rightDigit, recognizedA, recognizedB, margin, drive, slots } = m;
  lastDrive = drive;
  lastPredicted = predictedSum;
  lastSlotsA = m.slotsA || [];
  lastSlotsB = m.slotsB || [];
  const unionSet = new Set([...lastSlotsA, ...lastSlotsB]);
  lastSlots = Array.from(unionSet.size ? unionSet : (slots || []));

  predSum.textContent = predictedSum;

  const recogInfo = (recognizedA !== undefined && recognizedB !== undefined)
    ? ` (ハエの認識: ${recognizedA} + ${recognizedB})`
    : '';
  const recogInfoHtml = (recognizedA !== undefined && recognizedB !== undefined)
    ? `<br>(ハエの認識: ${recognizedA} + ${recognizedB})`
    : '';

  if (m.mode === 'sample') {
    resultBox.className = `result-box ${isCorrect ? 'correct' : 'incorrect'}`;
    verdictMark.textContent = isCorrect ? '○' : '×';
    verdictMark.className = `verdict-mark ${isCorrect ? 'ok' : 'no'}`;
    resultSub.innerHTML = `正解: ${leftDigit} + ${rightDigit} = ${target}${recogInfoHtml}`;

    asked++;
    if (isCorrect) right++;
    hist.push(isCorrect);
    updateTally();

    if (isCorrect) sound.ok();
    else sound.ng();

    if (isCorrect) {
      statusEl.textContent = `「${predictedSum}」— 正解！${recogInfo} 餌が出ます。ハエが砂糖水を飲んでいます。`;
      setMode('eat');
    } else {
      statusEl.textContent = `「${predictedSum}」— 不正解（正解は ${target}、ハエの認識は ${recognizedA}+${recognizedB}）。ハエが首をかしげています。`;
      setMode('idle');
    }
  } else {
    // Custom handwritten answer
    resultBox.className = 'result-box';
    verdictMark.textContent = '';
    resultSub.innerHTML = `ハエの予測: 和 ${predictedSum}${recogInfoHtml}`;
    statusEl.textContent = `ハエの判定: 「${predictedSum}」${recogInfo}（第2候補: 和 ${m.secondSum}）`;
    setMode('idle');
  }

  // Update MBON drive bars & Kenyon cell map
  updateDriveMeters(drive, predictedSum);
  drawBrainBars(drive, predictedSum);
  drawKC(lastSlots);

  if (recogA) {
    recogA.textContent = recognizedA !== undefined ? recognizedA : '?';
    recogA.parentElement.className = 'recog-label known';
  }
  if (recogB) {
    recogB.textContent = recognizedB !== undefined ? recognizedB : '?';
    recogB.parentElement.className = 'recog-label known';
  }

  // 3D Fly Flag & Feeding Behavior
  if (fly) {
    fly.hold = false;
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
    metaCells = m.cells || metaCells;
    chunkMs = m.chunkMs || chunkMs;
    kcXY = await loadKCPositions(m.kc);
    drawKC();
    setBrain(brainOn);
    if (isFlyReady) startIfReady();
  } else if (m.type === 'tick') {
    if (brainOn && !replaying) {
      const mode = motorSent > 0.01 ? (currentBwMode.startsWith('write') ? currentBwMode : 'writeA') : 'idle';
      if (currentBwMode !== mode && !currentBwMode.startsWith('write') && !currentBwMode.startsWith('read') && currentBwMode !== 'calc') {
        setMode(mode);
      }
      feed(m, mode);
    }
  } else if (m.type === 'sampled') {
    runWritingChoreography(m);
  } else if (m.type === 'answer') {
    if (brainOn && (m.framesL?.length || m.framesR?.length)) {
      statusEl.textContent = `🧠 第1キノコ体 (MB1) が数字A「${m.leftDigit}」を視覚認識中…`;
      cardA.classList.add('reading');
      cardB.classList.remove('reading');
      if (recogA) {
        recogA.textContent = '認識中…';
        recogA.parentElement.className = 'recog-label reading';
      }
      replay(m.framesL, 'readA', () => {
        cardA.classList.remove('reading');
        if (recogA) {
          recogA.textContent = m.recognizedA !== undefined ? m.recognizedA : '?';
          recogA.parentElement.className = 'recog-label known';
        }
        setMode('idle');
        setTimeout(() => {
          statusEl.textContent = `🧠 第1キノコ体 (MB1) が数字B「${m.rightDigit}」を視覚認識中…`;
          cardB.classList.add('reading');
          if (recogB) {
            recogB.textContent = '認識中…';
            recogB.parentElement.className = 'recog-label reading';
          }
          replay(m.framesR, 'readB', () => {
            cardB.classList.remove('reading');
            if (recogB) {
              recogB.textContent = m.recognizedB !== undefined ? m.recognizedB : '?';
              recogB.parentElement.className = 'recog-label known';
            }
            // Step 3: Run MB2 associative calculation!
            runCalculation(m, () => {
              revealAnswer(m);
            });
          });
        }, 220);
      });
    } else {
      runCalculation(m, () => {
        revealAnswer(m);
      });
    }
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
  statusEl.textContent = '🧠 手書きの数字をハエのキノコ体が読み始めます…';
  worker.postMessage({
    type: 'ask_pair',
    imgL: Array.from(imgL),
    imgR: Array.from(imgR),
    leftDigit: '✍',
    rightDigit: '✍',
    target: null,
    mode: 'custom',
  });
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
  if (recogA) { recogA.textContent = '–'; recogA.parentElement.className = 'recog-label'; }
  if (recogB) { recogB.textContent = '–'; recogB.parentElement.className = 'recog-label'; }
  for (let i = 0; i <= 18; i++) {
    const item = document.getElementById(`drive-${i}`);
    item.className = 'drive-item';
    item.querySelector('.val').textContent = '–';
    item.querySelector('.drive-bar-fill').style.width = '0%';
  }
  if (bpBars) bpBars.innerHTML = '';
  if (heat) heat.fill(0);
  drawKC(null);
  setMode('idle');
  statusEl.textContent = '消去しました。数字を描くか「ランダム出題」を押してください。';
});
