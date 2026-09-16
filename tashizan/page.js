// Page controller for Tashizan (Addition) Demo

import { FlagFly } from '../suji/flag.js?v=38';

const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });

// UI Elements
const statusBanner = document.getElementById('statusBanner');
const statusText = document.getElementById('statusText');
const flyStateText = document.getElementById('flyStateText');
const flyCanvas = document.getElementById('flyCanvas');

const padLeft = document.getElementById('padLeft');
const padRight = document.getElementById('padRight');
const prevLeft = document.getElementById('prevLeft');
const prevRight = document.getElementById('prevRight');

const resultBox = document.getElementById('resultBox');
const predSum = document.getElementById('predSum');
const resultLabel = document.getElementById('resultLabel');
const badgeTag = document.getElementById('badgeTag');

const btnSample = document.getElementById('btnSample');
const btnSolve = document.getElementById('btnSolve');
const btnClear = document.getElementById('btnClear');
const driveGrid = document.getElementById('driveGrid');

let fly = null;
let currentSample = null;

// 1. Initialize 19 Drive Items
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

// 2. Initialize 3D Fly
try {
  fly = new FlagFly(flyCanvas);
  flyStateText.textContent = '元気なハエが待機しています';
} catch (err) {
  console.warn('3D Fly initialization error:', err);
  flyStateText.textContent = '3Dモデル準備中';
}

// 3. Drawing Pad Handler (Mouse & Touch)
function setupDrawPad(canvas, preview) {
  const ctx = canvas.getContext('2d');
  const prevCtx = preview.getContext('2d');
  let drawing = false;

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
    // Downsample 132x132 canvas to 12x12
    prevCtx.drawImage(canvas, 0, 0, 12, 12);
  }

  canvas.addEventListener('mousedown', start);
  canvas.addEventListener('mousemove', move);
  window.addEventListener('mouseup', stop);

  canvas.addEventListener('touchstart', start, { passive: false });
  canvas.addEventListener('touchmove', move, { passive: false });
  window.addEventListener('touchend', stop);

  return {
    clear: () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      prevCtx.clearRect(0, 0, 12, 12);
    },
    getImageFloatArray: () => {
      const imgData = prevCtx.getImageData(0, 0, 12, 12).data;
      const arr = new Float32Array(144);
      for (let i = 0; i < 144; i++) {
        // Red channel normalized to 0..1
        arr[i] = imgData[i * 4] / 255.0;
      }
      return arr;
    },
    drawPixels: (pixels) => {
      // Draw 12x12 onto preview and scaled up onto draw pad
      const imgData = prevCtx.createImageData(12, 12);
      for (let i = 0; i < 144; i++) {
        const v = Math.round(pixels[i] * 255);
        imgData.data[i * 4] = v;
        imgData.data[i * 4 + 1] = v;
        imgData.data[i * 4 + 2] = v;
        imgData.data[i * 4 + 3] = v > 0 ? 255 : 0;
      }
      prevCtx.putImageData(imgData, 0, 0);

      // Scaled to pad
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(preview, 0, 0, canvas.width, canvas.height);
    },
  };
}

const padL = setupDrawPad(padLeft, prevLeft);
const padR = setupDrawPad(padRight, prevRight);

// 4. Update UI with Prediction Result
function showResult(data) {
  predSum.textContent = data.predictedSum;
  resultBox.className = 'result-box';

  if (data.mode === 'sample') {
    const isCorrect = data.isCorrect;
    resultBox.classList.add(isCorrect ? 'correct' : 'incorrect');
    badgeTag.style.display = 'block';
    badgeTag.className = `badge-tag ${isCorrect ? 'ok' : 'no'}`;
    badgeTag.textContent = isCorrect ? '正解！' : `不正解 (正解: ${data.target})`;
    resultLabel.textContent = `正解: ${data.leftDigit} + ${data.rightDigit} = ${data.target}`;

    if (fly) {
      flyStateText.textContent = isCorrect ? '正解！ハエが喜んで餌を食べています' : '不正解。ハエが考え込んでいます';
      try {
        if (isCorrect) fly.mark(true);
        else fly.mark(false);
      } catch { /* ignore animation error */ }
    }
  } else {
    badgeTag.style.display = 'none';
    resultLabel.textContent = `ハエの予測: 和 ${data.predictedSum}`;
    if (fly) flyStateText.textContent = `ハエの判定: ${data.predictedSum}`;
  }

  // Update Drive Meters
  const drive = data.drive;
  const minDrive = Math.min(...drive);
  const maxDrive = Math.max(...drive);
  const range = maxDrive - minDrive || 1;

  for (let i = 0; i <= 18; i++) {
    const item = document.getElementById(`drive-${i}`);
    const isWinner = i === data.predictedSum;
    item.className = `drive-item ${isWinner ? 'winner' : ''}`;

    const valEl = item.querySelector('.val');
    valEl.textContent = drive[i].toFixed(2);

    // Invert bar: lower drive = stronger activation (longer bar)
    const normalizedStrength = Math.max(0, Math.min(100, ((maxDrive - drive[i]) / range) * 100));
    const fillEl = item.querySelector('.drive-bar-fill');
    fillEl.style.width = `${normalizedStrength.toFixed(0)}%`;
  }
}

// 5. Worker Message Handling
worker.onmessage = (e) => {
  const { type, message, ...data } = e.data;

  if (type === 'progress') {
    statusText.textContent = message || '準備中...';
  } else if (type === 'ready') {
    statusBanner.className = 'ready';
    statusText.textContent = '✅ ハエ脳モデル準備完了！好きな数字を描くか、ランダム出題を押してください。';
    btnSample.disabled = false;
    btnSolve.disabled = false;
  } else if (type === 'answer') {
    showResult(data);
    btnSample.disabled = false;
    btnSolve.disabled = false;
  } else if (type === 'error') {
    statusBanner.style.background = '#3a181c';
    statusBanner.style.color = '#ffc0c0';
    statusText.textContent = `⚠️ エラー: ${message}`;
  }
};

// Start initialization
btnSample.disabled = true;
btnSolve.disabled = true;
worker.postMessage({ type: 'init' });

// 6. User Event Listeners
btnSample.addEventListener('click', () => {
  btnSample.disabled = true;
  btnSolve.disabled = true;
  predSum.textContent = '...';
  resultBox.className = 'result-box';
  badgeTag.style.display = 'none';
  resultLabel.textContent = 'キノコ体が思考中...';
  if (fly) flyStateText.textContent = 'ハエが左右の数字を見ています...';

  worker.postMessage({ type: 'sample_and_ask' });
});

worker.addEventListener('message', (e) => {
  if (e.data.type === 'answer' && e.data.mode === 'sample') {
    padL.drawPixels(e.data.imgL);
    padR.drawPixels(e.data.imgR);
  }
});

btnSolve.addEventListener('click', () => {
  const imgL = padL.getImageFloatArray();
  const imgR = padR.getImageFloatArray();

  btnSample.disabled = true;
  btnSolve.disabled = true;
  predSum.textContent = '...';
  resultBox.className = 'result-box';
  badgeTag.style.display = 'none';
  resultLabel.textContent = 'キノコ体が思考中...';
  if (fly) flyStateText.textContent = 'ハエが手書きの数字を解いています...';

  worker.postMessage({
    type: 'ask',
    payload: {
      imgL: Array.from(imgL),
      imgR: Array.from(imgR),
    },
  });
});

btnClear.addEventListener('click', () => {
  padL.clear();
  padR.clear();
  predSum.textContent = '?';
  resultBox.className = 'result-box';
  badgeTag.style.display = 'none';
  resultLabel.textContent = '手書きを入力してください';
  for (let i = 0; i <= 18; i++) {
    const item = document.getElementById(`drive-${i}`);
    item.className = 'drive-item';
    item.querySelector('.val').textContent = '–';
    item.querySelector('.drive-bar-fill').style.width = '0%';
  }
});
