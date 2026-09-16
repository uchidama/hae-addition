# アーキテクチャ解説: ハエ脳モデル (suji / MNIST 数字認識)

本ドキュメントでは、`hae.satoru.net` (リポジトリ: `uchidama/hae-addition`) におけるハエ脳（キノコ体: Mushroom Body）を用いた数字認識（`suji` コース）の全体構造とデータフローを解説します。

---

## 1. 全体データフロー

MNIST 画像の入力から、キノコ体でのスパイク計算、MBON からのクラス判定、ドーパミンによるシナプス可塑性更新までのデータフローは以下の通りです。

```text
MNIST image (28x28, 手書き数字1〜9)
    ↓ [前処理: 12x12 に縮小, 0-1正規化]
    ↓ ファイル: juku/trainer.mjs (関数: img()) / juku/data/mnist12_*.bin.gz
image preprocessing (144 pixels, Float32Array)
    ↓ [ラウンドロビン配分: 144ピクセル → 685本のALPN]
    ↓ ファイル: juku/reader.mjs (関数: stimulate())
PN (Antennal Lobe Projection Neurons: 685個)
    ↓ [ポアソン発火刺激: hz = 220 * pixel_value]
    ↓ [FlyWire v783 コネクトーム上のシナプス伝達]
    ↓ ファイル: flybrain/flybrain.js, flybrain/src/brain.c
KC (Kenyon Cells: キノコ体固有ニューロン, スパース発火)
    ↓ [KC -> MBON シナプス結合: 学習可能な重み gain]
    ↓ [KCスパイク数 × 重み × シナプス数 による総入力計算]
    ↓ ファイル: flybrain/src/brain.c (関数: fb_drive_probe_add / driveByGroup)
KC -> MBON (96個のMBONを9文字分のコンパートメントに配分)
    ↓ [最もドライブ（KCからの総入力）が「低い」コンパートメントを選択]
    ↓ ファイル: juku/reader.mjs (関数: decide())
class prediction (予測クラス 1〜9)
    ↓ [正解判定: 不正解の場合のみフィードバック]
    ↓ ファイル: juku/reader.mjs (関数: practise() -> feedback())
dopamine / plasticity update
    - 正解コンパートメント: 正のドーパミン (+1) → シナプス減弱 (LTD)
    - 誤答コンパートメント: 負のドーパミン (-1) → シナプス増強 (LTP)
    - 画像呈示を継続しながら 300ms シミュレーション
    ↓ ファイル: flybrain/src/brain.c (関数: fb_dopa_set, step_plastic())
```

---

## 2. モジュール構成とファイル一覧

| モジュール / ファイル | 役割・概要 |
|---|---|
| `flybrain/src/brain.c` | C言語で書かれた LIF (Leaky Integrate-and-Fire) スパイクシミュレータコア。WASMへコンパイル。イベント駆動による超高速計算とドーパミン依存型シナプス可塑性を内包。 |
| `flybrain/flybrain.wasm` | `brain.c` を wasm32-freestanding にコンパイルしたバイナリ (~19 KB)。 |
| `flybrain/flybrain.js` | WebAssembly インスタンスのラッパー。コネクトームグラフの読み込み、スパイク刺激 (`stimulate`)、シミュレーション実行 (`run`)、可塑性パラメータ設定などを提供。 |
| `flybrain/data/flywire783.fbg.gz` | FlyWire v783 コネクトームグラフ (138,639 ニューロン、1,500万シナプス)。 |
| `flybrain/data/mb783.json` | キノコ体関連ニューロン（KC, MBON, DAN, MBIN, ALPN）のインデックス定義。 |
| `juku/reader.mjs` | ハエ脳の読字エンジン（推論・練習）。MBONのコンパートメント割り当て、画像からALPNへの刺激変換、推論決定 (`decide`)、フィードバック (`feedback`) を統括。 |
| `juku/trainer.mjs` | 継続的学習（学校）プロセス。MNISTデータのロード、練習ループ (`practise`)、定期テスト (`sitTest`)、状態・重み保存を担当。 |
| `juku/data/mnist12_*.bin.gz` | 12x12ピクセルにリサイズされたMNISTバイナリデータ（train: 54,077枚, test: 9,020枚）。 |
| `juku/state/suji/` | 学習結果の保存ディレクトリ。重み (`brain-<n>.bin.gz`)、状態 (`status.json`)、履歴 (`log.jsonl`)。 |
| `suji/` | Web UI（ブラウザ上で 3D のハエが数字を読み、前脚で一筆書きするフロントエンド）。 |

---

## 3. 各処理の詳細仕様

### 3.1 MNIST データ形式と前処理
- **画像サイズ**: 12×12 = 144 ピクセル
- **対象数字**: **1 〜 9 の 9クラス**（0 は除外されている）
- **バイナリフォーマット**: 1サンプルあたり 145 バイト
  - 先頭 1 バイト: 正解ラベル（1〜9）
  - 後続 144 バイト: ピクセル値（0〜255）
- **正規化処理** (`juku/trainer.mjs: img`):
  ```javascript
  const img = (buf, n) => Float32Array.from(buf.subarray(n * rec + 1, (n + 1) * rec), (v) => v / 255);
  ```

### 3.2 PN (Projection Neuron) への入力変換
- **使用ニューロン**: 嗅覚受容系投射神経 `ALPN` (685本)
- **チャンネル配分** (`juku/reader.mjs: makeReader`):
  - 12345 をシードとする疑似乱数で 685本の ALPN をシャッフル。
  - 144 ピクセルに対してラウンドロビン方式で均等に割り振る（各ピクセルに約4〜5本の ALPN が対応）。
- **刺激印加** (`juku/reader.mjs: stimulate`):
  - インク閾値: `C.ink = 0.05`
  - 発火周波数: `C.hz = 220 Hz`
  - ピクセル値が 0.05 を超える場合、対応する ALPN 群に `220 * pixel_value` Hz のポアソン入力スパイクを与える。

### 3.3 キノコ体 (KC) と MBON の構造
- **キノコ体以外の抑制**:
  - 全脳モデルに嗅覚入力を入れると暴走（runaway）するため、KC, MBON, DAN, MBIN, ALPN 以外の全ニューロン（約13万個）を `brain.silence()` で無効化。
- **MBON のコンパートメント割り当て**:
  - 全 96 個の MBON を、各数字コンパートメント（1〜9の9個）に配分。
  - KC からの入力シナプス数が多い MBON から順に、現時点で最も入力合計が少ないコンパートメントへ順次割り当てる（Greedy Balancing）。
  - 各コンパートメントには 10〜11 個の MBON が属し、約 6,900 本の KC 入力を受ける。

### 3.4 推論とクラス判定 (Readout)
- **シミュレーション**: 画像刺激を与えて 400 ms (`C.ms = 400`) シミュレーションを実行。
- **KC スパイク集計**: 各 KC のスパイク数を取得。
- **ドライブ計算** (`brain.driveByGroup(KC, spikes)`):
  - 各コンパートメントについて、「KC スパイク数 × シナプス数 × 現在の重み (gain)」をナイーブ（初期状態: gain=1.0）時の値で割った相対ドライブ比率を計算。
- **判定ルール** (`juku/reader.mjs: decide`):
  - **ドライブが最も小さい（抑制・減弱が最も進んでいる）コンパートメントを答えとする**。
  - これはショウジョウバエの忌避学習回路（学習した刺激に対して MBON 発火が低下する）の生体機構を反映している。

### 3.5 ドーパミンによる学習則 (Plasticity)
- **学習タイミング**:
  - 推論が**不正解**だった場合のみ更新（エラードリブン）。
  - 画像呈示を維持したまま、300 ms (`C.feedbackMs = 300`) 追加シミュレーション。
- **ドーパミン信号**:
  - **正解コンパートメント**: ドーパミン `+1.0` を印加 → シナプス重み減弱（LTD: 次回同じ画像が来たとき MBON が静かになり、選ばれやすくなる）。
  - **誤答コンパートメント**: ドーパミン `-1.0` を印加 → シナプス重み増強（LTP: 次回 MBON が強く反応し、選ばれにくくなる）。
- **可塑性パラメータ**:
  - 学習率: `eta = 6e-5`
  - 重み下限: `gainMin = 0.05`
  - 重み上限: `gainMax = 2.0`
  - トレース時定数: `tauTrace = 40 ms`
  - ドーパミン時定数: `tauDopa = 1e7 ms`

### 3.6 重みの保存形式
- **形式**: Gzip圧縮された単精度浮動小数点数配列 (`Float32Array`)
- **ファイル名**: `brain-<practised_count>.bin.gz` (例: `brain-110217.bin.gz`)
- **シナプス数**: 62,261 本（KC → 96 MBON の全シナプス重み）
- **サイズ**: 展開時 約243 KB、Gzip時 約98 KB

### 3.7 実行方法
- **推論/学習サーバー実行**:
  ```bash
  node juku/trainer.mjs suji
  ```
- **Web UI 実行**:
  ```bash
  npx serve .
  ```
  ブラウザで `http://localhost:3000/suji/` を開く。
