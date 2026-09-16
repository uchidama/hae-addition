# 足し算タスクにおける2入力符号化の設計検討 (Addition Input Design)

本ドキュメントでは、2枚の手書き数字画像（$A + B$）をハエ脳（キノコ体モデル）に入力するための神経符号化方式の設計、比較検討、および選定理由を記録します。

---

## 1. 比較検討した候補方式

ハエ脳モデルにおける嗅覚投射神経群（ALPN: Antennal Lobe Projection Neurons, 計 685本）に対して、2枚の 12×12 手書き画像（各 144 ピクセル）を入力する方式として、以下の3案を比較検討しました。

| 方式 | 構造 | メリット | デメリット / 課題 |
|---|---|---|---|
| **案A: 横連結方式**<br>(Horizontal Concatenation) | 2枚の画像を横に連結し、1枚の 12×24（288ピクセル）画像として扱い、685本の ALPN に配分。 | - 実装が最も単純。<br>- 既存の単一画像処理パイプラインと類似。 | - 左右境界での受容野重複や受容野境界の曖昧化。<br>- 各ピクセルあたりの投射神経本数が減少（約2.3本/pixel）。 |
| **案B: 投射神経分割方式**<br>(Split-PN Encoding) **【採用】** | 685本の ALPN を左群（342本）と右群（343本）に明確に分割。左画像は左 ALPN 群へ、右画像は右 ALPN 群へそれぞれ独立してラウンドロビン配分。 | - **左右の数字位置が神経解剖学的に厳密に分離**される。<br>- $2 + 7$ と $7 + 2$ が明確に異なる入力パターンとなり、交換法則（Phase 7）の検証が真の汎化テストとして成立する。<br>- C言語/WASMコアへの変更が一切不要。 | - 左右それぞれに利用可能な投射神経数が半分（342本 / 343本）になるが、144ピクセルに対して各2.3〜2.4本と十分な接続数を確保。 |
| **案C: 時間差入力方式**<br>(Sequential / Temporal) | 同じ 685本の ALPN 群に対して、時刻 $t_1$ に第1オペランド、時刻 $t_2$ に第2オペランドを時間差で入力。 | - 生体ハエの系列受容に近い。<br>- 全 685本の ALPN をフル活用可能。 | - 生体 LIF モデルにおける膜電位やシナプス痕跡の時定数調整が必要。<br>- シミュレーション時間が倍増し、短期記憶機構（ recurrent / trace ）の挙動が複雑化。 |

---

## 2. 選定理由: 案B (Split-PN Encoding) の採用

指示書における第一候補である **案B（Split-PN Encoding）** を採用しました。主な選定理由は以下の通りです：

1. **左右オペランドの位置特異性の完全な保持**:
   - 足し算において、$A + B$ と $B + A$ は同じ和（可換）を持ちますが、入力としては「第1オペランドが $A$、第2オペランドが $B$」と「第1オペランドが $B$、第2オペランドが $A$」が神経系内で明確に区別できる必要があります。
   - もし同一の投射神経群を共有してしまうと、入力段階で可換性が自明になってしまい、キノコ体（KC）および MBON が「学習によって可換性を獲得したのか」それとも「入力表現が同じだったから正解できたのか」を分離できなくなります。
   - Split-PN 方式により、$2 + 7$ と $7 + 2$ は完全に直交に近い PN 発火パターンとして入力されます。

2. **既存コードへの低侵襲性と計算効率**:
   - `flybrain.wasm`（C言語コア）には一切手を加える必要がありません。
   - JavaScript 側の `makeAdditionReader` において、`chanLeft` (144ピクセル → 左342本の ALPN) と `chanRight` (144ピクセル → 右343本の ALPN) をあらかじめ構築しておき、`stimulate(imgLeft, imgRight)` で同時にポアソン刺激を印加するだけで動作します。

3. **スパース発火特性の維持**:
   - 144ピクセルに対して 342〜343本の ALPN を割り当てるため、各ピクセルに平均約 2.38本の ALPN が割り当てられます。
   - 手書き数字の平均黒画素数（約 20〜30 ピクセル）に対し、約 50〜70 本の ALPN が左右それぞれで活性化し、全体で約 100〜140 本の ALPN が発火します。
   - これはキノコ体固有ニューロン（KC）の健全なスパース発火（全体の約 5〜10%）を誘発するのに極めて適した活性化レベルです。

---

## 3. 具体的な配分アルゴリズム

```javascript
// 685本の ALPN を固定シードでシャッフル
const deck = [...ALPN];
shuffle(deck, seed = 12345);

// 左右に等分
const half = Math.floor(deck.length / 2); // 342
const leftPNs = deck.slice(0, half);     // 342本
const rightPNs = deck.slice(half);       // 343本

// 144ピクセルにラウンドロビン配分
const chanLeft = Array.from({ length: 144 }, () => []);
leftPNs.forEach((p, i) => chanLeft[i % 144].push(p));

const chanRight = Array.from({ length: 144 }, () => []);
rightPNs.forEach((p, i) => chanRight[i % 144].push(p));
```

刺激印加時：
```javascript
function stimulate(imgLeft, imgRight) {
  brain.clearStimuli();
  for (let i = 0; i < 144; i++) {
    if (imgLeft[i] > C.ink) brain.stimulate(chanLeft[i], C.hz * imgLeft[i], { byIndex: true });
    if (imgRight[i] > C.ink) brain.stimulate(chanRight[i], C.hz * imgRight[i], { byIndex: true });
  }
}
```
