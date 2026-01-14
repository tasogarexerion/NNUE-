# TASO NNUE USI Proxy

YaneuraOu系エンジン（USI）をラップして、bestmove の強さを歪めずに、対局・検討で読みやすい「状況表示」と「候補の配膳」を行う Python 製プロキシです。

- bestmove は原則そのまま（安全フィルタのみ）
- 表示は少語で、判断の“線”を作る（stance / game phase / intent）
- WATCHでは「本筋 + BULL(ATK) + HEDGE(DEF)」を中心に候補を提示

---

## 特徴

- USIプロキシとして動作（GUI/クライアント側からはUSIエンジンに見える）
- 表示の整理
  - game phase（BUILD→PROBE→TENSION→CLASH→CONVERT→FINISH）
  - stance（ADV/EVEN/DEFICIT/CRISIS）
  - intent（MAIN / BULL(ATK) / HEDGE(DEF)）
- 安全フィルタ
  - 既知の自玉詰み（mate負）がPVに見えている候補の回避
  - post-move の safety_check（短時間で相手mate+を検出したら代案へ）
- WATCH/ANALYZE モード
  - WATCH: 少数候補を読みやすく
  - ANALYZE: 候補の詳細指標も表示（ただしbestmoveは歪めない）

---

## 動作要件

- Python 3.9+（目安）
- USI対応の将棋エンジン（例: YaneuraOu）
- NNUE評価ファイル（エンジン側が必要とする形式）

---

## 使い方（基本）

1. エンジンと同じように起動できる場所に置く  
   例:
   - taso_proxy.py（このプロキシ）
   - ./YaneuraOu（エンジン本体）
   - ./eval/（NNUEフォルダ）

2. 環境変数でエンジン/NNUEパスを指定（任意）

3. GUIから「USIエンジン」としてこのプロキシを登録  
   GUI側のエンジン実行ファイルに taso_proxy.py を指定します。

---

## 環境変数

- TASO_ENGINE : エンジンへのパス（デフォルト ./YaneuraOu）
- TASO_NNUE : NNUEフォルダ（デフォルト eval）
- TASO_MODE : PLAY / WATCH / ANALYZE（デフォルト WATCH）
- TASO_MULTIPV : MultiPV数（デフォルト 3）
- TASO_SAFETY_MS : safety_check用の短時間探索(ms)

ほか多数あります（ソース冒頭の CONFIG を参照）。

---

## 設計ポリシー（重要）

- bestmove は強さを歪めない  
  表示のためのスコアは人間向けの説明であり、bestmove選択の原則は維持します（安全フィルタのみ例外）。
- UIは少語  
  たくさん喋るより、毎手の「状況」と「意図の線」を安定して見せることを重視しています。

---

## 免責事項

- 本プロジェクトは研究・学習・個人利用を主目的としています。
- 対局結果や性能についていかなる保証もしません。
- 使用するエンジンや評価関数（NNUE）のライセンスは各自で遵守してください。

---

## ライセンス

このリポジトリ（プロキシ本体）は MIT License です。詳細は LICENSE を参照してください。
