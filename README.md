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

対応OS
	•	macOS 12+（Apple Silicon / Intel どちらもOK）
	•	Windows 10/11（64-bit）
	•	Linux（x86_64 / aarch64）

必須ソフト
	•	Python 3.10+（推奨: 3.11+）
	•	USI対応将棋エンジン（例: YaneuraOu系）
	•	NNUE評価ファイル（エンジンが要求する形式一式）
	•	シェル実行環境（macOS/Linux: bash/zsh、Windows: PowerShell推奨）

CPU

Minimum
	•	2コア / 4スレッド以上（例: ノートPC標準クラス）

Recommended
	•	6コア / 12スレッド以上
	•	TASO_THREADS を実コア〜論理コア範囲で調整推奨
	•	TASO_MULTIPV を増やすほどCPU負荷が増える

メモリ（重要）

このプロジェクトで一番現実的に効いてくるのがここ。

Minimum
	•	8GB RAM
	•	TASO_HASH_MB を 512〜1024 に落とす前提（デフォルト 4096 は厳しい場合あり）

Recommended
	•	16GB RAM（快適）
	•	TASO_HASH_MB=2048〜4096 が現実的
	•	MultiPVを5以上にするなら 16GB あると安心

Heavy / Analyze 推奨
	•	32GB RAM
	•	MODE=ANALYZE ＋ MULTIPV=8〜10 ＋ HASH=4096+ を狙うならここ

ストレージ
	•	空き 1GB 以上（エンジン＋NNUE＋book等）
	•	推奨: SSD（体感の安定性が上がる）

画面・UI
	•	UIはUSI出力（標準出力）なので、GUIは不要
	•	対局GUI（ShogiGUI / 将棋所 / Kento 等）側の表示品質に依存

パフォーマンス目安（ざっくり）
	•	MODE=WATCH, MULTIPV=3〜5, THREADS=4〜8：一般的なノートでも運用可
	•	MODE=ANALYZE, MULTIPV=8〜10：CPUとRAMが素直に必要（特にRAM）

推奨の初期設定（初回にコケにくい）
	•	TASO_THREADS: 実環境の論理コア数の 50〜100%（発熱が気になるなら 50%）
	•	TASO_HASH_MB: 16GB RAMなら 2048〜4096 / 8GBなら 512〜1024
	•	TASO_MULTIPV: WATCH=5、ANALYZE=8 目安
	•	TASO_SAFETY_MS: 40〜90（安全寄りにするほど重くなる）


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
