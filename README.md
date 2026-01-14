# TASO NNUE USI Proxy

NNUE系エンジン（USI）をラップして、bestmove の強さを歪めずに、対局・検討で読みやすい「状況表示」と「候補の配膳」を行う Python 製プロキシです。


## 特徴

- USIプロキシとして動作（GUI/クライアント側からはUSIエンジンに見える）
- 表示の整理
  - stance（ADV/EVEN/DEFICIT/CRISIS）
  - intent（MAIN / BULL(ATK) / HEDGE(DEF)）
- 安全フィルタ
  - 既知の自玉詰み（mate負）がPVに見えている候補の回避
  - post-move の safety_check（短時間で相手mate+を検出したら代案へ）
- WATCH/ANALYZE モード
  - WATCH: 少数候補を読みやすく
  - ANALYZE: 候補の詳細指標も表示（ただしbestmoveは歪めない）

---

##UI表示の見方（TASO表示）

1) stance=…

現在局面の大まかな形勢（雰囲気）です。
	•	WIN: 解析上「勝ち筋が見えている」状態（詰みが見える/相当な優勢の目安）
	•	ADV: 優勢
	•	EVEN: 互角
	•	DEFICIT: 劣勢
	•	CRISIS: 危機（受けを最優先で探す局面）

※ stance は瞬間の評価値のブレをならして出しているので、急にフラつきにくい設計です。

⸻

2) intent=本筋 / BULL / HEDGE（…reason…）

「この手番で、どの“線”を太くして読ませるか」の選択です（bestmove自体は基本そのまま）。
	•	本筋（MAIN）: もっとも素直で、全体として納得感の高い進行を代表する線
	•	BULL: 前向きな手。局面を動かしにいく／主導権を作りにいく方向の線
	•	HEDGE: 手厚い手。崩れにくさ・安全性を重視する方向の線

括弧内の reason は「なぜその intent を維持/切替したか」の簡単な理由です（keep / switch / CRISIS規則など）。

⸻

3) △=…（圧縮CP）

△= は “圧縮したCP表示” です。
エンジンの生CP（score cp）をそのまま出さず、不確かさ（候補の散り方・PVの割れ方）に応じて控えめに丸めた値を出します。
	•	△=mate: 詰みが見えている（またはそれに準ずる）状態
	•	△=+xxx / -xxx: 圧縮された形勢の目安
	•	数字が大きいほど、（その局面評価が）安定している・確信が高い傾向
	•	数字が小さめでも、生CPが低いとは限らない（“不確かだから控えめに出してる”場合がある）

※ これは「人間が見て振り回されない」ための表示で、内部計算の強さを歪める目的ではありません。

⸻

4) flow=↑/→/↓ + 数値

flow= は 局面評価の“流れ（トレンド）” です。
直近の評価推移（ならした変化量）を出します。
	•	↑: 改善傾向（形勢が良くなっている方向）
	•	→: 横ばい
	•	↓: 悪化傾向（形勢が悪くなっている方向）
	•	後ろの +/-数値 はその変化量（どれくらい動いたかの目安）

⸻

5) 候補表示（WATCH/ANALYZE）

候補には tag= が付きます。
	•	tag=本筋: 本筋枠の候補
	•	tag=BULL: 前向きな候補（局面を動かしにいく）
	•	tag=HEDGE: 手厚い候補（崩れにくさ重視）

加えて、候補ごとに
	•	score=…: 表示用の総合スコア（見せる順のための尺度）
	•	cp=…: エンジンの生CP（mate-only のときもある）
	•	Δ=…: best候補からの差分（どれくらい落ちる/近いか）

が出ます。

⸻

6) CRISIS時の注意表示

stance=CRISIS なのに HEDGE（受け筋）候補が見つからない場合は、
	•	⚠ CRISIS: DEF候補なし（受け筋なし）

が出ます。これは「受けを最優先で探したが、候補タグとして成立する受け筋を立てられなかった」合図です。

⸻


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
