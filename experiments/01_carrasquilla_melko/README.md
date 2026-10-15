# 01. 相の分類器（Carrasquilla & Melko, 2017）

スピン配位が強磁性相か常磁性相かを、ニューラルネットワークで2値分類します。さらに、正方格子で学習したモデルが三角格子の相転移も検出できるかを調べます。

- 元論文：J. Carrasquilla and R. G. Melko, *Nature Physics* **13**, 431 (2017), [arXiv:1605.01735](https://arxiv.org/abs/1605.01735)

## ファイル

| ファイル | 内容 |
|---|---|
| `gen_data.ipynb` | 熱浴法で 32×32 正方格子の配位を生成し、`data/conf/` に保存 |
| `gen_data_tri.ipynb` | 同じく三角格子の配位を生成し、`data/conf_tri/` に保存 |
| `prm.py` | 生成する逆温度と保存ファイル名の一覧（`train.ipynb` から読み込む） |
| `train.ipynb` | FCN / CNN で学習し、各逆温度での出力確率をプロットして `image/` に保存 |
| `data/model_square_CNN.pth` | 正方格子で学習した CNN の重み |

## 実行手順

1. データを生成します。ノートブックは純 Python なので数十分かかります。高速版のスクリプトなら数秒で終わります。
   ```bash
   python scripts/generate_data.py carrasquilla   # リポジトリのルートで実行
   ```
2. `train.ipynb` の先頭のセルで、格子（`'square'` / `'triangular'`）とモデル（`'FCN'` / `'CNN'`）を選んで実行します。
3. 格子を `'square'` にした場合は、最後のセルで、学習したモデルに三角格子の配位を入力した結果も出力されます。

## 出力される図

| ファイル | 内容 |
|---|---|
| `image/plot_{square,triangular}_{FCN,CNN}*.png` | 逆温度ごとの出力確率。破線は厳密な臨界点 |
| `image/plot_square_*_tridata.png` | 正方格子で学習したモデルに三角格子の配位を入力した結果 |
| `image/graph_*.png` | 学習曲線（損失と正解率） |

## 既知の問題（当時のコードのまま残しています）

- `gen_data_tri.ipynb` と `prm.py` では、β = 0.54 と β = 0.51 の保存先がどちらも `L32b090_` になっています。そのため β = 0.54 のデータは β = 0.51 のデータで上書きされ、学習では β = 0.51 のデータを2回読み込んでいます。三角格子のファイル名の数字（`b090` など）は、実際の β とは対応していません。
