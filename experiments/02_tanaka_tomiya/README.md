# 02. 温度測定器による相転移の検出（Tanaka & Tomiya, 2017）

スピン配位から温度を当てるモデル（温度測定器）を学習します。学習した重みから、相転移の情報を与えずに臨界温度を推定します。

- 元論文：A. Tanaka and A. Tomiya, *J. Phys. Soc. Jpn.* **86**, 063001 (2017), [arXiv:1609.09087](https://arxiv.org/abs/1609.09087)

## ファイル

| ファイル | 内容 |
|---|---|
| `Ising_samp.ipynb` | メトロポリス法で配位を生成し、`data/Ising_data_L{L}.pkl` に保存（β = 0.2〜1.0 を 100 等分、各 100 配位） |
| `temp_measure.ipynb` | FCN / CNN で温度測定器を学習し、重みを `data/prm_data_L{L}_{FCN,CNN}.pth` に保存 |
| `prm_analysis_FCNN.ipynb` | FCN の最後の層の重みを温度ラベルごとに足し合わせ、tanh でフィットして臨界温度を推定 |
| `prm_analysis_CNN.ipynb` | CNN の畳み込みフィルタと最後の層の重みを可視化し、同様に臨界温度を推定 |
| `data/*.pth` | 学習済みの重み（論文の図を作ったもの） |

## 実行手順

1. 重みの解析だけを行う場合は、学習済みの重みが含まれているので、`prm_analysis_*.ipynb` をそのまま実行できます。
2. 学習からやり直す場合は、先にデータを生成します。
   ```bash
   python scripts/generate_data.py tanaka   # L=8 と L=16 を生成
   ```
   `temp_measure.ipynb` の先頭でモデルと格子サイズを選んで実行します。`num_epochs` は保存時点で 1 になっているので、必要に応じて増やしてください（コメントにある当時の設定は FCN が 1000、CNN が 20）。

## 出力される図

| ファイル | 内容 |
|---|---|
| `image/L{8,16}_{FCNN,CNN}_weight.png` | 最後の層の重みのヒートマップ |
| `image/L{8,16}_{FCNN,CNN}_weight_sum.png` | 温度ラベルごとの重みの和と tanh フィット |
| `image/L{8,16}_CNN_filter*.png` | CNN の畳み込みフィルタ |
| `image/spin_conf_*.png` | 生成したスピン配位の例 |

## 既知の問題（当時のコードのまま残しています）

- `Ising_samp.ipynb` には、配位の磁化が正なら符号を反転して負にそろえる処理があります。ところが、次のマルコフ連鎖が同じ配列をその場で書き換えるため、この処理が実際に効くのは各 β の最後の配位だけです。`scripts/generate_data.py` もこの動作をそのまま再現しています。
