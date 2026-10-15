# 03. ニューラルネットワークは何を学んでいるか（Kashiwa, Kikuchi & Tomiya, 2019）

温度測定器の隠れ層を3ユニットまで絞って、学習済みの重みを解析します。全結合ネットワーク（FCN）は磁化を、畳み込みネットワーク（CNN）は内部エネルギーをとらえていることを確かめます。

- 元論文：K. Kashiwa, Y. Kikuchi and A. Tomiya, *Prog. Theor. Exp. Phys.* **2019**, 083A04 (2019), [arXiv:1812.01522](https://arxiv.org/abs/1812.01522)

## ファイル

| ファイル | 内容 |
|---|---|
| `Ising_samp.ipynb` | メトロポリス法で 16×16 の配位を生成（β = 0.05〜1.0 を 20 等分、各 2000 配位） |
| `temp_measure.ipynb` | FCN（256 → 3 → 20）/ CNN（畳み込み3層 → 20）で温度測定器を学習し、重みを保存 |
| `prm_analysis_FCNN.ipynb` | FCN の重みを解析。隠れユニットの出力と磁化の関係を調べる |
| `prm_analysis_CNN.ipynb` | CNN の重みを解析。隠れユニットの出力と内部エネルギーの関係を調べる |
| `onsager.ipynb` | オンサーガーの厳密解を数値積分し、エネルギーの温度依存性を求める（重みの理論曲線と比較するため） |
| `data/*.pth` | 学習済みの重み |

## 実行手順

1. データを生成します（約 4 万配位、約 80 MB）。
   ```bash
   python scripts/generate_data.py kashiwa   # data/Ising_data_L16_v3.pkl を作る
   ```
   ノートブックは `_v3` の付いたファイル名でデータを読み込みます。`Ising_samp.ipynb` で生成する場合は、保存するファイル名を `Ising_data_L16_v3.pkl` に変えてください。
2. `temp_measure.ipynb` の先頭で、モデル（`'FCN'` / `'CNN'`）と隠れ層のユニット数を選んで実行します。
3. `prm_analysis_*.ipynb` で重みを解析します。磁化やエネルギーの計算に配位データを使うので、手順 1 が必要です。

## 出力される図

| ファイル | 内容 |
|---|---|
| `image/L16_{FCN,CNN}_Nh3_graph.png` | 学習曲線 |
| `image/L16_{FCN,CNN}_Nh3_wight*.png` | 最後の層の重み（ヒートマップと温度依存性） |
| `image/L16_{FCN,CNN}_Nh3_output.png` | 隠れユニットの出力 |
| `image/L16_CNN_Nh3_energy.png` | CNN の隠れユニットの出力と内部エネルギーの比較 |
| `image/L16_FCN_Nh80_*.png` | 隠れ層 80 ユニットの FCN の重み |
