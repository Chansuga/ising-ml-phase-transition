# 参考文献

## 再現した論文

1. J. Carrasquilla and R. G. Melko, "Machine learning phases of matter," *Nature Physics* **13**, 431–434 (2017).
   [arXiv:1605.01735](https://arxiv.org/abs/1605.01735)
2. A. Tanaka and A. Tomiya, "Detection of Phase Transition via Convolutional Neural Networks," *J. Phys. Soc. Jpn.* **86**, 063001 (2017).
   [arXiv:1609.09087](https://arxiv.org/abs/1609.09087)
3. K. Kashiwa, Y. Kikuchi and A. Tomiya, "Phase transition encoded in neural network," *Prog. Theor. Exp. Phys.* **2019**, 083A04 (2019).
   [arXiv:1812.01522](https://arxiv.org/abs/1812.01522)

## 実装で参考にしたもの

- 田中章詞・富谷昭夫・橋本幸士『ディープラーニングと物理学 原理がわかる、応用ができる』講談社 (2019)
  - 熱浴法による配位生成（`01_carrasquilla_melko/gen_data*.ipynb`）と相の分類器の構成。書籍では TensorFlow を使っているため、PyTorch で書き直した
- [マルコフ連鎖モンテカルロ法でイジング模型をシミュレーション](https://dreamer-uma.com/mcmc-ising/)（dreamer-uma.com）
  - メトロポリス法による配位生成（`02_*/Ising_samp.ipynb`, `03_*/Ising_samp.ipynb`）
