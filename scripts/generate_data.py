"""各実験ノートブックが読み込むスピン配位データを生成するスクリプト.

元の生成ノートブック (gen_data.ipynb, gen_data_tri.ipynb, Ising_samp.ipynb) と
同じアルゴリズム・パラメータ・ループ構造で配位を作り, 同じファイル名・形式で保存する.
純 Python のままでは数日かかるため, 1 スイープ分の更新だけを Numba で JIT コンパイルしている.
乱数列は元の実行と異なるので, 統計的に同等なデータになる (ビット単位では一致しない).

使い方 (リポジトリのルートで実行):
    python scripts/generate_data.py all
    python scripts/generate_data.py carrasquilla
    python scripts/generate_data.py tanaka --lattice-sizes 8 16
    python scripts/generate_data.py kashiwa
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
from numba import njit
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(ROOT, "experiments")


# ---------------------------------------------------------------------------
# 01: Carrasquilla & Melko (熱浴法, 正方格子 / 三角格子, L=32)
#     gen_data.ipynb / gen_data_tri.ipynb の heatbath() と同じ処理
# ---------------------------------------------------------------------------
@njit(cache=True)
def _heatbath_sweep(sc, beta, hz, triangular):
    Lx, Ly = sc.shape
    xlist = np.random.permutation(Lx)
    ylist = np.random.permutation(Ly)
    for x in xlist:
        for y in ylist:
            xu = (x + 1) % Lx
            xd = (x - 1) % Lx
            yu = (y + 1) % Ly
            yd = (y - 1) % Ly
            h = sc[xu, y] + sc[xd, y] + sc[x, yu] + sc[x, yd]
            if triangular:
                # 左上 (x-1, y+1) と右下 (x+1, y-1) のサイトを加える
                h += sc[xd, yu] + sc[xu, yd]
            h = h - hz
            p = np.exp(beta * h) / (np.exp(beta * h) + np.exp(-beta * h))
            if np.random.random() < p:
                sc[x, y] = 1
            else:
                sc[x, y] = -1


def _load_prm(name):
    """01_carrasquilla_melko/prm.py からパラメータリストを読み込む."""
    sys.path.insert(0, os.path.join(EXP_DIR, "01_carrasquilla_melko"))
    import prm

    return getattr(prm, name)


def generate_carrasquilla(seed):
    out_dir = os.path.join(EXP_DIR, "01_carrasquilla_melko")
    L = (32, 32)
    hz = -0.0005  # 相転移を見るために非常に弱い磁場を添加
    for lattice, prm_name, subdir in [
        ("square", "prm_list", "conf"),
        ("triangular", "prm_list_tri", "conf_tri"),
    ]:
        np.random.seed(seed)
        _seed_numba(seed)
        os.makedirs(os.path.join(out_dir, "data", subdir), exist_ok=True)
        prm_list = _load_prm(prm_name)
        sc = np.ones(L, dtype=np.int64)  # cold start
        for beta, n_sweep, n_discard, fname, save_every in tqdm(prm_list, desc=lattice):
            conf_cnt = 0
            for isweep in range(n_sweep):
                _heatbath_sweep(sc, beta, hz, lattice == "triangular")
                if isweep % save_every == 0 and isweep >= n_discard:
                    np.save(os.path.join(out_dir, f"{fname}{conf_cnt}"), sc.copy())
                    conf_cnt += 1


# ---------------------------------------------------------------------------
# 02 / 03: Tanaka & Tomiya, Kashiwa, Kikuchi & Tomiya (メトロポリス法)
#     Ising_samp.ipynb の LatticeIsing_2D / IsingMC と同じ処理
# ---------------------------------------------------------------------------
@njit(cache=True)
def _metropolis_sweep(state, beta, J, h):
    """IsingMC.step_MH: N*N 回ランダムにサイトを選んで反転を試みる (state をその場で更新)."""
    N = state.shape[0]
    for _ in range(N * N):
        x = np.random.randint(N)
        y = np.random.randint(N)
        s = state[y, x]
        nb = state[y, (x + 1) % N] + state[y, (x - 1) % N] + state[(y + 1) % N, x] + state[(y - 1) % N, x]
        delta_energy = 2 * (J * s * nb + h * s)
        if delta_energy < 0:
            state[y, x] *= -1
        elif np.random.random() <= np.exp(-beta * delta_energy):
            state[y, x] *= -1


def run_mcmc(init_state, beta, n_step, J=1, h=0.0):
    """IsingMC.run_mcmc の state_dynamics[n_step] にあたる, 最後の配位のコピー (float64) を返す.

    元コードと同様に init_state をその場で書き換える (呼び出し側の配列も変化する).
    元コードは全ステップの配位を記録した配列 (n_step+1, N, N) を作り, その 1 枚をビューとして返していた.
    ビューを保持し続けると配列全体がメモリに残るため (kashiwa で約 40 GB), 最後の配位だけをコピーして返す.
    返す配列が次の呼び出しでその場で書き換えられる点は元コードと同じなので, 生成されるデータは変わらない.
    """
    state = init_state
    for _ in range(n_step):
        _metropolis_sweep(state, beta, J, h)
    return state.astype(np.float64)


@njit(cache=True)
def _seed_numba(seed):
    np.random.seed(seed)


def _save_pickle(spin_data, label_data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((spin_data, label_data), f)
    print(f"saved: {os.path.relpath(path, ROOT)} ({len(spin_data)} configs)")


def generate_tanaka(lattice_sizes, seed):
    """02_tanaka_tomiya/Ising_samp.ipynb: β=0.2〜1.0 を 100 等分, 各 100 配位."""
    n_step_first, n_step_second = 1000, 500
    num_label, number = 100, 100
    for N in lattice_sizes:
        np.random.seed(seed)
        _seed_numba(seed)
        spin_data, label_data = [], []
        for i in tqdm(range(num_label), desc=f"tanaka L={N}"):
            beta = 0.2 + (i / (num_label - 1)) * 0.8
            init_state = np.full((N, N), -1)
            Ising_state = run_mcmc(init_state, beta, n_step_first)
            for _ in range(number):
                init_state = Ising_state
                Ising_state = run_mcmc(init_state, beta, n_step_second)
                if Ising_state.sum() > 0:
                    Ising_state *= -1
                spin_data.append(Ising_state)
                label_data.append(beta)
        _save_pickle(spin_data, label_data, os.path.join(EXP_DIR, "02_tanaka_tomiya", "data", f"Ising_data_L{N}.pkl"))


def generate_kashiwa(seed, N=16):
    """03_kashiwa_kikuchi_tomiya/Ising_samp.ipynb: β=0.05〜1.0 を 20 等分, 各 2000 配位."""
    n_step_first, n_step_second = 1000, 500
    num_label, number = 20, 2000
    np.random.seed(seed)
    _seed_numba(seed)
    spin_data, label_data = [], []
    for i in tqdm(range(num_label), desc=f"kashiwa L={N}"):
        beta = 0.05 + (i / (num_label - 1)) * 0.95
        init_state = np.full((N, N), -1)
        Ising_state = run_mcmc(init_state, beta, n_step_first)
        for _ in range(number):
            # IsingMC.flip: 1/2 の確率で配位全体の符号を反転
            init_state = -1 * Ising_state if np.random.choice([True, False]) else Ising_state
            Ising_state = run_mcmc(init_state, beta, n_step_second)
            spin_data.append(Ising_state)
            label_data.append(beta)
    # temp_measure.ipynb などは _v3 の付いたファイル名で読み込む
    _save_pickle(spin_data, label_data, os.path.join(EXP_DIR, "03_kashiwa_kikuchi_tomiya", "data", f"Ising_data_L{N}_v3.pkl"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", choices=["all", "carrasquilla", "tanaka", "kashiwa"])
    parser.add_argument("--lattice-sizes", type=int, nargs="+", default=[8, 16], help="tanaka で生成する格子サイズ")
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()

    start = time.time()
    if args.target in ("all", "carrasquilla"):
        generate_carrasquilla(args.seed)
    if args.target in ("all", "tanaka"):
        generate_tanaka(args.lattice_sizes, args.seed)
    if args.target in ("all", "kashiwa"):
        generate_kashiwa(args.seed)
    print(f"done in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
