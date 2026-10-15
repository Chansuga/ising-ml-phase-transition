"""各実験ノートブックが読み込むスピン配位データを生成するスクリプト.

元の生成ノートブック (gen_data.ipynb, gen_data_tri.ipynb, Ising_samp.ipynb) と
同じアルゴリズム・パラメータ・ループ構造で配位を作り, 同じファイル名・形式で保存する.
純 Python のままでは数日かかるため, 1 スイープ分の更新だけを Numba で JIT コンパイルしている.
乱数列は元の実行と異なるので, 統計的に同等なデータになる (ビット単位では一致しない).
オプションを指定しなければ, 元のノートブックと同じパラメータで生成する.

使い方 (リポジトリのルートで実行):
    python scripts/generate_data.py all
    python scripts/generate_data.py carrasquilla
    python scripts/generate_data.py tanaka --lattice-sizes 8 16
    python scripts/generate_data.py kashiwa

パラメータを変える例 (設定できる項目は `python scripts/generate_data.py tanaka -h` などで確認):
    python scripts/generate_data.py tanaka --lattice-sizes 32 --n-beta 50 --out-dir /tmp/tanaka_L32
    python scripts/generate_data.py kashiwa --configs-per-beta 500 --seed 1 --out-dir /tmp/kashiwa_small

ノートブックから使う例 (ファイルに保存せず配位を受け取る):
    import sys; sys.path.insert(0, "../../scripts")
    from generate_data import KASHIWA, sample_metropolis
    spin_data, label_data = sample_metropolis(16, KASHIWA.replace(configs_per_beta=100), random_flip=True)
"""

import argparse
import dataclasses
import importlib.util
import os
import pickle
import time

import numpy as np
from numba import njit
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(ROOT, "experiments")
CARRASQUILLA_DIR = os.path.join(EXP_DIR, "01_carrasquilla_melko")
TANAKA_DATA_DIR = os.path.join(EXP_DIR, "02_tanaka_tomiya", "data")
KASHIWA_DATA_DIR = os.path.join(EXP_DIR, "03_kashiwa_kikuchi_tomiya", "data")

DEFAULT_SEED = 777


@njit(cache=True)
def _seed_numba(seed):
    np.random.seed(seed)


def _seed_all(seed):
    """NumPy と Numba の乱数生成器は別々なので, 両方に同じ seed を設定する."""
    np.random.seed(seed)
    _seed_numba(seed)


# ---------------------------------------------------------------------------
# 01: Carrasquilla & Melko (熱浴法, 正方格子 / 三角格子, L=32)
#     gen_data.ipynb / gen_data_tri.ipynb の heatbath() と同じ処理
# ---------------------------------------------------------------------------
CARRASQUILLA_LATTICE_SIZE = (32, 32)
CARRASQUILLA_HZ = -0.0005  # 相転移を見るために非常に弱い磁場を添加

# 格子の種類 -> (prm.py のパラメータリスト名)
# 逆温度・スイープ数・保存ファイル名は train.ipynb と共通の prm.py で決まる
CARRASQUILLA_PRM_NAMES = {
    "square": "prm_list",
    "triangular": "prm_list_tri",
}


@njit(cache=True)
def _heatbath_sweep(spins, beta, hz, triangular):
    Lx, Ly = spins.shape
    x_order = np.random.permutation(Lx)
    y_order = np.random.permutation(Ly)
    for x in x_order:
        for y in y_order:
            xu = (x + 1) % Lx
            xd = (x - 1) % Lx
            yu = (y + 1) % Ly
            yd = (y - 1) % Ly
            local_field = spins[xu, y] + spins[xd, y] + spins[x, yu] + spins[x, yd]
            if triangular:
                # 左上 (x-1, y+1) と右下 (x+1, y-1) のサイトを加える
                local_field += spins[xd, yu] + spins[xu, yd]
            local_field = local_field - hz
            prob_up = np.exp(beta * local_field) / (np.exp(beta * local_field) + np.exp(-beta * local_field))
            if np.random.random() < prob_up:
                spins[x, y] = 1
            else:
                spins[x, y] = -1


def _load_prm_list(name):
    """01_carrasquilla_melko/prm.py からパラメータリストを読み込む."""
    spec = importlib.util.spec_from_file_location("prm", os.path.join(CARRASQUILLA_DIR, "prm.py"))
    prm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prm)
    return getattr(prm, name)


def generate_carrasquilla(lattices=("square", "triangular"), hz=CARRASQUILLA_HZ, seed=DEFAULT_SEED, out_dir=CARRASQUILLA_DIR):
    """熱浴法で配位を作り, 1 配位ずつ .npy で保存する.

    保存先は out_dir に prm.py のファイル名 (data/conf/L32b090_ など) をつなげたもの.
    """
    for lattice in lattices:
        _seed_all(seed)
        prm_list = _load_prm_list(CARRASQUILLA_PRM_NAMES[lattice])
        spins = np.ones(CARRASQUILLA_LATTICE_SIZE, dtype=np.int64)  # cold start
        for beta, n_sweep, n_discard, path_prefix, save_every in tqdm(prm_list, desc=lattice):
            os.makedirs(os.path.dirname(os.path.join(out_dir, path_prefix)), exist_ok=True)
            n_saved = 0
            for sweep in range(n_sweep):
                _heatbath_sweep(spins, beta, hz, lattice == "triangular")
                if sweep % save_every == 0 and sweep >= n_discard:
                    np.save(os.path.join(out_dir, f"{path_prefix}{n_saved}"), spins.copy())
                    n_saved += 1


# ---------------------------------------------------------------------------
# 02 / 03: Tanaka & Tomiya, Kashiwa, Kikuchi & Tomiya (メトロポリス法)
#     Ising_samp.ipynb の LatticeIsing_2D / IsingMC と同じ処理
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class MetropolisSettings:
    """メトロポリス法で配位を作るときのパラメータ."""

    beta_min: float
    beta_max: float
    n_beta: int  # 逆温度ラベルの数 (beta_min〜beta_max を等間隔に分ける)
    configs_per_beta: int  # 1 つの逆温度で保存する配位の数
    n_therm: int = 1000  # 各逆温度で最初に捨てるスイープ数 (元コードの n_step_first)
    n_interval: int = 500  # 保存する配位どうしの間のスイープ数 (元コードの n_step_second)

    def __post_init__(self):
        if self.n_beta < 2:
            raise ValueError("n_beta は 2 以上にしてください")

    def betas(self):
        """元コードと同じ式で逆温度ラベルを作る (np.linspace とは末尾の桁が異なることがある)."""
        return [self.beta_min + (i / (self.n_beta - 1)) * (self.beta_max - self.beta_min) for i in range(self.n_beta)]

    def replace(self, **changes):
        return dataclasses.replace(self, **changes)


# 02_tanaka_tomiya/Ising_samp.ipynb: β=0.2〜1.0 を 100 等分, 各 100 配位
TANAKA = MetropolisSettings(beta_min=0.2, beta_max=1.0, n_beta=100, configs_per_beta=100)
TANAKA_LATTICE_SIZES = (8, 16)

# 03_kashiwa_kikuchi_tomiya/Ising_samp.ipynb: β=0.05〜1.0 を 20 等分, 各 2000 配位
KASHIWA = MetropolisSettings(beta_min=0.05, beta_max=1.0, n_beta=20, configs_per_beta=2000)
KASHIWA_LATTICE_SIZE = 16


@njit(cache=True)
def _metropolis_sweep(state, beta, J, h):
    """IsingMC.step_MH: L*L 回ランダムにサイトを選んで反転を試みる (state をその場で更新)."""
    L = state.shape[0]
    for _ in range(L * L):
        x = np.random.randint(L)
        y = np.random.randint(L)
        spin = state[y, x]
        neighbor_sum = state[y, (x + 1) % L] + state[y, (x - 1) % L] + state[(y + 1) % L, x] + state[(y - 1) % L, x]
        delta_energy = 2 * (J * spin * neighbor_sum + h * spin)
        if delta_energy < 0:
            state[y, x] *= -1
        elif np.random.random() <= np.exp(-beta * delta_energy):
            state[y, x] *= -1


def run_mcmc(init_state, beta, n_step, J=1, h=0.0):
    """IsingMC.run_mcmc の state_dynamics[n_step] にあたる, 最後の配位のコピー (float64) を返す.

    元コードと同様に init_state をその場で書き換える (呼び出し側の配列も変化する).
    元コードは全ステップの配位を記録した配列 (n_step+1, L, L) を作り, その 1 枚をビューとして返していた.
    ビューを保持し続けると配列全体がメモリに残るため (kashiwa で約 40 GB), 最後の配位だけをコピーして返す.
    返す配列が次の呼び出しでその場で書き換えられる点は元コードと同じなので, 生成されるデータは変わらない.
    """
    state = init_state
    for _ in range(n_step):
        _metropolis_sweep(state, beta, J, h)
    return state.astype(np.float64)


def sample_metropolis(L, settings, seed=DEFAULT_SEED, *, force_negative=False, random_flip=False, desc=None):
    """各逆温度で配位を作り, (配位のリスト, 逆温度ラベルのリスト) を返す.

    force_negative: 配位の磁化が正なら符号を反転する (02 tanaka の処理)
    random_flip:    次の配位を作り始める前に, 1/2 の確率で符号を反転する (03 kashiwa の IsingMC.flip)

    元コードと同じく, リストに入れた配位を次の run_mcmc の初期配位としてその場で書き換える.
    そのため 02 では符号の反転が各 β の最後の配位にしか効かず, 03 では配位の一部が重複する
    (各実験の README の「既知の問題」を参照).
    """
    _seed_all(seed)
    spin_data, label_data = [], []
    for beta in tqdm(settings.betas(), desc=desc):
        state = run_mcmc(np.full((L, L), -1), beta, settings.n_therm)
        for _ in range(settings.configs_per_beta):
            if random_flip and np.random.choice([True, False]):
                init_state = -1 * state
            else:
                init_state = state
            state = run_mcmc(init_state, beta, settings.n_interval)
            if force_negative and state.sum() > 0:
                state *= -1
            spin_data.append(state)
            label_data.append(beta)
    return spin_data, label_data


def _save_pickle(spin_data, label_data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((spin_data, label_data), f)
    print(f"saved: {os.path.relpath(path, ROOT)} ({len(spin_data)} configs)")


def generate_tanaka(lattice_sizes=TANAKA_LATTICE_SIZES, settings=TANAKA, seed=DEFAULT_SEED, out_dir=TANAKA_DATA_DIR):
    """格子サイズごとに out_dir/Ising_data_L{L}.pkl を作る."""
    for L in lattice_sizes:
        spin_data, label_data = sample_metropolis(L, settings, seed, force_negative=True, desc=f"tanaka L={L}")
        _save_pickle(spin_data, label_data, os.path.join(out_dir, f"Ising_data_L{L}.pkl"))


def generate_kashiwa(L=KASHIWA_LATTICE_SIZE, settings=KASHIWA, seed=DEFAULT_SEED, out_dir=KASHIWA_DATA_DIR):
    """out_dir/Ising_data_L{L}_v3.pkl を作る (temp_measure.ipynb などは _v3 の付いたファイル名で読み込む)."""
    spin_data, label_data = sample_metropolis(L, settings, seed, random_flip=True, desc=f"kashiwa L={L}")
    _save_pickle(spin_data, label_data, os.path.join(out_dir, f"Ising_data_L{L}_v3.pkl"))


# ---------------------------------------------------------------------------
# コマンドライン
# ---------------------------------------------------------------------------
def _add_metropolis_args(parser, defaults):
    g = parser.add_argument_group("メトロポリス法のパラメータ")
    g.add_argument("--beta-min", type=float, default=defaults.beta_min, help="最小の逆温度 (default: %(default)s)")
    g.add_argument("--beta-max", type=float, default=defaults.beta_max, help="最大の逆温度 (default: %(default)s)")
    g.add_argument("--n-beta", type=int, default=defaults.n_beta, help="逆温度ラベルの数 (default: %(default)s)")
    g.add_argument("--configs-per-beta", type=int, default=defaults.configs_per_beta, help="1 つの逆温度で作る配位の数 (default: %(default)s)")
    g.add_argument("--n-therm", type=int, default=defaults.n_therm, help="最初に捨てるスイープ数 (default: %(default)s)")
    g.add_argument("--n-interval", type=int, default=defaults.n_interval, help="保存する配位の間のスイープ数 (default: %(default)s)")


def _metropolis_settings(args):
    return MetropolisSettings(
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        n_beta=args.n_beta,
        configs_per_beta=args.configs_per_beta,
        n_therm=args.n_therm,
        n_interval=args.n_interval,
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--seed", type=int, default=DEFAULT_SEED, help="乱数の seed (default: %(default)s)")
    sub = parser.add_subparsers(dest="target", required=True, metavar="{all,carrasquilla,tanaka,kashiwa}")

    p = sub.add_parser("all", parents=[common], help="3 つの実験のデータを既定の設定で生成する")
    p.add_argument("--lattice-sizes", type=int, nargs="+", default=list(TANAKA_LATTICE_SIZES), help="tanaka で生成する格子サイズ")

    p = sub.add_parser("carrasquilla", parents=[common], help="01: 熱浴法で正方格子・三角格子の配位を生成する")
    p.add_argument("--lattices", nargs="+", choices=list(CARRASQUILLA_PRM_NAMES), default=list(CARRASQUILLA_PRM_NAMES), help="生成する格子の種類 (default: 両方)")
    p.add_argument("--hz", type=float, default=CARRASQUILLA_HZ, help="外部磁場 (default: %(default)s)")
    p.add_argument("--out-dir", default=CARRASQUILLA_DIR, help="この下に prm.py のファイル名 (data/conf/...) で保存する (default: 実験フォルダ)")

    p = sub.add_parser("tanaka", parents=[common], help="02: メトロポリス法で温度測定器の学習データを生成する")
    p.add_argument("--lattice-sizes", type=int, nargs="+", default=list(TANAKA_LATTICE_SIZES), help="格子サイズ (default: 8 16)")
    p.add_argument("--out-dir", default=TANAKA_DATA_DIR, help="Ising_data_L{L}.pkl の保存先 (default: 実験の data フォルダ)")
    _add_metropolis_args(p, TANAKA)

    p = sub.add_parser("kashiwa", parents=[common], help="03: メトロポリス法で温度測定器の学習データを生成する")
    p.add_argument("--lattice-size", type=int, default=KASHIWA_LATTICE_SIZE, help="格子サイズ (default: %(default)s)")
    p.add_argument("--out-dir", default=KASHIWA_DATA_DIR, help="Ising_data_L{L}_v3.pkl の保存先 (default: 実験の data フォルダ)")
    _add_metropolis_args(p, KASHIWA)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.target in ("tanaka", "kashiwa"):
        try:
            settings = _metropolis_settings(args)
        except ValueError as e:
            parser.error(str(e))

    start = time.time()
    if args.target == "all":
        generate_carrasquilla(seed=args.seed)
        generate_tanaka(args.lattice_sizes, seed=args.seed)
        generate_kashiwa(seed=args.seed)
    elif args.target == "carrasquilla":
        generate_carrasquilla(args.lattices, args.hz, args.seed, args.out_dir)
    elif args.target == "tanaka":
        generate_tanaka(args.lattice_sizes, settings, args.seed, args.out_dir)
    elif args.target == "kashiwa":
        generate_kashiwa(args.lattice_size, settings, args.seed, args.out_dir)
    print(f"done in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
