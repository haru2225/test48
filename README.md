# test48 — DM2's own demo scripts (def-for-def unchanged), l≤5, on the SiO2 crystal

`train.py` / `generate.py` are DM2's own
[`demo/demo_training/denoiser_train_unconditional.py`](https://github.com/digital-synthesis-lab/DM2/blob/main/demo/demo_training/denoiser_train_unconditional.py)
and
[`demo/demo_generating/denoise_generate_unconditional.py`](https://github.com/digital-synthesis-lab/DM2/blob/main/demo/demo_generating/denoise_generate_unconditional.py),
with every function (`ase_graph`/`ase_graph_gpu`, `PeriodicStructureDataset`, `InitialEmbedding`,
`loss_fn`, `train`, `test`, `set_gpu`, `denoise_snapshot_with_noise_gpu`, `denoise_snapshot_gpu`)
copied **unchanged**. This is deliberately NOT test47.py's resumable/CLI-configurable rewrite —
it is DM2's own two-script workflow (`main()`'s configuration constants, not argparse), applied to
test47's SiO2 crystal (beta-cristobalite) instead of DM2's own 3000-atom glass, with exactly one
change requested: the NequIP irreps raised to l≤5 (test47's values, `64x0e + 32x1e + 16x2e + 8x3e
+ 4x4e + 2x5e` / edge `4x0e + 4x1e + 2x2e + 2x3e + 1x4e + 1x5e`; DM2's own default was l≤1 hidden /
l≤2 edge).

Only two things beyond the irreps were changed in `main()`:
- **Data paths**: `simu_data/reference_frames.npz` (test47's 18 beta-cristobalite thermal MD
  snapshots) in place of DM2's 3 glass `.dat` files; `inital_data/random_sio2_crystal_demo.data`
  (test48's own genuinely random 64 Si + 128 O configuration in the 13.573 Å cell, built the same
  crude way DM2's own `random_sio2_size_300_demo.dat` was verified to be: uniform random positions,
  no minimum-distance enforcement) in place of DM2's own random glass demo file.
- **`torch.serialization.add_safe_globals([slice])`** before importing `graphite`: not part of
  DM2's own def's, but needed on PyTorch ≥2.6 for e3nn 0.4.4's internal `torch.load` of its own
  Wigner-3j constants (same fix as test42–47).

Everything else in `main()` — `LARGE_CUTOFF=10`, `CUTOFF=5`, `BATCH_SIZE=16`, `NUM_UPDATES=30_000`,
`sigma_max_value=0.75`, the 2900-noisy + 100-polish generation schedule, `max_sigma_for_denoising
=1.0` — is DM2's own default, **not** test47's tuned values, exactly as requested.

## ⚠️ Known issue carried over unmodified (not fixed here, on purpose)

`LARGE_CUTOFF=10` is tuned for DM2's own ~36 Å glass box. test47's SiO2 crystal unit cell is only
13.573 Å across (half-box 6.79 Å), **smaller** than `LARGE_CUTOFF` — the same duplicate-periodic-
image bug test33's module docstring calls "fix #1" (same atom pair connected through 2+ periodic
images at once). test47.py works around this by setting `--large-cutoff 5.0`; test48 does not,
since "the exact same defs, only l≤5" was requested. If you want it fixed here too, say so.

## Other things carried over unmodified from DM2 (not bugs, just DM2's own choices)

- `torch.load(path_save_model)` (train.py) saves/loads the **whole model object** (pickled), not
  a `state_dict` — DM2's own convention, less portable than test47.py's checkpoint format.
- `generate.py`'s `main()` hardcodes `.to('cuda')` — will fail with no GPU, matching DM2's own
  script (no `--allow-cpu` escape hatch like test47.py has).
- No resumability, no time-budget guard — a run must fit inside one job's walltime.
- `from tqdm.notebook import trange` is imported but never actually called in either script (dead
  import, inherited as-is from DM2).
- Generation always starts from `TEST_FNAME` (`--init` is not a concept here); to compare against
  starting from the ideal crystal instead, point `TEST_FNAME` at
  `simu_data/silica_beta_cristobalite_init.data` (bundled) and re-run.

## スパコンでの実行（Singularity + DM2 をバインド、test45〜47 と同じ方式）

DM2(`graphite` パッケージ)はイメージに焼き込みません。ビルド時に GitHub へアクセスできない
サイトがあるため(Docker Hub / PyPI は通っても github.com だけ通らない、というケースを実際に
確認済み)、DM2 は別途クローンして実行時にバインドマウントします。

```bash
git clone https://github.com/haru2225/test48.git
git clone https://github.com/digital-synthesis-lab/DM2.git      # test48 と同じ階層に置く
cd test48

module load singularity
singularity build test48.sif Singularity.def
# Apptainer: apptainer build test48.sif Singularity.def
```

`Singularity.def` はもう GitHub にアクセスしません(Docker Hub と PyPI だけで完結します)。
これでも `singularity build` が失敗する場合:

```bash
# 1. Docker Hub / PyPI 自体がプロキシ経由でしか届かない場合
export https_proxy=http://proxy.example:8080 http_proxy=http://proxy.example:8080
singularity build test48.sif Singularity.def

# 2. apptainer で試す(HTTPクライアント実装が異なる)
apptainer build test48.sif Singularity.def

# 3. それでも失敗する場合: インターネットに出られる別のLinux環境でビルドしてから転送する
scp test48.sif your_cluster:/path/to/test48/
```

`DM2` 自体を github.com からクローンできないノードがある場合は、アクセスできる別のノード・
手元の PC で `git clone` するか zip を取得し、scp でスパコンへ転送してください。

```bash
qsub -P PROJECT_ID -v STAGE=train run_test48.pbs
# DM2 が ../DM2 以外にあるなら: qsub -P PROJECT_ID -v STAGE=train,DM2_ROOT=/abs/path/DM2 run_test48.pbs
# training complete後:
qsub -P PROJECT_ID -v STAGE=generate run_test48.pbs
```

## 動作確認状況

- `train.py`: CPU 上、極小設定（18 参照構造 x duplicate=2、4 更新）で最後まで完走を確認済み
  （モデル保存・損失グラフ出力まで到達）。
- `generate.py`: `.to('cuda')` 固定のため CPU では未確認。GPU での実行結果はまだありません。

出力: `test48_loss_figure.png`, `model/test48_model.pt`,
`gen_data/test48_denoised_random_sio2_crystal.extxyz`。DM2 本来の `main()` の設定値どおり、
途中経過ログはこの2つのスクリプトの標準出力にのみ出ます(ファイルには保存されません)。

`DM2` は MIT (Tim Hsu; Digital Synthesis Lab @ UCLA)。`simu_data/reference_frames.npz` の由来は
`simu_data/reference_frames_metadata.json` を参照。

## generate-v2.py — corrected generation (fixes an over-noised, structure-destroying sampler)

`generate.py`'s own generation loop (DM2's own demo code, kept unmodified) adds noise of magnitude
`sigma` itself at every one of 2900 annealing steps. Summed over the schedule (σ: 1.0 → 0.001),
that injected noise ALONE has a standard deviation of **~31 Å — more than twice the ~13.4 Å unit
cell** — regardless of how good the model is. Verified on a real, well-converged test47 run (same
network family, same generation algorithm; loss ~0.015): the generated structure had a mean
nearest-neighbor distance of 1.24 Å and a minimum of 0.24 Å (atoms on top of each other), with
completely flat bond/angle histograms, even starting from the ideal crystal. `generate.py` has the
identical issue; it is intentionally left as the DM2 port it is (see its own module docstring).

`generate-v2.py` fixes this with a properly SDE-consistent step (same convention as
`toy-model/SiO2-CG/test42.py`'s verified VE-SDE reverse update): the per-step noise scales with
`sqrt(dv)` (dv = σᵢ² − σᵢ₊₁², the *change* in noise level) instead of the raw `σᵢ`, and the
model's prediction is scaled by `dv/σᵢ²` (derived via Tweedie's formula from this network's own
`dx`-prediction training objective) rather than subtracted at full strength every step. It also
uses a geometric σ schedule (not DM2's linear one) and caps `SIGMA_MAX` at training's own 0.75 (not
DM2's generation-time 1.0, which exceeds what the model ever saw). No CLI (matches this repo's
config-constants style) — edit the `=== Change here ===` block at the top of the file (`INIT`,
`CHECKPOINT_PATH`, `CUTOFF` must match `train.py`'s architecture, etc.).

```bash
python generate-v2.py   # reads the === Change here === constants at the top of the file
```

スパコンでは `run_test48.pbs` の `STAGE=generate-v2` から実行できます(`model/test48_model.pt` が
必要):

```bash
qsub -P PROJECT_ID -v STAGE=generate-v2 run_test48.pbs
```

Reuses `ase_graph_gpu`/`set_gpu` from `generate.py` unchanged; defines its own `InitialEmbedding`
(required for unpickling `torch.save(model, ...)` from a different `__main__` script — see the
file's own comment). Adds a same-species (Si-Si, O-O) minimum-distance check to `metrics.json`,
which neither `generate.py` nor `train.py` compute. CPU-verified for correctness (tiny random
weights, no crash); not yet validated against a real trained checkpoint's actual generation
quality.
