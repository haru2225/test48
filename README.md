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

## スパコンでの実行（test47 と同じ: Singularity + DM2 をバインド）

```bash
git clone https://github.com/haru2225/test48.git && cd test48

module load singularity
singularity build test48.sif Singularity.def
# Apptainer: apptainer build test48.sif Singularity.def
```

`singularity build` が `ssl-verification-failed` で失敗する場合、多くはログイン/計算ノードから
Docker Hub への HTTPS 通信がプロキシ経由でないと通らない、または外部に出られないことが原因です
（このファイル自体の問題ではありません）。対処:

```bash
# 1. プロキシが要る場合
export https_proxy=http://proxy.example:8080 http_proxy=http://proxy.example:8080
singularity build test48.sif Singularity.def

# 2. apptainer で試す（singularityとHTTPクライアント実装が異なる）
apptainer build test48.sif Singularity.def

# 3. それでも失敗する場合: インターネットに出られる別のLinux環境でビルドしてから転送する
#    （このプロジェクトの他のtestでも使っている方法）
scp test48.sif your_cluster:/path/to/test48/
```

```bash
qsub -P PROJECT_ID -v STAGE=train run_test48.pbs
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
