#!/usr/bin/env python3
"""Corrected generation for a test48-trained checkpoint (model/test48_model.pt).

WHY THIS FILE EXISTS: DM2's own annealed-denoising generation (generate.py's
denoise_snapshot_with_noise_gpu, copied def-for-def from DM2) adds noise of magnitude `sigma`
itself at every step, fresh and independent each time. Summed over the full 2900-step schedule
(sigma: 1.0 -> 0.001), the injected noise ALONE has variance sum(sigma_i**2) =~ 968, i.e. a
standard deviation of ~31 Angstrom -- more than TWICE the 13.4-13.57 A unit cell -- regardless of
whether the model's prediction is any good. Verified empirically on a real test47 run (same
network family, same generation algorithm): mean nearest-neighbor distance 1.24 A, minimum 0.24 A
(atoms on top of each other), completely flat bond/angle histograms. generate.py (DM2's own code,
left unmodified on purpose) has the identical issue.

THE FIX: use the properly SDE-consistent per-step noise scale sqrt(dv), dv = sigma_i**2 -
sigma_{i+1}**2 (the *change* in noise level, not the noise level itself), matching the VE-SDE
reverse step already used and verified in this project's test42/43/44. The network here is
trained to predict `dx` (RattleParticles' raw added displacement, dx = sigma*eps), an eps/x0-
prediction denoiser, not a sigma-normalized score. Via Tweedie's formula eps ~= -sigma*score, so
sigma*score ~= -pred_dx/sigma, giving the corrected step used below:

    dv = sigma_i**2 - sigma_{i+1}**2
    pos <- pos - (dv / sigma_i**2) * pred_dx + sqrt(dv) * randn        (annealing phase)
    pos <- pos - pred_dx                                               (final zero-noise polish)

This is a NEW script, not a modification of generate.py (which is left as the DM2-def-for-def
port it was), so both behaviours stay directly comparable. Also adds a same-species (Si-Si, O-O)
minimum-distance check, which generate.py's output has no equivalent of.

Reuses `ase_graph_gpu` / `set_gpu` from generate.py unchanged; everything else here is new.
"""
import json
from pathlib import Path

import ase.io
import numpy as np
import torch
from torch import nn
from ase import Atoms
from ase.neighborlist import primitive_neighbor_list
from functools import partial
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from graphite.nn.basis import bessel

torch.serialization.add_safe_globals([slice])  # see train.py's docstring: e3nn/PyTorch>=2.6 shim
from generate import ase_graph_gpu, set_gpu  # noqa: E402  (needs the safe_globals call above first)


class InitialEmbedding(nn.Module):
    """Identical to train.py's/generate.py's own class -- torch.save(model, ...) pickles the whole
    object, and unpickling it needs this exact class to also exist as `__main__.InitialEmbedding`
    in WHICHEVER script does the loading (this is why DM2's own train/generate scripts each define
    their own copy instead of importing one shared definition); this script is no exception."""

    def __init__(self, num_species, cutoff):
        super().__init__()
        self.embed_node_x = nn.Embedding(num_species, 8)
        self.embed_node_z = nn.Embedding(num_species, 8)
        self.embed_edge = partial(bessel, start=0.0, end=cutoff, num_basis=16)

    def forward(self, data):
        data.h_node_x = self.embed_node_x(data.x)
        data.h_node_z = self.embed_node_z(data.x)
        data.h_edge = self.embed_edge(data.edge_attr.norm(dim=-1))
        return data

SCRIPT_DIR = Path(__file__).resolve().parent

# === Change here ===#
CHECKPOINT_PATH = SCRIPT_DIR / "model" / "test48_model.pt"
CRYSTAL_DATA = SCRIPT_DIR / "simu_data" / "silica_beta_cristobalite_init.data"  # --init crystal
OUTPUT_DIR = SCRIPT_DIR / "gen_data" / "generate_v2"
INIT = "crystal"          # "crystal" or "random"
STEPS = 2900              # annealing steps, SIGMA_MAX -> SIGMA_MIN
POLISH_STEPS = 100        # extra zero-noise steps at the end
SIGMA_MAX = 0.75          # train.py's own sigma_max_value; do NOT exceed what the model actually
                          # saw in training, unlike DM2's own gen-time max_sigma_for_denoising=1.0
SIGMA_MIN = 0.03
CUTOFF = 5.0              # must match train.py's CUTOFF the checkpoint was trained with
SEED = 1337
GPU_ID = 0
# ===================#


def sigma_schedule(sigma_max, sigma_min, steps):
    """Geometric (log-spaced), matching test42-44's VE-SDE schedule -- NOT DM2's linear one."""
    return np.geomspace(sigma_max, sigma_min, steps + 1)


def initial_positions(atoms, init):
    if init == "crystal":
        return np.asarray(atoms.positions, dtype=np.float32)
    if init == "random":
        lengths = np.diag(np.asarray(atoms.cell))
        return (np.random.rand(len(atoms), 3) * lengths).astype(np.float32)
    raise ValueError(f"Unknown init: {init}")


def bond_and_angle_stats(positions, numbers, cell):
    si_indices = np.where(numbers == 14)[0]
    o_indices = np.where(numbers == 8)[0]
    displacement = positions[si_indices, None, :] - positions[None, o_indices, :]
    fractional = displacement @ np.linalg.inv(cell)
    fractional -= np.round(fractional)
    displacement = fractional @ cell
    distances = np.linalg.norm(displacement, axis=-1)
    bonds, angles = [], []
    for si_local in range(len(si_indices)):
        nearest = np.argsort(distances[si_local])[:4]
        bonds.extend(distances[si_local, nearest].tolist())
        vectors = displacement[si_local, nearest]
        vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
        for a in range(4):
            for b in range(a + 1, 4):
                cosine = np.clip(np.dot(vectors[a], vectors[b]), -1.0, 1.0)
                angles.append(np.degrees(np.arccos(cosine)))
    return np.asarray(bonds), np.asarray(angles)


def same_species_min_distance(positions, numbers, cell, number):
    indices = np.where(numbers == number)[0]
    if len(indices) < 2:
        return None
    i, j, d = primitive_neighbor_list("ijd", pbc=[True, True, True], cell=cell,
                                       positions=positions[indices], cutoff=min(cell.diagonal()) / 2 - 1e-6)
    return float(d.min()) if len(d) else None


def wrap_positions(positions, cell):
    fractional = positions @ np.linalg.inv(cell)
    fractional -= np.floor(fractional)
    return fractional @ cell


def save_histogram(reference, generated, target, xlabel, title, marker):
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots()
    axis.hist(reference, bins=60, density=True, histtype="step", label="reference", linewidth=2)
    axis.hist(generated, bins=60, density=True, histtype="step", label="generated", linewidth=2)
    axis.axvline(marker, color="gray", linestyle=":")
    axis.set(xlabel=xlabel, title=title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(target, dpi=160)
    plt.close(figure)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    set_gpu(GPU_ID)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()

    atoms = ase.io.read(CRYSTAL_DATA, format="lammps-data", Z_of_type={1: 14, 2: 8})
    numbers = np.asarray(atoms.numbers)
    cell_np = np.asarray(atoms.cell)
    box = np.diag(cell_np)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start_positions = initial_positions(atoms, INIT)
    pos = torch.tensor(start_positions, dtype=torch.float32, device=device)
    species = torch.tensor(LabelEncoder().fit_transform(numbers), device=device).long()
    cell = torch.tensor(cell_np, dtype=torch.float32, device=device)

    sigmas = sigma_schedule(SIGMA_MAX, SIGMA_MIN, STEPS)
    print(f"init={INIT}  steps={STEPS}+{POLISH_STEPS}  sigma {SIGMA_MAX} -> {SIGMA_MIN} (geometric)", flush=True)

    with torch.no_grad():
        for step in range(STEPS):
            sigma_i, sigma_next = float(sigmas[step]), float(sigmas[step + 1])
            dv = sigma_i**2 - sigma_next**2
            data = Data(x=species, pos=pos, cell=cell, pbc=atoms.pbc, numbers=numbers)
            data = ase_graph_gpu(data, cutoff=CUTOFF)
            pred_dx = model(data)
            pos = pos - (dv / sigma_i**2) * pred_dx + np.sqrt(dv) * torch.randn_like(pos)
            pos = torch.remainder(pos, pos.new_tensor(box))
            if (step + 1) % 100 == 0 or step == STEPS - 1:
                print(f"anneal step {step + 1}/{STEPS}  sigma={sigma_i:.4g}", flush=True)

        for step in range(POLISH_STEPS):
            data = Data(x=species, pos=pos, cell=cell, pbc=atoms.pbc, numbers=numbers)
            data = ase_graph_gpu(data, cutoff=CUTOFF)
            pos = pos - model(data)
            pos = torch.remainder(pos, pos.new_tensor(box))
        print(f"polish steps: {POLISH_STEPS}", flush=True)

    final_positions = pos.detach().cpu().numpy()
    final_wrapped = wrap_positions(final_positions, cell_np)
    final_atoms = Atoms(numbers=numbers, positions=final_wrapped, cell=cell_np, pbc=True)
    ase.io.write(OUTPUT_DIR / "final_structure.extxyz", final_atoms)

    reference_positions = wrap_positions(np.asarray(atoms.positions), cell_np)
    generated_bonds, generated_angles = bond_and_angle_stats(final_wrapped, numbers, cell_np)
    reference_bonds, reference_angles = bond_and_angle_stats(reference_positions, numbers, cell_np)
    save_histogram(reference_bonds, generated_bonds, OUTPUT_DIR / "bond_comparison.png",
                   "Si-O distance (Angstrom)", "test48 generate-v2: Si-O bond length", 1.61)
    save_histogram(reference_angles, generated_angles, OUTPUT_DIR / "angle_comparison.png",
                   "O-Si-O angle (degree)", "test48 generate-v2: O-Si-O angle", 109.47)

    si_si_min = same_species_min_distance(final_wrapped, numbers, cell_np, 14)
    o_o_min = same_species_min_distance(final_wrapped, numbers, cell_np, 8)
    metrics = {
        "init": INIT,
        "steps": STEPS,
        "polish_steps": POLISH_STEPS,
        "sigma_max": SIGMA_MAX,
        "sigma_min": SIGMA_MIN,
        "reference_bond_mean_angstrom": float(reference_bonds.mean()),
        "reference_bond_std_angstrom": float(reference_bonds.std()),
        "generated_bond_mean_angstrom": float(generated_bonds.mean()),
        "generated_bond_std_angstrom": float(generated_bonds.std()),
        "reference_angle_mean_degree": float(reference_angles.mean()),
        "reference_angle_std_degree": float(reference_angles.std()),
        "generated_angle_mean_degree": float(generated_angles.mean()),
        "generated_angle_std_degree": float(generated_angles.std()),
        "generated_si_si_min_distance_angstrom": si_si_min,
        "generated_o_o_min_distance_angstrom": o_o_min,
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in (
        "generated_bond_mean_angstrom", "generated_angle_mean_degree",
        "generated_si_si_min_distance_angstrom", "generated_o_o_min_distance_angstrom")}, indent=2))


if __name__ == "__main__":
    main()
