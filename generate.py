"""
test48: this is DM2's own demo/demo_generating/denoise_generate_unconditional.py, def-for-def
unchanged (InitialEmbedding / ase_graph_gpu / set_gpu / denoise_snapshot_with_noise_gpu /
denoise_snapshot_gpu are byte-for-byte identical to the upstream script -- same module-level
CUTOFF = 5.0, same 2900 noisy-denoising steps + 100 polish steps, same max_sigma_for_denoising=1.0).
Only main()'s configuration is changed: the model/input/output paths point at test48's own crystal
run instead of DM2's own glass demo, and the initial structure is test48's own genuinely random SiO2
crystal-composition configuration (inital_data/random_sio2_crystal_demo.data: 64 Si + 128 O, uniform
random positions in the 13.573 A cell, no minimum-distance enforcement -- generated the same crude
way DM2's own inital_data/random_sio2_size_300_demo.dat was verified to be built). This is the
model trained by train.py (irreps raised to l<=5; see that file's docstring), so it expects the
same l<=5 checkpoint.
"""

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import ase.io
from ase.neighborlist import primitive_neighbor_list
from pathlib import Path
from functools import partial
from sklearn.preprocessing import LabelEncoder
from tqdm.notebook import trange
from torch_geometric.data import Data, Dataset
# See train.py's docstring/comment: e3nn==0.4.4's internal torch.load of its own Wigner-3j
# constants needs this allowlist on PyTorch >=2.6. Not part of DM2's own def's.
torch.serialization.add_safe_globals([slice])
from graphite.nn.basis import bessel
from graphite.transforms import RattleParticles, DownselectEdges
import time
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

# Global constants
CUTOFF = 5.0

class InitialEmbedding(nn.Module):
    def __init__(self, num_species, cutoff):
        super().__init__()
        self.embed_node_x = nn.Embedding(num_species, 8)
        self.embed_node_z = nn.Embedding(num_species, 8)
        self.embed_edge   = partial(bessel, start=0.0, end=cutoff, num_basis=16)

    def forward(self, data):
        # Embed node
        data.h_node_x = self.embed_node_x(data.x)
        data.h_node_z = self.embed_node_z(data.x)

        # Embed edge
        data.h_edge = self.embed_edge(data.edge_attr.norm(dim=-1))

        return data


# Compute the ase_graph and move to GPU
def ase_graph_gpu(data, cutoff):
    """GPU-optimized graph creation with proper shape handling"""
    # Compute neighbor list (still needs CPU as it's ASE-dependent)
    pos_cpu = data.pos.cpu().numpy()
    cell_cpu = data.cell.cpu().numpy() if torch.is_tensor(data.cell) else data.cell

    i, j, D = primitive_neighbor_list('ijD',
                                    cutoff=cutoff,
                                    pbc=data.pbc,
                                    cell=cell_cpu,
                                    positions=pos_cpu,
                                    numbers=data.numbers)

    # Move to GPU directly without concatenation
    device = data.pos.device
    data.edge_index = torch.from_numpy(np.stack((i, j))).to(device).long()
    data.edge_attr = torch.from_numpy(D).to(device).float()

    return data

def set_gpu(gpu_id):
    """Set PyTorch to use a specific GPU."""
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)


@torch.no_grad()
def denoise_snapshot_with_noise_gpu(atoms, model, scale=1.0, steps=8, max_sigma_for_denoising=0.1):
    device = next(model.parameters()).device
    print(f"Running denoising on device: {device}")

    # Pre-compute sigma values for all steps
    sigmas = torch.linspace(max_sigma_for_denoising, 0.001, steps).to(device)

    # Initialize data on GPU in one go
    x = torch.tensor(LabelEncoder().fit_transform(atoms.numbers), device=device).long()
    pos = torch.tensor(atoms.positions, device=device).float() * scale
    cell = torch.tensor(atoms.cell, device=device).float() * scale

    data = Data(
        x=x,
        pos=pos,
        cell=cell,
        pbc=atoms.pbc,
        numbers=atoms.numbers,
    )

    # Pre-allocate memory for trajectory
    pos_traj = [atoms.positions]
    pos_traj_gpu = []

    # denoising loop
    for i, sigma in enumerate(sigmas, 1):
        data = ase_graph_gpu(data, cutoff=CUTOFF)

        # Apply noise and compute displacement in a single GPU operation
        noisy_data = data.clone()
        noisy_data = RattleParticles(sigma_max=sigma.item())(noisy_data)

        # Compute displacement (all on GPU)
        gpu_disp = model(data) + noisy_data.dx.to(device)
        data.pos.sub_(gpu_disp)  # In-place subtraction

        # Append to trajectory (minimal CPU transfer)
        pos_traj_gpu.append(data.pos.clone())
        # pos_traj.append((data.pos.cpu() / scale).numpy())

        # Print progress
        print(f"Progress: {i}/{steps} steps (sigma = {sigma.item():.6f})", flush=True)

    # Transfer all positions to CPU at once at the end
    print("Transferring trajectory to CPU...")
    pos_traj = [pos.cpu().numpy() for pos in pos_traj_gpu]
    return pos_traj

@torch.no_grad()
def denoise_snapshot_gpu(atoms, model, scale=1.0, steps=8):
    # Get the device from the model
    device = next(model.parameters()).device
    print(f"Running denoising on device: {device}")

    # Convert to PyG format and move to GPU immediately
    x = torch.tensor(LabelEncoder().fit_transform(atoms.numbers), device=device).long()
    data = Data(
        x=torch.tensor(x, device=device).long(),
        pos=torch.tensor(atoms.positions, device=device).float(),
        cell=torch.tensor(atoms.cell, device=device).float(),
        pbc=atoms.pbc,
        numbers=atoms.numbers,
    )

    # Scale
    data.pos *= scale
    data.cell *= scale

    # Denoising trajectory
    pos_traj = [atoms.positions]
    pos_traj_gpu = []

    # Process each step
    for i in range(1, steps + 1):
        # Use the GPU-optimized graph creation
        data = ase_graph_gpu(data, cutoff=CUTOFF)

        # Compute displacement on GPU
        gpu_disp = model(data)

        # Update positions (still on GPU)
        data.pos.sub_(gpu_disp)  # In-place subtraction

        # Keep trajectory on GPU
        pos_traj_gpu.append(data.pos.clone())

        # Print progress
        print(f"Progress: {i}/{steps} steps", flush=True)

    # Transfer all positions to CPU at once at the end
    print("Transferring trajectory to CPU...")
    pos_traj = [(pos.cpu() / scale).numpy() for pos in pos_traj_gpu]

    return pos_traj

def main():
    start_time = time.time()

    # === Change here ===#
    gpu_id = 0
    set_gpu(gpu_id)
    model = torch.load('./model/test48_model.pt', weights_only=False)
    CUTOFF = 5
    model = model.to('cuda')
    TEST_FNAME = './inital_data/random_sio2_crystal_demo.data'
    OUTPUT_FNAME = './gen_data/test48_denoised_random_sio2_crystal.extxyz'
    # ===================#

    noisy_atoms = ase.io.read(TEST_FNAME, format='lammps-data', Z_of_type={1: 14, 2: 8})

    pos_traj = denoise_snapshot_with_noise_gpu(
        noisy_atoms, model,
        scale=1.0, steps=2900, max_sigma_for_denoising=1.0
    )
    a = ase.Atoms(
        symbols=noisy_atoms.get_chemical_symbols(),
        positions=pos_traj[-1],
        cell=noisy_atoms.cell,
        pbc=True
    )
    pos_traj_2 = denoise_snapshot_gpu(a, model, scale=1.0, steps=100)
    pos_traj.extend(pos_traj_2)
    denoising_traj = [
        ase.Atoms(
            symbols=noisy_atoms.get_chemical_symbols(),
            positions=pos,
            cell=noisy_atoms.cell,
            pbc=True
        )
        for pos in pos_traj
    ]
    for atoms in denoising_traj:
        atoms.wrap()
    ase.io.write(OUTPUT_FNAME, denoising_traj)

    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    print(f"\n✅ Total runtime: {elapsed_minutes:.2f} minutes ({elapsed_minutes/60:.2f} hours)")


if __name__ == "__main__":
    main()
