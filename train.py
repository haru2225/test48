#!/usr/bin/env python3
"""
Training script for a neural network denoiser model using PyTorch Geometric.
The model is designed to work with atomic structures and implements the NequIP architecture.

test48: this is DM2's own demo/demo_training/denoiser_train_unconditional.py, def-for-def
unchanged (ase_graph / PeriodicStructureDataset / InitialEmbedding / loss_fn / train / test /
set_gpu are byte-for-byte identical to the upstream script), applied to test47's SiO2 crystal
(beta-cristobalite) instead of DM2's own 3000-atom glass, with the NequIP irreps raised to l<=5
(test47's values). Everything else in main() was originally left at DM2's own defaults (as
requested at the time), EXCEPT:

- LARGE_CUTOFF is now 5 (== CUTOFF), not DM2's 10. DM2's 10 A is tuned for DM2's own ~36 A glass
  box; this SiO2 crystal unit cell is only 13.4-13.57 A across (half-box ~6.7-6.79 A), SMALLER
  than 10 A -- the duplicate-periodic-image issue test33's module docstring describes ("fix #1"),
  which corrupted the local geometry the network was actually trained on. Fixed on request after
  it was diagnosed as the likely dominant remaining cause of poor generation quality. As with
  test47, this leaves no rattle margin above CUTOFF (see that file's own note on the trade-off).
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import ase.io
import warnings
from ase.neighborlist import primitive_neighbor_list
from torch_geometric.data import Data, Dataset
from sklearn.preprocessing import LabelEncoder
from torch_geometric.loader import DataLoader
from torch import nn
from functools import partial
# e3nn==0.4.4 (DM2's pinned version) loads its own Wigner-3j constants via a bare torch.load with
# no weights_only argument -- PyTorch >=2.6 changed that default to True, which then rejects the
# slice objects stored in e3nn's constants.pt. Not part of DM2's own def's; a runtime-version shim
# needed for this script to import graphite at all here (see test42.py for the same fix).
torch.serialization.add_safe_globals([slice])
from graphite.nn.basis import bessel
from graphite.nn.models.e3nn_nequip import NequIP
from graphite.transforms import RattleParticles, DownselectEdges
from tqdm.notebook import trange
import time

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
warnings.filterwarnings("ignore", category=UserWarning, module='torch.jit._check')

def ase_graph(data, cutoff):
    """Convert atomic structure to graph representation."""
    i, j, D = primitive_neighbor_list('ijD', cutoff=cutoff, pbc=data.pbc,
                                    cell=data.cell, positions=data.pos.numpy(),
                                    numbers=data.numbers)
    data.edge_index = torch.tensor(np.stack((i, j)), dtype=torch.long)
    data.edge_attr = torch.tensor(D, dtype=torch.float)
    return data

class PeriodicStructureDataset(Dataset):
    """Dataset class for periodic atomic structures."""

    def __init__(self, atoms_list, large_cutoff, duplicate=128):
        super().__init__(None, transform=None, pre_transform=None)

        self.dataset = []
        for atoms in atoms_list:
            x = LabelEncoder().fit_transform(atoms.numbers)
            data = Data(
                x       = torch.tensor(x).long(),
                pos     = torch.tensor(atoms.positions).float(),
                cell    = atoms.cell,
                pbc     = atoms.pbc,
                numbers = atoms.numbers,
            )
            data = ase_graph(data, large_cutoff)
            self.dataset.append(data)
        self.dataset = [d.clone() for d in self.dataset for _ in range(duplicate)]

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]

class InitialEmbedding(nn.Module):
    """Initial embedding layer for the neural network."""

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

def loss_fn(model, data):
    """Calculate MSE loss between predicted and actual displacement."""
    pred_dx = model(data)
    return torch.nn.functional.mse_loss(pred_dx, data.dx)

def train(loader, model, optimizer, device, rattle_particles, downselect_edges, PIN_MEMORY):
    """Training loop for one epoch."""
    start_time = time.time()
    model.train()
    total_loss = 0.0
    for data in loader:
        optimizer.zero_grad(set_to_none=True)
        data = data.to(device, non_blocking=PIN_MEMORY)
        data = rattle_particles(data)
        data = downselect_edges(data)
        loss = loss_fn(model, data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    epoch_time = time.time() - start_time
    return total_loss / len(loader), epoch_time

@torch.no_grad()
def test(loader, model, device, rattle_particles, downselect_edges, PIN_MEMORY):
    """Validation loop for one epoch."""
    model.eval()
    total_loss = 0.0
    for data in loader:
        data = data.to(device, non_blocking=PIN_MEMORY)
        data = rattle_particles(data)
        data = downselect_edges(data)
        loss = loss_fn(model, data)
        total_loss += loss.item()
    return total_loss / len(loader)

def set_gpu(gpu_id):
    """Set PyTorch to use a specific GPU."""
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

# Main function
def main():
    # === Main configuration parameter ===
    NUM_SPECIES = 2         # important to check with your system or will casue error.
    gpu_id = 0              # adjustable based on your device.
    PIN_MEMORY  = True      # related to optimization for training, revert to False if you see any issues.
    NUM_WORKERS = 0         # related to optimization for training, revert to 1 if you see any issues.
    BATCH_SIZE  = 16        # adjust so that each minibatch fits in the (GPU) memory.
    LARGE_CUTOFF = 5        # test48: was DM2's default 10; lowered to == CUTOFF to avoid the
                            # duplicate-periodic-image bug in this small cell (see module docstring)
    CUTOFF      = 5         # recommend. May not the best value for every system and can affect model performace.
    LEARN_RATE  = 2e-4      # adjustable.
    NUM_UPDATES = 30_000    # need to see convergence in loss plot (more or less).
    train_ratio = 0.9       # adjustable.
    sigma_max_value = 0.75  # adjust added noise in noisying process (recommend 0.75).
    path_save_loss_fig = './test48_loss_figure.png'        # adjustable
    path_save_model = './model/test48_model.pt'            # adjustable
    # ====================

    # === Load data (test48: 184 beta-cristobalite thermal MD snapshots from 4 NVT trajectories
    # (fixed cell volume), in place of DM2's own 3 glass .dat files -- see
    # simu_data/reference_frames_metadata.json) ===
    archive = np.load('./simu_data/reference_frames.npz')
    positions, cell_lengths, numbers = archive['positions'], archive['cell_lengths'], archive['numbers']
    ideal_atoms_list = [
        ase.Atoms(numbers=numbers, positions=pos, cell=np.diag(lengths), pbc=True)
        for pos, lengths in zip(positions, cell_lengths)
    ]
    # ====================

    # === Initialize model ===
    # Recommend architecture, but can modify
    # test48: irreps raised to l<=5 (test47's values; DM2's own recommendation was l<=1
    # hidden / l<=2 edge, i.e. '64x0e + 32x1e' / '4x0e + 4x1e + 2x2e').
    model = NequIP(
        init_embed     = InitialEmbedding(num_species=NUM_SPECIES, cutoff=CUTOFF),
        irreps_node_x  = '8x0e',
        irreps_node_z  = '8x0e',
        irreps_hidden  = '64x0e + 32x1e + 16x2e + 8x3e + 4x4e + 2x5e',
        irreps_edge    = '4x0e + 4x1e + 2x2e + 2x3e + 1x4e + 1x5e',
        irreps_out     = '1x1e',
        num_convs      = 3,
        radial_neurons = [16, 64],
        num_neighbors  = 12,
    )
    # ====================

    # Setup transformations
    rattle_particles = RattleParticles(sigma_max=sigma_max_value)
    downselect_edges = DownselectEdges(cutoff=CUTOFF)

    # Prepare dataset
    dataset = PeriodicStructureDataset(ideal_atoms_list, large_cutoff=LARGE_CUTOFF)
    num_train = int(train_ratio * len(dataset))
    num_valid = len(dataset) - num_train
    ds_train, ds_valid = torch.utils.data.random_split(dataset, [num_train, num_valid])

    # Create dataloaders
    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    valid_loader = DataLoader(ds_valid, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # Training setup
    num_samples = len(dataset)
    num_epochs = int(NUM_UPDATES/(num_samples/BATCH_SIZE))
    print(f'{num_epochs} epochs needed to update the model {NUM_UPDATES} times.')

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARN_RATE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_gpu(gpu_id)
    model.to(device)

    # Training loop
    L_train, L_valid = [], []
    total_start_time = time.time()
    total_training_time = 0

    for epoch in range(num_epochs):
        train_loss, epoch_time = train(train_loader, model, optimizer, device,
                                     rattle_particles, downselect_edges, PIN_MEMORY)
        valid_loss = test(valid_loader, model, device,
                         rattle_particles, downselect_edges, PIN_MEMORY)
        total_training_time += epoch_time

        L_train.append(train_loss)
        L_valid.append(valid_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}] '
          f'Train Loss: {train_loss:.4f}, '
          f'Valid Loss: {valid_loss:.4f}, '
          f'Time: {epoch_time:.2f}s')

    total_time = time.time() - total_start_time
    print(f'\nTraining completed:')
    print(f'Total training time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)')
    print(f'Average time per epoch: {total_training_time/num_epochs:.2f} seconds')

    # Save model
    torch.save(model, path_save_model)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
    ax1.plot(L_train, label='train')
    ax1.plot(L_valid, label='valid')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epochs')
    ax1.legend()

    ax2.semilogy(L_train, label='train')
    ax2.semilogy(L_valid, label='valid')
    ax2.set_xlabel('Epochs')
    ax2.legend()
    plt.show()
    plt.savefig(path_save_loss_fig, bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    main()
