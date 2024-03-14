import torch
import torch.nn as nn
import torch_geometric as pyg

import gnn
from gnn.data import Molecule
from gnn import MessagePassing

from tqdm import tqdm

# torch_geometric ships with QM9
dset = pyg.datasets.QM9(gnn.datadir / 'pyg')
loader = pyg.loader.DataLoader(
    dset, 
    batch_size=128, 
    num_workers=4,
)
# parse molecule (CH4)
ch4 = Molecule.from_qm9(dset[0])

# feature dimensions
n0, n1 = 32, 16

# simple Euclid-invariant encoder
encoder = gnn.MoleculeEncoder(n0, n1)
mol = encoder(ch4)

# message-passing layer
mp_layer = gnn.MessagePassing(
    message = nn.Sequential(
        gnn.Cat(-1),
        gnn.MLP((2 * n0 + n1, 128, n0))
    ),
    update = lambda G, y: G.update(nodes = G.nodes + y)
)

# MPNN with shared weights 
mpnn = nn.Sequential(
    mp_layer, 
    mp_layer,
    mp_layer
)

def main():
    for batch in tqdm(loader):
        mol0 = Molecule.from_qm9(batch)
        mol1 = encoder(mol0)
        mol2 = mpnn(mol1)
        break
