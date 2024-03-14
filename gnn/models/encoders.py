import torch
import torch.nn as nn
import torch_geometric as pyg
import gnn
from .common import OneHot, RBF, MLP


class MoleculeEncoder(nn.Module):
    """
    Encode atomic numbers and interatomic distances.

    Note:
    ----
    Expects `gnn.data.Molecule` instances as returned by `Molecule.from_qm9`:
    - `mol.pos` spatial node features,
    - `mol.nodes` contains a single node feature (atomic number),
    - `mol.edges` contains scalar edge distances.
    """

    def __init__(self, node_features: int = 32, edge_features: int = 32):
        super().__init__()
        # encode atomic numbers
        self.node_encoder = nn.Sequential(OneHot(12), MLP((12, 128, node_features)))
        # encode inter-atomic distances
        self.edge_encoder = nn.Sequential(
            RBF(16, (0, 3)), MLP((16, 128, edge_features))
        )

    def forward(self, mol: gnn.data.Molecule):
        node_features = self.node_encoder(mol.nodes)
        edge_features = self.edge_encoder(mol.edges)
        return mol.update(nodes=node_features, edges=edge_features)


class MoleculeEncoder2(MoleculeEncoder):
    """
    Encode atomic numbers, interatomic distances and edge angles.

    TODO:
    ----
    - Create a new `Molecule` instance containing 2-faces `(ijk)` whenever
    `(ij)` and `(jk)` are linked in the molecule's 1-graph.
    - Assign 2-features F[ijk] that are functions of the angles and areas
    of the triangle `(ijk)`.
    """

    pass
