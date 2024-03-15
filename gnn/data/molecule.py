from __future__ import annotations
from typing import List, Optional, Union

import torch
from torch import Tensor

from .hypergraph import Hypergraph


class Molecule(Hypergraph):
    """
    Hypergraph with spatial node features `pos`.
    """

    def __init__(
        self,
        cells: List[Tensor],
        features: List[Tensor],
        pos: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__(cells, features, batch, device)
        self.pos = pos if pos is not None else torch.zeros(features[0].shape[0], 3)

    def update(self, pos: Optional[Tensor] = None, **kwargs) -> Molecule:
        out = super().update(**kwargs)
        return out.update_(pos=pos)

    def update_(self, pos: Optional[Tensor] = None, **kwargs) -> Molecule:
        super().update_(**kwargs)
        if pos is not None:
            self.pos = pos
        return self

    @classmethod
    def from_qm9(cls, data: object, degree: int = 1) -> Molecule:
        """
        Parse molecule attributes from a `pyg.data.Data` object.

        Parameters:
            data (`pyg.data.Data | pyg.data.Batch`):
                molecule descriptor as yielded by `pyg.datasets.QM9`
            degree (`int`) = 1:
                with `degree = 2`, also include 2-faces (ijk) obtained
                for any consecutive edges (ij) and (jk), carrying a
                scalar angle feature.
        """
        cells = [data.edge_index]
        i, j = data.edge_index
        nodes = data.z
        edges = (data.pos[i] - data.pos[j]).norm(dim=[-1])
        mol = cls(cells, [nodes, edges], pos=data.pos, batch=data.batch)
        if degree == 1:
            return mol
        cells_2 = mol.get_faces_from_edges()
        mol.cells.append(cells_2)
        i, j, k = cells_2
        v_ji, v_jk = mol.pos[i] - mol.pos[j], mol.pos[k] - mol.pos[j]
        n_ji, n_jk = v_ji.norm(dim=[-1]), v_jk.norm(dim=[-1])
        cos_ijk = (v_ji * v_jk).sum([-1]) / (n_ji * n_jk)
        if degree == 2:
            return mol.update(faces=cos_ijk)
        # TODO: Compute 3-faces (same logic as `get_faces_from_edges`)
        raise NotImplementedError("Only 2 faces are currently supported.")
