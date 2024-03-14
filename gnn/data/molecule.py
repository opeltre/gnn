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
    def from_qm9(cls, data: object) -> Molecule:
        cells = [data.edge_index]
        i, j = data.edge_index
        nodes = data.z
        edges = (data.pos[i] - data.pos[j]).norm(dim=[-1])
        return cls(cells, [nodes, edges], pos=data.pos, batch=data.batch)
