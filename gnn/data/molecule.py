from __future__ import annotations
from typing import List, Optional, Union

import torch
from torch import Tensor

from .hypergraph import Hypergraph

class Molecule(Hypergraph):
    """ 
    Hypergraph with spatial node features in front. 

    Properties:
    ----------
        pos (`torch.Tensor`): 
            returns `self.nodes[:3]`,
        z (`torch.Tensor`): 
            returns `self.nodes[3:]`.
    """

    @property
    def pos(self):
        return self.features[0][:,:3]

    @property
    def z(self):
        return self.features[0][:,3:]

    def update(
        self, 
        pos: Optional[Tensor] = None,
        z: Optional[Tensor] = None,
        **kwargs
    ) -> Molecule:
        if pos is None and z is None:
            return super().update(**kwargs)
        node_features = (
            self.pos if pos is None else pos,
            self.z if z is None else z
        )
        node_features = torch.cat(node_features, -1)
        return super().update(nodes=node_features, **kwargs)

    @classmethod
    def from_qm9(cls, data:object) -> Molecule:
        cells = [data.edge_index]
        i, j = data.edge_index
        nodes = torch.cat((data.pos, data.z.unsqueeze(-1)), -1)
        edges = (data.pos[i] - data.pos[j]).norm(dim=[-1])
        return cls(cells, [nodes, edges], batch=data.batch)
