from __future__ import annotations
from typing import Union, Optional, List

import torch
from torch import Tensor, LongTensor


class Hypergraph:
    """
    Dataclass for featured batched (hyper-)graphs.

    Attributes:
    ----------
        cells (`list`):
            list of edge, face, ... indices (shapes `(2, n1)`, `(3, n2)`, ...)
        features (`list`):
            list of node, edge, face... features (shapes `(n0, d0)`, `(n1, d1)`, ...)
        batch (`torch.Tensor`):
            integer vector mapping nodes to connected components (length `n0`)
        device (`str` or `torch.device`)
    """

    def __init__(
        self,
        cells: List[Tensor],
        features: List[Tensor],
        batch: Optional[Tensor] = None,
        device: Optional[str, torch.device] = None,
    ):
        self.cells = cells
        self.features = features
        if batch is None:
            batch = torch.zeros(features[0].shape[0])
        self.batch = batch
        self.device = device or features[0].device

    @classmethod
    def cat(cls, graphs: Hypergraph) -> Hypergraph:
        """
        Batch graphs by concatenating attributes.
        """
        device = graphs[0].device
        dim = graphs[0].dim
        dim_f = graphs[0].dim_features
        # --- concatenate features
        features = []
        for i in range(dim_f + 1):
            fi = torch.cat([g.features[i] for g in graphs])
            features.append(fi)
        # --- concatenate individual sizes: assumes graph.batch = [0, ..., 0]
        batch = torch.cat([g.batch + i for i, g in enumerate(graphs)])
        # --- concatenate adjacency matrices
        cells = []
        for i in range(dim):
            adj_i = torch.cat([g.cells[i] for g in graphs], -1)
            cells.append(adj_i)
        return self.__class__(cells, features, batch, device)

    def update(
        self,
        nodes: Optional[Tensor] = None,
        edges: Optional[Tensor] = None,
        faces: Optional[Tensor] = None,
        *args,
        **kwargs,
    ) -> Hypergraph:
        """
        Hypergraph instance with updated features.
        """
        features = []
        for xi, yi in zip(self.features, [nodes, edges, faces, *args]):
            features.append(yi if yi is not None else xi)
        return self.__class__(
            self.cells,
            features=features,
            batch=self.batch,
            device=self.device,
            **kwargs,
        )

    def update_(
        self,
        nodes: Optional[Tensor] = None,
        edges: Optional[Tensor] = None,
        faces: Optional[Tensor] = None,
        *args,
    ) -> Hypergraph:
        """
        In-place update.
        """
        for i, yi in enumerate([nodes, edges, faces, *args]):
            if i < len(self.features) and yi is not None:
                self.features[i] = yi
            elif yi is not None:
                self.features.append(yi)
        return self

    @property
    def sizes(self):
        if "_sizes" in dir(self):
            return self._sizes
        unique, sizes = torch.unique(self.batch, return_counts=True)
        self._sizes = sizes
        return sizes

    @property
    def nodes(self):
        return self.features[0]

    @property
    def edges(self):
        return self.features[1] if len(self.features) > 1 else None

    @property
    def faces(self):
        return self.features[2] if len(self.features) > 2 else None

    @property
    def edge_indices(self):
        return self.cells[0]

    @property
    def face_indices(self):
        return self.cells[1] if len(self.cells) > 1 else None

    @property
    def dim(self):
        """
        Dimension of the hypergraph (1 for a graph, 2 for a 2-graph, ...).
        """
        return len(self.cells) - 1

    @property
    def dim_features(self):
        """
        Dimension of carried features (<= dim).
        """
        return len(self.features)

    def to(self, device: Union[str, torch.device]) -> Hypergraph:
        """Move graph instance to device."""
        self.device = device
        for i, adj in enumerate(self.cells):
            self.cells[i] = adj.to(device)
        for i, tensor in enumerate(self.features):
            if tensor is not None:
                self.features[i] = tensor.to(device)
        self.batch = self.sizes.to(batch)
        return self

    def __repr__(self) -> str:
        out = f"Graph (n={len(self.sizes)})\n"
        out += f"\tsizes:\t{self.sizes.tolist()}\n"
        out += f"\tnodes:\t{list(self.nodes.shape)}\t{self.nodes.dtype}\n"
        out += f"\tedges:\t{list(self.edges.shape)}\t{self.edges.dtype}\n"
        return out


class Graph(Hypergraph):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.cells) == 1
        assert len(self.features) <= 2

    @classmethod
    def from_pyg(cls, data: object) -> Graph:
        """
        Convert from torch_geometric `Data` / `Batch` instance.
        """
        cells = [data.edge_index]
        features = [data.x, data.edge_attr] if "edge_attr" in dir(data) else [data.x]
        return cls([data.edge_index], features, data.batch, device=data.x.device)
