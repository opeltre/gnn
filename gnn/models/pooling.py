import torch
import torch.nn as nn

import gnn

class NodePool(nn.Module):
    """
    Aggregate node features batch-wise.
    """

    def __init__(self, reduce:str='sum', dim:int=0):
        super().__init__()
        self.reduce = reduce
        self.dim = dim

    def forward(self, graph:gnn.data.Hypergraph) -> torch.Tensor:
        return scatter(graph.nodes, graph.batch, dim=self.dim, reduce=self.reduce)
        

class GlobalPool(nn.Module):
    """
    Aggregate node, edge, ... features batch-wise.
    """

    pass
