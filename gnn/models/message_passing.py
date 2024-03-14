import torch 
import torch.nn as nn
from torch_scatter import scatter

from typing import Union, Callable

import gnn
from gnn.data import Graph, Hypergraph

class MessagePassing(nn.Module):
    """
    Simple message passing layer with node updates.  
    """ 
    def __init__(
        self, 
        message: nn.Module, 
        update: Union[nn.Module, Callable, bool] = True,
        reduce: str = 'sum'
    ): 
         """
        Initialize module. 

        Parameters:
        ----------
            message (`nn.Module`): `Tensor -> Tensor`
                Compute messages from concatenated node and edge features
                `(xi, xj, xij)` by default. 
            update (`callable | bool`): `(Hypergraph, Tensor) -> Hypergraph`
                Update of features based on aggregated messages. 
                If `True`, apply residual update: `xj = xj + sum(Mij)`. 
                If `False`, replace node features: `xj = sum(Mij)`.
            reduce (`str`): aggregation of messages, default = `"sum"`.
        """

        super().__init__()
        self.message = message
        if callable(update):
            self.update = update
        elif update: 
            self.update = lambda G, y: G.update(nodes = y + G.nodes)
        else: 
            self.update = lambda G, y: G.update(nodes = y)
        self.reduce = reduce

    def message_features(self, graph:Graph) -> torch.Tensor:
        """
        Feed input to `self.message`, override as needed.
        """
        i, j = graph.edge_indices
        e_ij = graph.edges
        x_i, x_j = graph.nodes[i], graph.nodes[j]
        return torch.cat((x_i, x_j, e_ij), -1)

    def forward(self, graph:Graph) -> Graph:
        x_ij = self.message_features(graph)
        M_ij = self.message(x_ij)
        y_j = scatter(M_ij, j, -2, reduce=self.reduce)
        return self.update(graph, y_j)


class MessagePassing2(nn.Module):
    """ 
    Message-passing layer on a 2-graph.
    """

    pass
