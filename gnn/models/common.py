import torch
import torch.nn as nn

from typing import Tuple, Iterable, Union


class OneHot(nn.Module):
    """
    One-hot encoding of atomic numbers.
    """

    def __init__(self, n: int = 12):
        super().__init__()
        # one-hot encoding of atomic numbers
        self._eye = nn.Parameter(
            torch.eye(n),
            requires_grad=False,
        )

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        return self._eye[Z]


class RBF(nn.Module):
    """
    Radial Basis Functions on scalar inputs.

    We use sines by default, update behaviour as wished.
    """

    def __init__(
        self,
        n_basis: int = 32,
        bounds: Tuple[float, float] = (0.0, 1.0),
    ):
        super().__init__()
        self.n_basis = n_basis
        self.min, self.max = bounds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.unsqueeze(-1) - self.min
        y = y * (1 + torch.arange(self.n_basis, device=x.device))
        y = y / (self.max - self.min)
        return torch.sin(y)


class MLP(nn.Sequential):
    """
    Multi-Layer Perceptron.
    """

    def __init__(
        self,
        ns: Iterable[int],
        act: Union[callable, Iterable[callable], None] = None,
    ):
        if act is None:
            act = nn.Tanh()
        if not isinstance(act, (list, tuple)):
            act = [act] * (len(ns) - 1)
        layers = [
            nn.Sequential(nn.Linear(n1, n2), f)
            for n1, n2, f in zip(ns[:-1], ns[1:], act)
        ]
        super().__init__(*layers)


class Cat(nn.Module):

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, xs: Tuple[torch.Tensor]) -> torch.Tensor:
        return torch.cat(xs, self.dim)
