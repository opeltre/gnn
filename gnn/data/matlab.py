import torch
import numpy as np
import scipy
import requests

import os
from pathlib import Path

from typing import Union, Optional, Iterable, NamedTuple

#--- set environment --- 
os.environ['GNN_DATA'] = str(Path(__file__).parent.parent.parent / "data")
datadir = Path(os.environ["GNN_DATA"] if "GNN_DATA" in os.environ else "data")

class Matlab (torch.utils.data.Dataset):
    """ 
    Interface to matlab tensor-dictionary format. 
    """
    
    url = None 

    def __init__(self, 
                 data : Union[Path, str, dict],
                 keys : Optional[Iterable[str]] = None):
        """
        Load Dict[np.ndarray] data from .mat file or python dictionary.
        """
        if isinstance(data,  str):
            path = datadir / data if data[0] != "/" else data
            self.data = scipy.io.loadmat(path)
        elif isinstance(data, Path):
            path = data
            self.data = scipy.io.loadmat(path)
        elif isinstance(data, dict):
            self.data = data
        else:
            raise RuntimeError(f'Unsupported data type {type(data)}\n' 
                               + str(Matlab.__init__.__annotations__))
        
        if keys is None:
            self.keys = self.read_keys()
        else:
            self.keys = keys

        for k in self.keys:
            setattr(self, k, self.data[k])
        
        self.yield_type = NamedTuple('TensorTuple', **{k : np.ndarray for k in self.keys})
    
    @classmethod
    def download(cls):
        if cls.url is None:
            return None
        print(f"> Getting dataset from {cls.url}")
        response = requests.get(cls.url)
        path = datadir / cls.__name__.lower() + ".mat"
        with os.open(path, "wb") as fbin:
            fbin.write(response.content)
        print(f"> Saved dataset at {path}")
        return path
    
    def read_keys (self):
        return [k for k in self.data.keys() if not k[:2] == "__"]

    def get (self, *keys : str):
        if len(keys) == 1:
            return self.data[keys[0]]
        else:
            return tuple((self.get(k) for k in keys))

    def __getitem__(self, idx:Union[int, torch.LongTensor]) -> tuple:
        """
        Return NamedTuple of (batched) tensors.
        """
        batches = (self.data[k][idx] for k in self.keys)
        return self.yield_type(*batches)

    def subset(
        self, 
        idx:Union[torch.LongTensor, np.ndarray]
    ) -> torch.utils.data.Dataset:
        """
        Return reindexed dataset instance. 
        """
        ntuple = self[idx]
        data = {k: xk for k, xk in zip(self.keys, ntuple)}
        return self.__class__(data, self.keys)
        
    def __len__(self):
        return self.data[self.keys[0]].shape[0]

    def __repr__(self):
        out = self.__class__.__name__ + ":"
        n_keys = len(self.keys)
        for i, k in enumerate(self.keys):
            indent = (f"\n |- {k}\t:" if i < n_keys - 1 else
                      f"\n `- {k}\t:")
            out += indent + f"{self.data[k].shape}"
        return out
