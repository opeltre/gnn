import torch
import numpy as np

from .matlab import Matlab

from typing import Union, Optional, Iterable, NamedTuple
from pathlib import Path

class QM7 (Matlab):
    """
    QM7 dataset. 
    """

    url = "http://quantum-machine.org/data/qm7.mat"
    
    def __init__(self, 
                 data   : Union[Path, str, dict]  = "qm7.mat",
                 keys   : Optional[Iterable[str]] = None, 
                 folds  : Iterable[int] = None):
        """
        Load $GNN_DATA/qm7.mat by default.
        """
        dset = Matlab(data)
        dset.data['T'] = dset.T[0]
        #--- ignore fold key 
        if 'P' in dset.data:
            self.folds = dset.data.pop('P')
        #--- filter folds ---
        if folds is not None: 
            idx = np.concatenate([self.folds[i] for i in folds])
            dset.data = Matlab(dset.data).subset(idx).data
            self.folds = self.folds[folds]
        super().__init__(dset.data)

    @property
    def positions(self):
        return self.R
    
    @property
    def energy(self):
        return self.T

    @property 
    def coulomb(self):
        return self.X
