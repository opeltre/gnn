# Graph Neural Networks

This didactic repository contains example code for desiging message-passing layers on graph and hypergraph structures.

It was prepared for a [datacraft](https://datacraft.paris) state-of-the-art session on 
[_Message-Passing Neural Networks for Generation of Chemical Structures_](https://github.com/opeltre/gnn/blob/main/Message%20Passing%20Neural%20Networks.pdf) (slides).

## Installation 

With python 3.9.18:

```sh
$ git clone git@github.com:opeltre/gnn && cd gnn
$ pip install -r requirements.txt && pip install -e .
```
With python 3.12, installation of `torch-scatter` may fail depending on the `torch` version. 
A working `requirements.txt` or `pyproject.toml` should be uploaded!

## Useful links and references

### Equivariant MPNNs
- [Equivariant Message Passing Neural Network for Crystal Material Discovery](https://ojs.aaai.org/index.php/AAAI/article/view/26673) (Klipfel et al 2023)
- [MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields](https://arxiv.org/abs/2206.07697) (Batatia et al 2022)
- [HamGNN:Transferable equivariant graph neural networks for the Hamiltonians of molecules and solids](https://www.nature.com/articles/s41524-023-01130-4)(Zhong et al 2023)
- [Tensor field networks: Rotation- and translation-equivariant neural networks for 3D point clouds](https://arxiv.org/pdf/1802.08219.pdf) (Thomas et al 2018)

### GNNs
- [A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/)
- [A Hitchhiker’s Guide to Geometric GNNs for 3D Atomic Systems](https://arxiv.org/abs/2312.07511)

### Libraries
- [torch_scatter](https://pytorch-scatter.readthedocs.io/en/latest/index.html)
- [torch_cluster](https://github.com/rusty1s/pytorch_cluster/tree/master)
- [e3nn](https://e3nn.org/)
- [e3x](https://e3x.readthedocs.io/stable/)

## Datasets

### QM9 dataset 

The QM9 dataset contains ~130k small organic molecules, its upstream url is [quantum-machine.org](https://quantum-machine.org/datasets). 

An interface to QM9 ships with `torch_geometric`, see [QM9](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html#torch_geometric.datasets.QM9) 
and [examples/graph_mpnn.py](examples/graph_mpnn.py)

### QM7 dataset 

Find the QM7 dataset and its description from [quantum-machine.org](https://quantum-machine.org/datasets):

```
export GNN_DATA=".data"
curl http://quantum-machine.org/data/qm7.mat > $GNN_DATA/qm7.mat
```
