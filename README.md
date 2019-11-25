# Source codes for AISTATS 2020 submission "Graph Coarsening with Preserved Spectral Properties"

This is the source codes for AISTATS 2020 submission "Graph Coarsening with Preserved Spectral Properties".


## Installation
1. Download the software package

2. Install software dependencies
- numpy
- networkx
- sklearn


## Usage

### Graph classification with coarse graphs.

`main_classification.py` contains the experimental codes for graph classification for coarse graphs. 
The basic usage is 

```
python main_classification.py
```

Parameter options are

-- dataset: MUTAG, ENZYMES, NCI1, NCI109, PROTEINS, PTC
-- method: mgc, sgc
-- ratio: the ratio between coarse and original graphs n/N


### Block recovery of random graphs from stochastic block model. 

`main_sbm.py` contains the experimental codes for graph classification for coarse graphs. 
The basic usage is 
```
python main_classification.py
```
The parameter options are

--sbm_type: associative, dissociative, mixed
--method: mgc, sgc
--N, node size of original graphs
--n, node size of coarse graphs
--p, edge probability between nodes within the same blocks
--q, edge probability between nodes in different blocks







