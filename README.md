# Source codes for AISTATS 2020 Paper "Graph Coarsening with Preserved Spectral Properties"


## Installation
1. Download the software package

2. Install software dependencies
- numpy
- networkx
- sklearn

3. Install Netlsd from https://github.com/xgfs/NetLSD


## Usage

### Graph classification with coarse graphs.

`main_classification.py` contains the experimental codes for graph classification for coarse graphs. 
The basic usage is 

```
python main_classification.py
```

Parameter options are

-dataset: MUTAG, ENZYMES, NCI1, NCI109, PROTEINS, PTC

-method: mgc, sgc

-ratio, the ratio between coarse and original graphs n/N

The default setting is 
```
python main_classification.py --dataset MUTAG --method mgc --ratio 0.2
```

### Block recovery of random graphs from stochastic block model. 

`main_sbm.py` contains the experimental codes for graph classification for coarse graphs. 
The basic usage is 
```
python main_classification.py
```
The parameter options are

-sbm_type: associative, dissociative, mixed

-method: mgc, sgc

-N, node size of original graphs

-n, node size of coarse graphs

-p, edge probability between nodes within the same blocks

-q, edge probability between nodes in different blocks

The default setting is 
```
python main_classification.py --sbm_type associative --method mgc --N 200 --n 10 --p 0.5 --q 0.1 --max_trials 10
```





