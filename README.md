# Live-Graph-Lab
This is the source code for our experiments.

## Requirements
* python3.8
* pytorch==1.11.0
* torch-scatter==2.0.9
* torch-sparse==0.6.13
* torch-cluster==1.6.0
* torch-geometric==2.0.4
* networkx==2.8.6
* numpy==1.21.5
* matplotlib==3.5.3
* pandas==1.4.2
* scipy==1.9.1
* deepsnap==0.2.1
* snap

## Metadata 
We extract all the blocks before Aug 1st, 2022 (i.e., from block #0 to block #15,255,104). Then, we parse all the transaction data and log data via toolkit Ethereum ETL. Through this way, we obtain a network with more than 4.5 million nodes and 124 million edges. The data are stored in a tabular format with the following headers: `collection address`, `block number`, `from address`, `to address`, `token id`, `transaction hash`, `value` and `timestamp`. The metadata is available at [here](https://drive.google.com/file/d/1lyCcfGbmU0eW7aHijKSMvmVqBVmwTsmV).

## Analysis
All the codes for graph analysis are included in folder `./analysis`. Just execuate each python file to generate the figures. For example, you can reproduce Figure 1 and Figure 3 by running `local-graph-properties.py` and `global-graph-properties.py`, respectively.


## Temporal Link Prediction and Node classification
For link prediction task, we remove all the transactions associated with the `Null` address, which results in 3.13 million nodes and 23.13 million edges in the directed graph. The data file includes four columns: `from id`, `to id`, `edge weight` and `timestamp`. The processed dataset is available at [here](https://drive.google.com/file/d/1kEIJYp126nnC8L7lTv-yQaNalW7j4EFO).

For node classification task, we first filter out nodes that only have one transaction. Then, each node' maximum transaction interval is calculated. If the maximum interval is within one day, we call it daily trader. Likewise, if the maximum interval is within one week and larger than one day, we call it weekly trader, and so forth. This process results in a large-scale directed graph with about 1.80 million nodes and 21.83 million edges. The processed dataset is available at [here](https://drive.google.com/file/d/15-v2wywEf-vobQw63t8bPno2Qj-1Cd_q) and [here](https://drive.google.com/file/d/1XbjRHK87-FDtk5k2DPtyLkyNiRtnIdAr).

For detailed experiment settings, please refer to `README.md` in `./link-pred-and-node-classify`.

## Continuous Subgraph Matching
Similar to link prediction task, we also remove all the transactions associated with the `Null` address. We use NFT transactions from year 2017 to the end of 2021 as the initial graph, and then the transactions in the year of 2022 are regarded as the insertion streams. Since the original nodes and edges are unlabeled, we randomly assign one of 30 labels to each node, and we do not assign labels for edges and we use the edges' directions as labels. The data are structured as follows:

### Query Graph
Each line in the query graph file represent a vertex or an edge.

* A vertex is represented by `v <vertex-id> <vertex-label>`.
* An edge is represented by `e <vertex-id-1> <vertex-id-2> <edge-label>`.

The two endpoints of an edge must appear before the edge. For example,
```
v 0 0
v 1 0
e 0 1 0
v 2 1
e 0 2 1
e 2 1 2
```

### Initial Data Graph
The initial data graph file has the same format as the query graph file.

### Graph Update Stream
Graph update stream is a collection of insertions and deletions of a vertex or an edge.

* A vertex insertion is represented by `v <vertex-id> <vertex-label>`.
* A vertex deletion is represented by `-v <vertex-id> <vertex-label>`.
* An edge insertion is represented by `e <vertex-id-1> <vertex-id-2> <edge-label>`.
* An edge deletion is represented by `-e <vertex-id-1> <vertex-id-2> <edge-label>`.

The vertex or edge to be deleted must exist in the graph, and the label must be the same as that in the graph. If an edge is inserted to the data graph, both its endpoints must exist. For example,
```
v 3 1
e 2 3 2
-v 2 1
-e 0 1 0
```
The graph datasets and their corresponding querysets used in our paper can be downloaded [here](https://drive.google.com/drive/folders/12rLiMMV1CslbEy__MvuaT5DYrSqtJSK-).

For detailed experiment settings, please refer to `README.md` in `./csm`.

