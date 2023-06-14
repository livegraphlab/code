# Temporal Link Prediction and Node Classification
We construct our models based on the source code provided by Roland due to its efficiency, which is publicly avaliable at [here](https://github.com/snap-stanford/roland).

## Execuate
We provide the `yaml` file for parameter settings. 

### Temporal Link Prediction
For link prediction task, please set task as follows:
```
dataset:
	task: link_pred
	node_label: None
``` 

### Temporal Node Classification
For node classification task, please set task and node_label as follows:
```
dataset:
	task: node
	node_label: node-label.csv
``` 

For different time granularities, set `snapshot_freq` in `transaction` as `D`, `W` and `M` to represent `day snapshot`, `week snapshot` and `month snapshot`, respectively. For different GNN models, set `layer_type` in `gnn` with the following types: `residual_edge_conv`, `evolve_gcn_h`, `evolve_gcn_o`, `tgcn`, `gconv_gru`, `gconv_lstm`.

### Run
Just execuate the following command:
```
python main.py --cfg './example.yaml' --repeat 3
```