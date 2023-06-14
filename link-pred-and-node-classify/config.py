import logging
import os
from yacs.config import CfgNode as CN

# Global config object
cfg = CN()


def set_cfg(cfg):
    r'''
    This function sets the default config value.
    1) Note that for an experiment, only part of the arguments will be used
    The remaining unused arguments won't affect anything.
    So feel free to register any argument in graphgym.contrib.config
    2) We support *at most* two levels of configs, e.g., cfg.dataset.name

    :return: configuration use by the experiment.
    '''

    # ------------------------------------------------------------------------ #
    # Basic options
    # ------------------------------------------------------------------------ #

    # Set print destination: stdout / file
    cfg.print = 'both'

    # Select device: 'cpu', 'cuda:0', 'auto'
    cfg.device = 'auto'

    # Output directory
    cfg.out_dir = 'results'

    # Config destination (in OUT_DIR)
    cfg.cfg_dest = 'config.yaml'

    # Random seed
    cfg.seed = 1

    # Print rounding
    cfg.round = 4

    # Tensorboard support for each run
    cfg.tensorboard_each_run = False

    # Tensorboard support for aggregated results
    cfg.tensorboard_agg = True

    # Additional num of worker for data loading
    cfg.num_workers = 0

    # Max threads used by PyTorch
    cfg.num_threads = 6

    # The metric for selecting the best epoch for each run
    cfg.metric_best = 'auto'

    # If visualize embedding.
    cfg.view_emb = False

    # ------------------------------------------------------------------------ #
    # Dataset options
    # ------------------------------------------------------------------------ #
    cfg.dataset = CN()

    # Name of the dataset
    cfg.dataset.name = 'Cora'

    cfg.dataset.split_method = 'chronological_temporal'

    # if PyG: look for it in Pytorch Geometric dataset
    # if NetworkX/nx: load data in NetworkX format
    cfg.dataset.format = 'PyG'

    # Dir to load the dataset. If the dataset is downloaded, this is the
    # cache dir
    cfg.dataset.dir = './datasets'

    # Task: node, edge, graph, link_pred
    cfg.dataset.task = 'node'

    # Type of task: classification, regression, classification_binary
    # classification_multi
    cfg.dataset.task_type = 'classification'

    # Transductive / Inductive
    # Graph classification is always inductive
    cfg.dataset.transductive = True

    # Split ratio of dataset. Len=2: Train, Val. Len=3: Train, Val, Test
    cfg.dataset.split = [0.8, 0.1, 0.1]

    # Whether shuffle dataset while split
    cfg.dataset.shuffle = True

    # Whether to use an encoder for the node features
    cfg.dataset.node_encoder = True

    # Name of node encoder
    cfg.dataset.node_encoder_name = 'Atom'

    # If add batchnorm after node encoder
    cfg.dataset.node_encoder_bn = True

    # Whether to use an encoder for the edge features
    cfg.dataset.edge_encoder = False

    # Name of edge encoder
    cfg.dataset.edge_encoder_name = 'Bond'

    # If add batchnorm after edge encoder
    cfg.dataset.edge_encoder_bn = True

    # Dimension of the encoded features.
    # For now the node and edge encoding dimensions
    # are the same.
    cfg.dataset.encoder_dim = 128

    # Dimension for edge feature. Updated by the real dim of the dataset
    cfg.dataset.edge_dim = 128

    # ============== Link/edge tasks only

    # all or disjoint
    cfg.dataset.edge_train_mode = 'all'

    # Used in disjoint edge_train_mode. The proportion of edges used for
    # message-passing
    cfg.dataset.edge_message_ratio = 0.8

    # The ratio of negative samples to positive samples
    cfg.dataset.edge_negative_sampling_ratio = 1.0

    cfg.dataset.node_label = None

    # ==============

    # feature augmentation
    # naming is consistent with DeepSNAP graph:
    # node_xxx is feature on nodes; edge_xxx is feature on edge; graph_xxx is
    # feature on graph.
    # a list of tuples (feature, feature_dim)
    cfg.dataset.augment_feature = []
    cfg.dataset.augment_feature_dims = []
    # 'balanced', 'equal_width', 'bounded', 'original', 'position'
    cfg.dataset.augment_feature_repr = 'original'

    # If non-empty, this replaces the label with structural features
    # For example, setting label = 'node_degree' causes the model to 
    # replace the node labels with node degrees (overwriting previous node
    # labels)
    # Note: currently only support 1 label
    cfg.dataset.augment_label = ''
    cfg.dataset.augment_label_dims = 0

    # What transformation function is applied to the dataset
    cfg.dataset.transform = 'none'

    # Whether cache the splitted dataset
    # NOTE: it should be cautiouslly used, as cached dataset may not have
    # exactly the same setting as the config file
    cfg.dataset.cache_save = False
    cfg.dataset.cache_load = False

    # Whether remove the original node features in the dataset
    cfg.dataset.remove_feature = False

    # Simplify TU dataset for synthetic tasks
    cfg.dataset.tu_simple = True

    # ------------------------------------------------------------------------ #
    # Training options
    # ------------------------------------------------------------------------ #
    cfg.train = CN()

    # Training (and validation) pipeline mode
    cfg.train.mode = 'standard'

    # Total graph mini-batch size
    cfg.train.batch_size = 16

    # Minibatch node
    cfg.train.sample_node = False

    # Num of sampled node per graph
    cfg.train.node_per_graph = 32

    # Radius: same, extend. same: same as cfg.gnn.layers_mp, extend: layers+1
    cfg.train.radius = 'extend'

    # Evaluate model on test data every eval period epochs
    cfg.train.eval_period = 10

    # Save model checkpoint every checkpoint period epochs
    cfg.train.ckpt_period = 100

    # Resume training from the latest checkpoint in the output directory
    cfg.train.auto_resume = False

    # The epoch to resume. -1 means resume the latest epoch.
    cfg.train.epoch_resume = -1

    # Clean checkpoint: only keep the last ckpt
    cfg.train.ckpt_clean = True

    cfg.train.stop_live_update_after = 99999999

    cfg.train.tbptt_freq = 5

    cfg.train.internal_validation_tolerance = 5

    # Computing MRR is slow in the baseline setting.
    # Only start to compute MRR in the test set range after certain time.
    cfg.train.start_compute_mrr = 0

    cfg.meta = CN()
    # Whether to do meta-learning via initialization moving average.
    # Default to False.
    cfg.meta.is_meta = False

    # choose between 'moving_average' and 'online_mean'
    cfg.meta.method = 'moving_average'
    # For online mean:
    # new_mean = (n-1)/n * old_mean + 1/n * new_value.
    # where *_mean corresponds to W_init.

    # Weight used in moving average for model parameters.
    # After fine-tuning the model in period t and get model M[t],
    # Set W_init = (1-alpha) * W_init + alpha * M[t].
    # For the next period, use W_init as the initialization for fine-tune
    # Set cfg.meta.alpha = 1.0 to recover the original algorithm.
    cfg.meta.alpha = 0.9


    # ------------------------------------------------------------------------ #
    # Validation options
    # ------------------------------------------------------------------------ #
    cfg.val = CN()

    # Minibatch node
    cfg.val.sample_node = False

    # Num of sampled node per graph
    cfg.val.node_per_graph = 32

    # Radius: same, extend. same: same as cfg.gnn.layers_mp, extend: layers+1
    cfg.val.radius = 'extend'

    # ------------------------------------------------------------------------ #
    # Model options
    # ------------------------------------------------------------------------ #
    cfg.model = CN()

    # Model type to use
    cfg.model.type = 'gnn'

    # Auto match computational budget, match upper bound / lower bound
    cfg.model.match_upper = True

    # Loss function: cross_entropy, mse
    cfg.model.loss_fun = 'cross_entropy'

    # size average for loss function
    cfg.model.size_average = True

    # Threshold for binary classification
    cfg.model.thresh = 0.5

    # ============== Link/edge tasks only
    # Edge decoding methods.
    #   - dot: compute dot(u, v) to predict link (binary)
    #   - cosine_similarity: use cosine similarity (u, v) to predict link (
    #   binary)
    #   - concat: use u||v followed by an nn.Linear to obtain edge embedding
    #   (multi-class)
    cfg.model.edge_decoding = 'dot'

    # Shape of the edge prediction, if 'all', then predict over all node pairs
    cfg.model.edge_pred_shape = 'label_index'
    # ===================================

    # ================== Graph tasks only
    # Pooling methods.
    #   - add: global add pool
    #   - mean: global mean pool
    #   - max: global max pool
    cfg.model.graph_pooling = 'add'
    # ===================================

    # ------------------------------------------------------------------------ #
    # GNN options
    # ------------------------------------------------------------------------ #
    cfg.gnn = CN()

    cfg.gnn.embed_update_method = 'mlp'
    cfg.gnn.skip_connection = 'affine'

    # Number of layers before message passing
    cfg.gnn.layers_pre_mp = 0

    # Number of layers for message passing
    cfg.gnn.layers_mp = 2

    # Number of layers after message passing
    cfg.gnn.layers_post_mp = 0

    # Hidden layer dim. Automatically set if train.auto_match = True
    cfg.gnn.dim_inner = 16

    # Type of graph conv: generalconv, gcnconv, sageconv, gatconv, ...
    cfg.gnn.layer_type = 'generalconv'

    # Stage type: 'stack', 'skipsum', 'skipconcat'
    cfg.gnn.stage_type = 'stack'

    # How many layers to skip each time
    cfg.gnn.skip_every = 1

    # Whether use batch norm
    cfg.gnn.batchnorm = True

    # Activation
    cfg.gnn.act = 'relu'

    # Dropout
    cfg.gnn.dropout = 0.0

    # Aggregation type: add, mean, max
    # Note: only for certain layers that explicitly set aggregation type
    # e.g., when cfg.gnn.layer_type = 'generalconv'
    cfg.gnn.agg = 'add'

    # Normalize adj
    cfg.gnn.normalize_adj = False

    # Message direction: single, both
    cfg.gnn.msg_direction = 'single'

    # Number of attention heads
    cfg.gnn.att_heads = 1

    # After concat attention heads, add a linear layer
    cfg.gnn.att_final_linear = False

    # After concat attention heads, add a linear layer
    cfg.gnn.att_final_linear_bn = False

    # Normalize after message passing
    cfg.gnn.l2norm = True

    # randomly use fewer edges for message passing
    cfg.gnn.keep_edge = 0.5

    cfg.gnn.only_update_top_state = False
    # Method to update node embedding from old node embedding and new node features.
    # Options: 'moving_average', 'masked_gru', 'gru'
    # moving average: new embedding = r * old + (1-r) * node_feature.
    # gru: new embedding = GRU(node_feature, old_embedding).
    # masked_gru: only apply GRU to active nodes.
    cfg.gnn.embed_update_method = 'moving_average'
    # what kind of GRU kernel to use if GRU is required for embedding updating.
    cfg.gnn.gru_kernel = 'linear'
    # how many layers to use in the MLP updater.
    # default: 1, use a simple linear layer.
    cfg.gnn.mlp_update_layers = 2

    # ------------------------------------------------------------------------ #
    # Optimizer options
    # ------------------------------------------------------------------------ #
    cfg.optim = CN()

    # optimizer: sgd, adam
    cfg.optim.optimizer = 'adam'

    # Base learning rate
    cfg.optim.base_lr = 0.01

    # L2 regularization
    cfg.optim.weight_decay = 5e-4

    # SGD momentum
    cfg.optim.momentum = 0.9

    # scheduler: none, steps, cos
    cfg.optim.scheduler = 'cos'

    # Steps for 'steps' policy (in epochs)
    cfg.optim.steps = [30, 60, 90]

    # Learning rate multiplier for 'steps' policy
    cfg.optim.lr_decay = 0.1

    # Maximal number of epochs
    cfg.optim.max_epoch = 200

    # ------------------------------------------------------------------------ #
    # Batch norm options
    # ------------------------------------------------------------------------ #
    cfg.bn = CN()

    # BN epsilon
    cfg.bn.eps = 1e-5

    # BN momentum (BN momentum in PyTorch = 1 - BN momentum in Caffe2)
    cfg.bn.mom = 0.1

    # ------------------------------------------------------------------------ #
    # Memory options
    # ------------------------------------------------------------------------ #
    cfg.mem = CN()

    # Perform ReLU inplace
    cfg.mem.inplace = False

        # example argument group
    cfg.transaction = CN()

    # whether use snapshot
    cfg.transaction.snapshot = False

    # snapshot split method 1: number of snapshots
    # split dataset into fixed number of snapshots.
    cfg.transaction.snapshot_num = 100

    # snapshot split method 2: snapshot frequency
    # e.g., one snapshot contains transactions within 1 day.
    cfg.transaction.snapshot_freq = 'D'

    cfg.transaction.check_snapshot = False

    # how to use transaction history
    # full or rolling
    cfg.transaction.history = 'full'


    # type of loss: supervised / meta
    cfg.transaction.loss = 'meta'

    # feature dim for int edge features
    cfg.transaction.feature_int_dim = 32
    cfg.transaction.feature_edge_int_num = [50, 8, 252, 252, 3, 3]
    cfg.transaction.feature_node_int_num = [0]

    # feature dim for amount (float) edge feature
    cfg.transaction.feature_amount_dim = 64

    # feature dim for time (float) edge feature
    cfg.transaction.feature_time_dim = 64

    #
    cfg.transaction.node_feature = 'raw'

    # how many days look into the future
    cfg.transaction.horizon = 1

    # prediction mode for the task; 'before' or 'after'
    cfg.transaction.pred_mode = 'before'

    # number of periods to be captured.
    # set to a list of integers if wish to use pre-defined periodicity.
    # e.g., [1,7,28,31,...] etc.
    cfg.transaction.time_enc_periods = [1]

    # if 'enc_before_diff': attention weight = diff(enc(t1), enc(t2))
    # if 'diff_before_enc': attention weight = enc(t1 - t2)
    cfg.transaction.time_enc_mode = 'enc_before_diff'

    # how to compute the keep ratio while updating the recurrent GNN.
    # the update ratio (for each node) is a function of its degree in [0, t)
    # and its degree in snapshot t.
    cfg.transaction.keep_ratio = 'linear'

    cfg.experimental = CN()

    # How many negative edges for each node to compute rank-based evaluation
    # metrics such as MRR and recall at K.
    # E.g., if multiplier = 1000 and a node has 3 positive edges, then we
    # compute the MRR using 1000 randomly generated negative edges
    # + 3 existing positive edges.
    cfg.experimental.rank_eval_multiplier = 100

    # Only use the first n snapshots (time periods) to train the model.
    # Empirically, the model learns rich dynamics from only a few periods.
    # Set to -1 if using all snapshots.
    cfg.experimental.restrict_training_set = -1

    # Whether to visualize edge attention of GNN layer after training.
    cfg.experimental.visualize_gnn_layer = False

    cfg.metric = CN()
    # how to compute MRR.
    # available: f = 'min', 'max', 'mean'.
    # Step 1: get the p* = f(scores of positive edges)
    # Step 2: compute the rank r of p* among all negative edges.
    # Step 3: RR = 1 / rank.
    # Step 4: average over all users.
    # expected MRR(min) <= MRR(mean) <= MRR(max).
    cfg.metric.mrr_method = 'max'

    # Specs for the link prediction task using BSI dataset.
    # All units are days.
    cfg.link_pred_spec = CN()

    # The period of `today`'s increase: how often the system is making forecast.
    # E.g., when = 1,
    # the system forecasts transactions in upcoming 7 days for everyday.
    # One training epoch loops over
    # {Jan-1-2020, Jan-2-2020, Jan-3-2020..., Dec-31-2020}
    # When = 7, the system makes prediction every week.
    # E.g., the system forecasts transactions in upcoming 7 days
    # on every Monday.
    cfg.link_pred_spec.forecast_frequency = 1

    # How many days into the future the model is trained to predict.
    # The model forecasts transactions in (today, today + forecast_horizon].
    # NOTE: forecast_horizon should >= forecast_frequency to cover all days.
    cfg.link_pred_spec.forecast_horizon = 7


def assert_cfg(cfg):
    """Checks config values invariants."""
    if cfg.dataset.task not in ['node', 'edge', 'graph', 'link_pred']:
        raise ValueError('Task {} not supported, must be one of'
                         'node, edge, graph, link_pred'.format(cfg.dataset.task))
    if 'classification' in cfg.dataset.task_type and cfg.model.loss_fun == \
            'mse':
        cfg.model.loss_fun = 'cross_entropy'
        logging.warning(
            'model.loss_fun changed to cross_entropy for classification.')
    if cfg.dataset.task_type == 'regression' and cfg.model.loss_fun == \
            'cross_entropy':
        cfg.model.loss_fun = 'mse'
        logging.warning('model.loss_fun changed to mse for regression.')
    if cfg.dataset.task == 'graph' and cfg.dataset.transductive:
        cfg.dataset.transductive = False
        logging.warning('dataset.transductive changed to False for graph task.')
    if cfg.gnn.layers_post_mp < 1:
        cfg.gnn.layers_post_mp = 1
        logging.warning('Layers after message passing should be >=1')


set_cfg(cfg)
