import torch
import torch.nn as nn

import dgl
from dgllife.model.model_zoo.attentivefp_predictor import AttentiveFPPredictor

### GNN Model
class GnnModel(nn.Module):
  def __init__(self, node_feat_size, edge_feat_size, n_layers, n_timesteps, graph_feat_size, dropout, output_size):
    super(GnnModel, self).__init__()

    ### Use model(= AttentiveFPPredictor)
    self.AttentiveFP = AttentiveFPPredictor(node_feat_size = node_feat_size,
                                       edge_feat_size = edge_feat_size,
                                       num_layers = n_layers, ### Need HyperParameter tunning
                                       num_timesteps = n_timesteps,### Need HyperParameter tunning
                                       graph_feat_size = graph_feat_size,### Need HyperParameter tunning
                                       dropout = dropout,### Need HyperParameter tunning
                                       n_tasks = output_size)

  def forward(self, graph, atom_feats, bond_feats):

    ### make drug feature
    gnn_out = self.AttentiveFP(graph, atom_feats, bond_feats)

    return gnn_out
