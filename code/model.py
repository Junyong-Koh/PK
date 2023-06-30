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

### RNN Model
class PKModel(nn.Module):
  def __init__(self, input_dim, output_dim, hid_dim, n_layers):
    super(PKModel, self).__init__()

    ### embedding_layer(for patient)
    self.embedding_layer = nn.Embedding(num_embeddings = 8, embedding_dim = 3)

    ### preprocess layer(time, pk, dose + add_info)
    self.lin_pre = nn.Linear(7, input_dim)
    
    ### RNN layers(GRU)
    self.gru = nn.GRU(input_dim, hid_dim, n_layers)

    ### linear layer
    self.lin_out = nn.Linear(hid_dim, output_dim)

  def forward(self, timepoints, pk_data, input_len, emb_patient, drug_route, dose, smiles, teacher_forcing_ratio = 0.5):

    ### Number of timepoints
    input_len = input_len

    ### batch_size(1)
    batch_size = 1

    ### Tensor to save results [input_len, batch_size(1)]
    outputs = torch.zeros(input_len, batch_size)

    ### 1st timepoint and concentration
    pk = pk_data[0]
    timepoint = timepoints[0]
    outputs[0] = pk

    ### dose: [batch_size(1)] -> [batch_size(1), 1]
    dose = dose.unsqueeze(1)

    ### patient: [batch_size(1), 3], drug_route: [batch_size(1), 1]
    patient = self.embedding_layer(torch.tensor([emb_patient]))
    drug_route = torch.tensor([drug_route])
    drug_route = drug_route.unsqueeze(0)

    ### add_info: [batch_size(1), 4]
    add_info = torch.cat([patient, drug_route], dim = -1)

    ### smiles: [batch_size(1), hid_dim]
    ### hidden: [batch_size(1), hid_dim] -> [1, batch_size(1), hid_dim]
    hidden = smiles
    hidden = hidden.unsqueeze(0)

    for time in range(1, input_len):

      ### Chagne pk and timepoint vectors shape
      ### [batch_size(1)] -> [batch_size(1), 1]
      pk = pk.unsqueeze(0)
      timepoint = timepoint.unsqueeze(0)

      ### Concat pk, timepoint, dose, add_info
      ### [batch_size(1), 7]
      input = torch.cat([pk, timepoint, dose, add_info], dim = -1)

      ### [batch_size(1), 7] -> [batch_size(1), 16]
      input = self.lin_pre(input)

      ### [1, batch_size(1), input_dim]
      input = input.unsqueeze(0)

      ### Insert input PK value, previous hidden states
      ### Receive output tensor (predictions) and new hidden state
      rnn_output, hidden = self.gru(input, hidden)

      ### Results
      ### rnn_output = [1, batch_size(1), hid dim]

      ### rnn_output: [1, batch_size(1), hid dim] -> [batch_size(1), 1] -> [batch_size(1)]
      predicted_pk = self.lin_out(rnn_output.squeeze(0))
      predicted_pk = predicted_pk.squeeze(1)

      ### Results
      ### predicted_pk = [batch_size(1)]

      ### Save output
      outputs[time] = predicted_pk

      ### Decide if we are going to use teacher forcing or not(used in train)
      teacher_force = random.random() < teacher_forcing_ratio

      ### Set next pk and timepoint
      if (teacher_force):
        pk = predicted_pk
      else:
        pk = pk_data[time]
      timepoint = timepoints[time]

    return outputs

### Total Model
class main_model(nn.Module):
  def __init__(self, gnn, pkmodel):
    super().__init__()

    self.gnn = gnn
    self.pkmodel = pkmodel

  def forward(self, timepoints, pk_data, input_len, emb_patient_info, drug_route_info, dose, graph, atom_feats, bond_feats, teacher_forcing_ratio):

    ### GNN
    gnn_out = self.gnn(graph, atom_feats, bond_feats) ### Make Graph Feature
    ### RNN
    output = self.pkmodel(timepoints, pk_data, input_len, emb_patient_info, drug_route_info, dose, gnn_out, teacher_forcing_ratio)

    return gnn_out, output
