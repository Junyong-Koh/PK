import random
import numpy as np

import torch
import torch.nn as nn

from torchmetrics import R2Score

import dgl
from dgllife.model.model_zoo.attentivefp_predictor import AttentiveFPPredictor

# import matplotlib.pyplot as plt

### GNN Model
class GnnModel(nn.Module):
  def __init__(self, node_feat_size, edge_feat_size, n_layers, n_timesteps, graph_feat_size, dropout, output_size):
    super(GnnModel, self).__init__()

    ### Use model(= AttentiveFPPredictor)
    self.AttentiveFP = AttentiveFPPredictor(node_feat_size = node_feat_size,
                                       edge_feat_size = edge_feat_size,
                                       num_layers = n_layers, ### Number of GNN layers(Need HyperParameter tunning)
                                       num_timesteps = n_timesteps,### Times of updating the graph representations with GRU(Need HyperParameter tunning)
                                       graph_feat_size = graph_feat_size,### Size for the learned graph representations(Need HyperParameter tunning)
                                       dropout = dropout,### Probability for performing the dropout(Need HyperParameter tunning)
                                       n_tasks = output_size)

  def forward(self, graph, atom_feats, bond_feats):

    ### Make drug feature
    gnn_out = self.AttentiveFP(graph, atom_feats, bond_feats)

    return gnn_out

### RNN Model
class PKModel(nn.Module):
  def __init__(self, input_dim, output_dim, hid_dim, n_layers):
    super(PKModel, self).__init__()

    ### Embedding_layer(for patient) 
    ### 'Y,NR' - 0, 'Y,N' - 1, 'NR,NR' - 2, 'NR,N' - 3, 'sickle cell disease,Y' - 4, 'NR,Y' - 5, 'Y,Y' - 6, 'N,NR' - 7
    self.embedding_layer = nn.Embedding(num_embeddings = 8, embedding_dim = 3)

    ### Preprocess layer(time, pk, dose + add_info(patient + route))
    self.lin_pre = nn.Linear(7, input_dim)
    
    ### RNN layers(GRU)
    self.gru = nn.GRU(input_dim, hid_dim, n_layers)

    ### Linear layer
    self.lin_out = nn.Linear(hid_dim, output_dim)

  def forward(self, timepoints, pk_data, input_len, emb_patient, drug_route, dose, smiles, teacher_forcing_ratio = 0.5):

    ### Number of timepoints
    input_len = input_len

    ### batch_size(1)
    batch_size = 1

    ### Tensor to save results [input_len, batch_size(1)]
    outputs = torch.zeros(input_len, batch_size)

    ### 1st timepoint and pk
    pk = pk_data[0]
    timepoint = timepoints[0]
    
    ### Save 1st pk
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
      pk = pk.unsqueeze(1)
      timepoint = timepoint.unsqueeze(1)

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
      ### rnn_output = [1, batch_size(1), hid dim]

      ### rnn_output: [1, batch_size(1), hid dim] -> [batch_size(1), 1] -> [batch_size(1)]
      predicted_pk = self.lin_out(rnn_output.squeeze(0))
      predicted_pk = predicted_pk.squeeze(1)

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

    ### Load GNN and PKMODEL
    self.gnn = gnn
    self.pkmodel = pkmodel

  def forward(self, timepoints, pk_data, input_len, emb_patient_info, drug_route_info, dose, graph, atom_feats, bond_feats, teacher_forcing_ratio):

    ### GNN
    gnn_out = self.gnn(graph, atom_feats, bond_feats) ### Make Drug feature
    ### RNN
    output = self.pkmodel(timepoints, pk_data, input_len, emb_patient_info, drug_route_info, dose, gnn_out, teacher_forcing_ratio)

    return gnn_out, output

class EarlyStopping:
  """EarlyStopping handler can be used to stop the training if no improvement after a given number of events"""
  def __init__(self, patience = 7, verbose = False, delta = 0, path = 'checkpoint.pt'):
      """
      Args:
          patience (int): Number of events to wait if no improvement and then stop the training
                          Default: 7
          verbose (bool): displays messages when the callback takes an action
                          Default: False
          delta (float): A minimum increase in the score to qualify as an improvement
                          Default: 0
          path (str): checkpoint path
                      Default: 'checkpoint.pt'
      """
      self.patience = patience
      self.verbose = verbose
      self.counter = 0
      self.best_score = None
      self.early_stop = False
      self.val_loss_min = np.Inf
      self.delta = delta
      self.path = path

  def __call__(self, val_loss, model):

      score = -val_loss

      if self.best_score is None:
          self.best_score = score
          self.save_checkpoint(val_loss, model)
      elif score < self.best_score + self.delta:
          self.counter += 1
          print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
          if self.counter >= self.patience:
              self.early_stop = True
      else:
          self.best_score = score
          self.save_checkpoint(val_loss, model)
          self.counter = 0

  def save_checkpoint(self, val_loss, model):
      if self.verbose:
          print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
      torch.save(model.state_dict(), self.path)
      self.val_loss_min = val_loss


### TRAIN
def train(model, iterator, smiles, optimizer, criterion):

    ### Set Train Mode
    model.train()
    ### Save Loss
    epoch_loss = 0

    for batch_id, batch_data in enumerate(iterator):
      ### Extract Node and Edge Feature
      smiles_index = int(batch_data[:,42])

      _, bg, _, _ = smiles[smiles_index]

      atom_feats = bg.ndata['h']
      bond_feats = bg.edata['e']

      ### batch_data.shape: torch.Size([batch_size(1), 84])
      ### Timepoints: [batch_size(1), timepoints(19)] -> [timepoints(19), batch_size(1)]: normalized time value
      timepoints = batch_data[:, 0:19]
      timepoints = torch.transpose(timepoints, 0, 1)

      ### pk_data: [batch_size(1), pks(19)] -> [pks(19), batch_size(1)]: label value
      pk_data = batch_data[:, 19:38]
      pk_data = torch.transpose(pk_data, 0, 1)

      ### time_len: [1]
      input_len = int(batch_data[:, 38])

      ### patient, drug
      emb_patient_info = int(batch_data[:, 39])
      drug_route = int(batch_data[:, 40])

      ### dose: [1]
      dose = batch_data[:, 41]

      ### Run
      gnn_out, predicted = model(timepoints, pk_data, input_len, emb_patient_info, drug_route, dose, bg, atom_feats, bond_feats, teacher_forcing_ratio = 0.5)
      
      ### [1:] - Exclude starting point
      predicted = predicted[1:input_len].view(-1)
      label = batch_data[:, 20:19+input_len].view(-1)
    
      ### Calculate Loss
      loss = criterion(predicted, label)

      ### Sets the gradients of all optimized torch.Tensors to zero.
      optimizer.zero_grad()

      ### Computes the gradient of current tensor w.r.t. graph leaves.
      loss.backward(retain_graph = True)

      ### Updates the parameters
      optimizer.step()

      epoch_loss += loss.item()

    return epoch_loss / len(iterator)
  
### VALIDATION
def evaluate(model, iterator, smiles, criterion):

  ### validation
  model.eval()
  epoch_loss = 0

  for batch_id, batch_data in enumerate(iterator):
      ### Extract Node and Edge Feature
      smiles_index = int(batch_data[:,42])

      _, bg, _, _ = smiles[smiles_index]

      atom_feats = bg.ndata['h']
      bond_feats = bg.edata['e']

      ### batch_data.shape: torch.Size([batch_size(1), 84])
      ### Timepoints: [batch_size(1), timepoints(19)] -> [timepoints(19), batch_size(1)]: normalized time value
      timepoints = batch_data[:, 0:19]
      timepoints = torch.transpose(timepoints, 0, 1)

      ### pk_data: [batch_size(1), pks(19)] -> [pks(19), batch_size(1)]: label value
      pk_data = batch_data[:, 19:38]
      pk_data = torch.transpose(pk_data, 0, 1)

      ### time_len: [1]
      input_len = int(batch_data[:, 38])

      ### patient, drug
      emb_patient_info = int(batch_data[:, 39])
      drug_route = int(batch_data[:, 40])

      ### dose: [1]
      dose = batch_data[:, 41]

      ### Run
      gnn_out, predicted = model(timepoints, pk_data, input_len, emb_patient_info, drug_route, dose, bg, atom_feats, bond_feats, teacher_forcing_ratio = 1)

      predicted = predicted[1:input_len].view(-1)
      label = batch_data[:, 20:19+input_len].view(-1)
      
      ### Calculate Loss
      loss = criterion(label, predicted)

      epoch_loss += loss.item()

  return epoch_loss / len(iterator)

### TEST
def test(model, model_path, test_data, smiles, label_title, drug_title, patient_info, route_info, norm_pk_info):
  ### Load the model that we saved at the end of the training loop
  model.load_state_dict(torch.load(model_path))
  ### Test Mode
  model.eval()

  acc_list = []
  r2score = R2Score()

  with torch.no_grad():
      for batch_id, batch_data in enumerate(test_data):

        ### Extract Node and Edge Feature
        smiles_index = int(batch_data[:,42])

        _, bg, _, _ = smiles[smiles_index]

        atom_feats = bg.ndata['h']
        bond_feats = bg.edata['e']

        ### batch_data.shape: torch.Size([batch_size(1), 84])
        ### Timepoints: [batch_size(1), timepoints(19)] -> [timepoints(19), batch_size(1)]: normalized time value
        timepoints = batch_data[:, 0:19]
        timepoints = torch.transpose(timepoints, 0, 1)

        ### pk_data: [batch_size(1), pks(19)] -> [pks(19), batch_size(1)]: label value
        pk_data = batch_data[:, 19:38]
        pk_data = torch.transpose(pk_data, 0, 1)

        ### time_len: [1]
        input_len = int(batch_data[:, 38])

        ### patient, drug
        emb_patient_info = int(batch_data[:, 39])
        drug_route = int(batch_data[:, 40])

        ### dose: [1]
        dose = batch_data[:, 41]

        ### Run
        gnn_out, predicted = model(timepoints, pk_data, input_len, emb_patient_info, drug_route, dose, bg, atom_feats, bond_feats, teacher_forcing_ratio = 1)

        ### predicted: [pks(19), batch_size(1)] -> [pks(num_pks), batch_size(1)] -> [batch_size(1), pks(num_pks)]
        ### label: [batch_size(1), pks(19)] -> [batch_size(1), pks(num_pks)]
        predicted = torch.transpose(predicted[:input_len], 0, 1)
        label = batch_data[:,19:19+input_len]

        if (batch_id == 0):
          ### (1, input_len-1)
          total_predicted = predicted[0, 1:]
          ### (1, input_len-1)
          total_label = label[0, 1:]
          f, axes = plt.subplots(2, 5)
          f.set_size_inches((20, 10))
          plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
          
        else:
          ### Concat Data
          total_predicted = torch.cat([total_predicted, predicted[0, 1:]], dim = -1)
          total_label = torch.cat([total_label, label[0, 1:]], dim = -1)

        ### Solve noramliazation
        predicted = predicted * norm_pk_info[1] + norm_pk_info[0]
        predicted = torch.exp(predicted)
        
        real_label = batch_data[:,65:65+input_len]

        acc_list.append(r2score(predicted[0, 1:]/1000000, real_label[0, 1:]))

        ### real_timepoints: [batch_size(1), input_len]
        real_timepoints = batch_data[:, 44:44+input_len]

        ### data_index
        data_index = int(batch_data[:, 43])

        ### patient_index
        patient_index = int(batch_data[:, 63])

        ### route_index
        route_index = int(batch_data[:, 64])

        if (batch_id % 10 == 0 and batch_id != 0):
          plt.show()
          f, axes = plt.subplots(2, 5)
          f.set_size_inches((20, 10))
          plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

        ### Plot
        x = real_timepoints[0].tolist()
        y1 = real_label[0]
        y1 = y1.tolist()
        y2 = predicted[0] / 1000000
        y2 = y2.tolist()

        axes[int((batch_id%10)/5), (batch_id%10)%5].plot(x, y1, "ro-", x, y2, "bo-")
        axes[int((batch_id%10)/5), (batch_id%10)%5].set_title(label_title[int(data_index)] + "\n" + drug_title[smiles_index] + "\n" + patient_info[int(patient_index)] + ", " + route_info[int(route_index)] + "\n" + str(dose) +  ", " + str(int(data_index)))

  plt.show()
  #print("R2SCORE: ", sum(acc_list) / len(acc_list))
  #total_predicted = total_predicted.contiguous().view(-1)
  #total_label = total_label.contiguous().view(-1)
