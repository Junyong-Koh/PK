import time
import random
import numpy as np
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

from sklearn.model_selection import KFold

from Make_NewData_all_time_final import *
from model import *

### Set seed
def set_random_seed(seed = 0):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
      
### Calculate Time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
      
def make_dataloader(input_data, batch_size):
  ### Total Data
  NumOfData = len(input_data)
  ### Train Data
  NumOfTrain = int(NumOfData * 0.8)
  ### Validation Data
  NumOfVal = int(NumOfData * 0.1)
  ### Test Data
  NumOfTest = NumOfData - NumOfTrain - NumOfVal

  ### SHOW
  print("Num of Train: ", NumOfTrain)
  print("Num of val: ", NumOfVal)
  print("Num of Test: ", NumOfTest)

  ### Split Data
  Trainset, Valset, Testset = random_split(input_data, [NumOfTrain, NumOfVal, NumOfTest])
  ### Make Dataloader
  train_dataloader = DataLoader(Trainset, batch_size = batch_size, shuffle = True, drop_last = False)
  val_dataloader = DataLoader(Valset, batch_size = batch_size, shuffle = True, drop_last = False)
  test_dataloader = DataLoader(Testset, batch_size = batch_size, shuffle = True, drop_last = False)
  return train_dataloader, val_dataloader, test_dataloader

#############################################################################################################
### Parameter
### Data parameter
SEED = 123
BATCH_SIZE = 1

### GNN parameter
N_LAYERS_GNN = 2
N_TIMESTEPS = 4
DROPOUT = 0.3
GRAPH_FEAT = 128
GRAPH_OUT = 64

### RNN parameter
INPUT_DIM = 16
OUTPUT_DIM = 1
HID_DIM = 64
N_LAYERS_RNN = 1

N_EPOCHS = 250
BEST_VALID_LOSS = float('inf')
#############################################################################################################

def RUN():
  ### Set seed
  set_random_seed(SEED)

  #############################################################################################################
  ### MODEL
  ### Load GNN
  gnn_model = GnnModel(node_feat_size = 39, edge_feat_size = 11,
                       n_layers = N_LAYERS_GNN, n_timesteps = N_TIMESTEPS,
                       graph_feat_size = GRAPH_FEAT, dropout = DROPOUT, output_size = GRAPH_OUT)

  ### Load PKMODEL
  pkmodel = PKModel(input_dim = INPUT_DIM, output_dim = OUTPUT_DIM, hid_dim = HID_DIM, n_layers = N_LAYERS_RNN)

  ### Load Main model
  model = main_model(gnn_model, pkmodel)
  
  ### Set Optimizer Function
  optimizer = optim.Adam(model.parameters(), lr = 0.0005)

  ### Set Loss Function
  criterion = nn.MSELoss()
  
  early_stopping = EarlyStopping(patience = 30, verbose = True, path = "FINAL_MODEL.pt")
  ##############################################################################################################
  
  ##############################################################################################################
  ### DATA
  ### Load data
  total_input_data, smiles, label, drug, patient_info, route_info, norm_pk_info = make_input(print_mode=0)
  
  ### random dataloader
  # train_dataloader, val_dataloader, test_dataloader = make_dataloader(total_input_data, BATCH_SIZE)
  
  ### Train and Test data index
  selected_train_index = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 
                          22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
                          41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
                          60, 62, 63, 64, 65, 67, 68, 69, 70, 72, 73, 74, 76, 77, 78, 79, 80, 81, 
                          82, 83, 84, 85, 86, 87, 89, 91, 92, 93, 94, 95, 97]
  selected_test_index = [1, 19, 24, 42, 61, 66, 71, 75, 88, 90, 96]
  Trainset = total_input_data[selected_train_index]
  test_dataloader = DataLoader(total_input_data[selected_test_index], batch_size = BATCH_SIZE, shuffle = True, drop_last = False) ### FIX
  
  ### K-fold cross validation
  kfold = KFold(n_splits = 5, shuffle = True)
  ##############################################################################################################

  ##############################################################################################################
  ### TRAIN AND VALIDATION
  epochs = []
  train_list = []
  val_list = []

  for epoch in range(N_EPOCHS):

    start_time = time.time()
    fold_train = []
    fold_valid = []
    for _, (train_index, val_index) in enumerate(kfold.split(Trainset)):
      train_dataloader = DataLoader(Trainset[train_index], batch_size = BATCH_SIZE, shuffle = True, drop_last = False)
      val_dataloader = DataLoader(Trainset[val_index], batch_size = BATCH_SIZE, shuffle = True, drop_last = False)

      train_loss = train(model, train_dataloader, smiles, optimizer, criterion)
      valid_loss = evaluate(model, val_dataloader, smiles, criterion)

      fold_train.append(train_loss)
      fold_valid.append(valid_loss)

    ### Average train and test loss
    real_train_loss = sum(fold_train) / len(fold_train)
    real_valid_loss = sum(fold_valid) / len(fold_valid)

    epochs.append(epoch)
    train_list.append(real_train_loss)
    val_list.append(real_valid_loss)

    end_time = time.time()

    ### Calculate time
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    ### Show result
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {real_train_loss:.6f} | Train PPL: {math.exp(real_train_loss):7.6f}')
    early_stopping(real_valid_loss, model)

    if early_stopping.early_stop:
      print("Early stopping")
      break
  ##############################################################################################################
  
  ##############################################################################################################
  ### TEST(colab - visualization)
  test_gnn_model = GnnModel(node_feat_size = 39, edge_feat_size = 11,
                            n_layers = N_LAYERS_GNN, n_timesteps = N_TIMESTEPS,
                           graph_feat_size = GRAPH_FEAT, dropout = DROPOUT, output_size = GRAPH_OUT)

  ### Load PKMODEL
  test_pkmodel = PKModel(input_dim = INPUT_DIM, output_dim = OUTPUT_DIM, hid_dim = HID_DIM, n_layers = N_LAYERS_RNN)

  ### Load Main model
  test_model = main_model(test_gnn_model, test_pkmodel)
  
  test(test_model, "FINAL_MODEL.pt", test_dataloader, smiles, label, drug, patient_info, route_info, norm_pk_info)
  ##############################################################################################################