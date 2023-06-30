import random
import numpy as np

import torch
from torch.utils.data import random_split, DataLoader

from Make_NewData_all_time_final import *

def set_random_seed(seed = 0):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)

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
      
### Data parameter
SEED = 123
BATCH_SIZE = 1

### GNN parameter
N_LAYERS_GNN = 2
N_TIMESTEPS = 5
DROPOUT = 0.1
GRAPH_FEAT = 64

### RNN parameter
INPUT_DIM = 16
OUTPUT_DIM = 1
HID_DIM = 64
N_LAYERS_RNN = 1

N_EPOCHS = 250
BEST_VALID_LOSS = float('inf')

### Set seed
set_random_seed(SEED)

### Load data
total_input_data, smiles, label, drug, patient_info, route_info, norm_time_info, norm_pk_info = make_input(print_mode=0)

### CASE1: Use processed dataset
train_dataloader = torch.load("TRAIN_DATA.pth")
val_dataloader = torch.load("VAL_DATA.pth")
test_dataloader = torch.load("TEST_DATA.pth")

'''
### CASE2: Use random dataset
train_dataloader, val_dataloader, test_dataloader = make_dataloader(total_input_data, BATCH_SIZE)
'''
