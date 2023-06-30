import random
import numpy as np

import torch

from Make_NewData_all_time_final import *

def set_random_seed(seed = 0):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
      
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
total_input_data, smiles, label, drug, patient_info, route_info, norm_time_info, norm_pk_info = make_input()
