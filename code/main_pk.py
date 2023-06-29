from Make_NewData_all_time_final.py import *

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
