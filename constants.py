import os

from experimental.scaling import get_best_scale

TASK = ['find_lr', 'train', 'val'][1]
DATASET_PATH = r'C:\Data\asl'
MODELS_PATH = DATASET_PATH + '_models'
external = os.path.join(DATASET_PATH, 'external')
EXTERNAL_DATASETS = [
    # os.path.join(external, 'lsfb', 'train.csv'),
    # os.path.join(external, 'signasl', 'train.csv'),
    # os.path.join(external, 'signingsavvy', 'train.csv'),
    # os.path.join(external, 'wlasl', 'train.csv'),
]
CHECKPOINT_PATH = ''
TEACHER_PATH = ''
RESUME = False
EPOCHS = 100
STEPS_PER_EPOCH = 1024 * 1 // 1
BATCH_SIZE = 256
ACCUMULATE = 1
LR = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.
OPTIMIZER = 'lion'
SCHEDULER = '1cycle'
DEVICES = 1
WORKERS = 0
MIXUP = .6

EMA = True
QAT = False
MAX_LEN = 256
N_PARAMETERS = 10E6 / 5
N_COORDS = 2
ARCHITECTURE = 'transformer'
KV_SIZE = 64
MLP_RATIO = 4
DEPTH2WIDTH_RATIO = .52

N_HEADS, N_LAYERS = get_best_scale(N_PARAMETERS, DEPTH2WIDTH_RATIO, 192, MAX_LEN, 250, KV_SIZE, MLP_RATIO)
N_FEATURES = N_HEADS * KV_SIZE

CAUSAL_FRACTION = 0 / 8
CAUSAL_FORESIGHT = 2 if CAUSAL_FRACTION else 0

DROPOUT = 0.

assert OPTIMIZER in ['sgd', 'adamw', 'lion', 'radam']
assert SCHEDULER in ['1cycle', 'cosine']

if not WORKERS:
    BATCH_SIZE = 7
