import os

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
SCHEDULER = 'cosine'
DEVICES = 4
WORKERS = 16
MIXUP = .5
MIXUP_ELEMENTS = 2

ARCHITECTURE = 'llama'
EMA = True
QAT = False
MAX_LEN = 512
N_PARAMETERS = 10E6 / 1 / {
    'lstm': .5
}.get(ARCHITECTURE, 1)
N_COORDS = 2
KV_SIZE = 64
MLP_RATIO = 4
DEPTH2WIDTH_RATIO = .52

N_HEADS = 6
N_LAYERS = 5
N_FEATURES = N_HEADS * KV_SIZE

CAUSAL_FRACTION = 0 / 8
CAUSAL_FORESIGHT = 2 if CAUSAL_FRACTION else 0

DROPOUT = 0.1

assert OPTIMIZER in ['sgd', 'adamw', 'lion']
assert SCHEDULER in ['1cycle', 'cosine']

if not WORKERS:
    BATCH_SIZE = 7
