import pandas as pd
import os
from collections import OrderedDict


NUM_PATCHES = 300
BATCH = 5
IMG_SIZE = (128, 128)
LR = 0.001
EPOCHS = 15

PATCH_DIR = '/home/ubuntu/Documents/internship/dataset/patches'
# split = 80

train_set, val_set = pd.read_csv('../dataset/train.csv'), pd.read_csv('../dataset/val.csv')

if not os.path.exists('logs'):
    os.makedirs('logs')

TRACKER_CSV_NAME = 'model_comp.csv'

if os.path.exists(TRACKER_CSV_NAME):
    track = pd.read_csv(TRACKER_CSV_NAME)
else:
    track = pd.DataFrame()

track_dict = OrderedDict({
    'Model': 'uni_update_max',
    'resolution': IMG_SIZE[0],
    'Pretrained layer changed': 0,
    'Batch': BATCH,
    'LR': LR,
    'Num patch': NUM_PATCHES,
    'Best Train C-ind': 0,
    'Best Val C-ind': 0,
    'Least loss': 1e9,
})

SAVE_PATH = f'logs/{track_dict["Model"]}.pth'
