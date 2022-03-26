import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
import argparse

import numpy as np
import pandas as pd

from utils.resnet_model import get_model
from utils.dataprocessing import chunk_generator, get_chunked_generator

import argparse
from tqdm import tqdm
from pathlib import Path
script_path = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description='Path to weights and precomputed tensors')
parser.add_argument('--weights', type=str, help='path to weights')
parser.add_argument('--mol_dir', type=str, help='path to directory with precomputed tensors')
parser.add_argument('--output', type=str, help='path to .csv with predictions')
args = parser.parse_args()

####create model and load weights
model = get_model()
model.load_weights(args.weights)
model._average_over_rotations = True

####score numpy arrays
arrays = list(Path(args.mol_dir).glob("*.npy")) #get all .npy files with molecules
print("{} molecules will be scored".format(len(arrays)))
y_preds = []
y_trues = []
smiles = []
ids = []

for a in tqdm(arrays):

    f = np.load(a)
    
    molecule =f['molecule'].squeeze() # (4, 40, 40, 40, 7)
    labels = np.repeat(f['target'], 4).tolist() 
    indices = np.repeat(f['idx'], 4).tolist()
    smis = np.repeat(f['smile'][0].decode('utf-8'), 4)
    predictions = model.predict_on_batch(molecule)
    
    y_preds.extend(predictions.flatten().tolist())
    y_trues.extend(labels)
    ids.extend(indices)
    smiles.extend(smis)

df = pd.DataFrame(None)
df['y_preds'] = y_preds
df['y_trues'] = y_trues
df['idx'] = ids
df['smiles'] = smiles

df.to_csv(script_path.joinpath(args.output))

