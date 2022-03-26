import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
import argparse

import pandas as pd
import numpy as np

from tqdm import tqdm

from utils.molprocessing import conf_energies, generate_representation_v2, voxelize_molecule
from utils.resnet_model import get_model

from rdkit import Chem

BATCH_SIZE = 128

#arguments
parser = argparse.ArgumentParser(description='Path to weights and precomputed tensors')
parser.add_argument('--weights', type=str, help='path to weights')
parser.add_argument('--sdf', type=str, help='path to a sdf file')
parser.add_argument('--output', type=str, help='path to .csv with predictions')
parser.add_argument('--augmentation', type=int, help='if True use 4 conformer, if False 1')
parser.add_argument('--robustness', type=int, help='use averaging over 24 orientations or not')

args = parser.parse_args()

if args.augmentation:
    print("Use 4 conformers for scoring")
    N = 4
else:
    print("Use single conformer for scoring")
    N = 1

####create model and load weights
print("Creating a model . . . ")
model = get_model()
model.load_weights(args.weights)
if args.robustness:
    print("average over 24 orientations for each conformer")
    model._average_over_rotations = True
else:
    print("NO averaging over cubic orientations")
    model._average_over_rotations = False

#read sdf file
print("Reading .sdf file . . . ")
suppl = Chem.SDMolSupplier(args.sdf)
mols = []
for i, x in tqdm(enumerate(suppl)):
    mols.append(x)
print("{} molecules were read".format(len(mols)))

#make predictions one by one
print("Inference . . . ")
predictions = []
B = int(np.ceil(len(mols) / BATCH_SIZE))
for i in tqdm(range(B)):
    
    batch_mols = mols[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
    M = len(batch_mols) #can be le than BATCH_SIZE for last batch
    batch = np.zeros((N * M, 40, 40, 40, 7))

    count = 0
    for m in batch_mols:
        for j in range(N):
            tensor, _, _ = voxelize_molecule(m, box_size=[20, 20, 20], rotation=True)
            if tensor is None:
                print("Conformer can not be generated for: ", Chem.MolToSmiles(m))
            batch[count] = tensor
            count += 1

    y_pred = model.predict_on_batch(batch)
    y_pred = np.mean(y_pred.reshape(-1, N), axis=1) #average over N conformations
    
    predictions.extend(y_pred)

smiles = [Chem.MolToSmiles(m) for m in mols]
df = pd.DataFrame(zip(smiles, predictions), columns=['smiles', 'preds'])
df['rank'] = np.array(predictions).flatten().argsort().argsort()
df.to_csv(args.output)

