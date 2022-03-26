import pandas as pd
from tqdm import tqdm
import numpy as np
from time import time
import argparse
tqdm.pandas()

from rdkit import Chem
from rdkit.Chem import AllChem, PandasTools
from rdkit.Chem.AllChem import UFFGetMoleculeForceField, Conformer
from rdkit.Chem.rdMolTransforms import TransformConformer, CanonicalizeMol, CanonicalizeConformer
from rdkit.Chem.Descriptors import ExactMolWt

from moleculekit.smallmol.smallmol import SmallMol
from moleculekit.tools.voxeldescriptors import getVoxelDescriptors, viewVoxelFeatures
from moleculekit.util import uniformRandomRotation
from pathlib import Path

CHUNK_DIR = Path('data/precomputed_chunks')
TENSOR_DIR = Path('data/precomputed_tensors')

#load dataset
df = pd.read_csv("data/predret_dataset.csv", index_col=0)

#systems with enough molecules for transferlearning
all_systems = df['System'].value_counts()
all_systems = all_systems[all_systems >= 400]
print(all_systems)

#make directory for each dataset
for s in all_systems.index:
    CHUNK_DIR.joinpath(s).joinpath('train').mkdir(parents=True, exist_ok=True)
    CHUNK_DIR.joinpath(s).joinpath('test').mkdir(parents=True, exist_ok=True)
    CHUNK_DIR.joinpath(s).joinpath('splits').mkdir(parents=True, exist_ok=True)

#make and voxelize conformations for each molecule
for s in all_systems.index:


    tp = np.dtype([('molecule', '<f2', (4, 40, 40, 40, 7)), ('table_idx', int), ('target', '<f4'), ('smile',   'S315')]) 
    X = np.zeros(all_systems[s], dtype=tp)
    indices = df[df['System'] == s].index.values.copy().tolist()

    np.random.shuffle(indices)
    train_indices = indices[:len(indices) * 3 // 4]
    test_indices = indices[len(indices) * 3 // 4:]
    
    np.save(CHUNK_DIR.joinpath(s).joinpath('splits/{}_test_indices.npy'.format(s)), test_indices)
    np.save(CHUNK_DIR.joinpath(s).joinpath('splits/{}_train_indices.npy'.format(s)), train_indices)

    print('System: ', s)
    print('Molecules: ', len(indices))
    count = 0
    for j, idx in tqdm(enumerate(train_indices)):

        f = np.load(TENSOR_DIR.joinpath("predret/{}.npy".format(idx)))
        X[count] = f
        count += 1

    print('Saved train molecules: ', count)
    np.save(CHUNK_DIR.joinpath(s).joinpath('train/chunk_0.npy'), X[:count])

    count = 0
    for j, idx in tqdm(enumerate(train_indices)):

        f = np.load(TENSOR_DIR.joinpath("predret/{}.npy".format(idx)))
        X[count] = f
        count += 1

    print('Saved test molecules: ', count)
    np.save(CHUNK_DIR.joinpath(s).joinpath('test/chunk_0.npy'), X[:count])
