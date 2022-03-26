import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, PandasTools
from rdkit.Chem.AllChem import UFFGetMoleculeForceField, Conformer
from rdkit.Chem.rdMolTransforms import TransformConformer, CanonicalizeMol, CanonicalizeConformer

from moleculekit.smallmol.smallmol import SmallMol
from moleculekit.tools.voxeldescriptors import getVoxelDescriptors, viewVoxelFeatures
from moleculekit.util import uniformRandomRotation

from utils.molprocessing import conf_energies, generate_representation_v2, voxelize_molecule

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',        default='SMRT_set.csv', type=str)
parser.add_argument('--target_column',  default='RT', type=str)
parser.add_argument('--output',         default='metlin', type=str)
parser.add_argument('--rotation',       default=1, type=int)
args = parser.parse_args()

#load dataset
script_path = Path(__file__).resolve().parent
data_path   = script_path.joinpath("data/")

mol_dir    = data_path.joinpath("precomputed_molecules/{}".format(args.output))
mol_dir.mkdir(parents=True, exist_ok=True)
tensor_dir = data_path.joinpath("precomputed_tensors/{}".format(args.output))
tensor_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_path.joinpath(args.dataset), index_col=0)
df['Molecule'] = df['SMILES'].apply(Chem.MolFromSmiles)

#make and voxelize conformations for each molecule
tp = np.dtype([('molecule', np.float16, (4, 40, 40, 40, 7)), ('idx', 'S10'), ('target', np.float16), ('smile',   'S315')])
X = np.zeros(1, dtype=tp)

for idx in tqdm(df.index.values):
    m = df.loc[idx, 'Molecule']
    l = df.loc[idx, args.target_column]
    canon_smi = Chem.MolToSmiles(m)

    voxelized_mols = [voxelize_molecule(m, box_size=[20, 20, 20], rotation=args.rotation) for j in range(4)]
    vox_conformations = [conf[0] for conf in voxelized_mols]
    for conf in vox_conformations:
        if conf is None:
            print("No enough conformations for: ", canon_smi)
            continue

    molecules = [conf[2] for conf in voxelized_mols]

    vox_conformations = np.stack(vox_conformations, axis=0) #(num_of_conf, h, w, d, channels)

    X[0] = (vox_conformations, idx, l, canon_smi)

    for count, m in enumerate(molecules):
        m.write(mol_dir.joinpath('{}_{}.mol2'.format(args.output, idx, count)))
    
    np.save(tensor_dir.joinpath('{}.npy'.format(idx)), X)

