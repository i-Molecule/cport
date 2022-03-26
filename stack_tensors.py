import numpy as np
np.random.seed(0)

from pathlib import Path
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='stack .npy files with voxelized molecules into chunks')
parser.add_argument('--mol_dir', type=str, required=True,
                    help='directory with tensors')
parser.add_argument('--scaffold', type=int, default=0, required=False,
                    help='use precomputed scaffold split or not')
args = parser.parse_args()

SCAFFOLD = args.scaffold
N_chunk = 512
MOL_DIR = Path(args.mol_dir)    #Path("data/precomputed_tensors/rt_rdkit_rotated")
if not SCAFFOLD:
    SAVE_DIR = Path(args.mol_dir.replace("tensors", "chunks"))   #Path("data/precomputed_chunks/rt_rdkit_rotated")
else:
    SAVE_DIR = Path(args.mol_dir.replace("tensors", "chunks_scaffold"))
TRAIN_DIR = SAVE_DIR.joinpath('train')
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR = SAVE_DIR.joinpath('test')
TEST_DIR.mkdir(parents=True, exist_ok=True)

#load data
numpy_arrays = list(MOL_DIR.glob("*.npy"))
if not SCAFFOLD:
    print("random split will be used")
    np.random.shuffle(numpy_arrays)
    test_files = numpy_arrays[int(0.75*len(numpy_arrays)):]
    train_files = numpy_arrays[:int(0.75*len(numpy_arrays))]
else: #only for metlin
    numpy_arrays = list(MOL_DIR.glob("*.npy"))
    train_subset = np.load("data/splits/SMRT_scaffold_train.npy", allow_pickle=True)
    test_subset = np.load("data/splits/SMRT_scaffold_test.npy", allow_pickle=True)
    print("scaffold split will be used")
    train_files = [i for i in numpy_arrays if str(i).split("/")[-1][:-4] in train_subset]
    test_files = [i for i in numpy_arrays if str(i).split("/")[-1][:-4] in test_subset]
    print("train files: ", len(train_files))
    print("test files: ", len(test_files))


tp = np.dtype([('molecule', np.float16, (4, 40, 40, 40, 7)),\
                        ('target', np.float16),\
                        ('smile', 'S300')])

X = np.zeros(N_chunk, dtype=tp) #chunk


#save train files
chunks = int(np.ceil(len(train_files) / N_chunk))
chunk_count = 0
for i in tqdm(range(chunks)):
    count = 0
    for f in train_files[i*N_chunk:(i+1)*N_chunk]:
        
        array = np.load(f)
        
        X[count]['target'] = array['target']
        conformations = np.squeeze(array['molecule'], axis=0)
        if conformations.shape[0] != 4:
            conformations = np.tile(conformations, reps=(4, 1, 1, 1, 1))[:4] #copy up to 4

        X[count]['molecule'] = conformations
        X[count]['smile'] = array['smile'][0]
        count += 1
         
    np.save(TRAIN_DIR.joinpath("chunk_{}.npy".format(chunk_count)), X[:count])
    chunk_count += 1

#save test files
chunks = int(np.ceil(len(test_files) / N_chunk))
chunk_count = 0
for i in tqdm(range(chunks)):
    count = 0
    for f in test_files[i*N_chunk:(i+1)*N_chunk]:
        
        array = np.load(f)
        
        X[count]['target'] = array['target']
        conformations = np.squeeze(array['molecule'], axis=0)
        if conformations.shape[0] != 4:
            conformations = np.tile(conformations, reps=(4, 1, 1, 1, 1))[:4] #copy up to 4
        X[count]['molecule'] = conformations
        X[count]['smile'] = array['smile'][0]
        count += 1
        
    np.save(TEST_DIR.joinpath("chunk_{}.npy".format(chunk_count)), X[:count])
    chunk_count += 1
