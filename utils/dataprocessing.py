import numpy as np
np.random.seed(3)
from numpy import rot90
import gc

from tensorflow import data
from tensorflow import keras
import tensorflow as tf

def chunk_generator(files, batch_size, N, cubic_rotations, shuffling=False):


    while True:
        if shuffling:
            np.random.shuffle(files)
        for f in files:
            fin = np.load(f, 'r')
            length = len(fin)*N
            molecules = fin['molecule'][:, :N, ... ].reshape(length, 40, 40, 40, 7) # make (length*N, 40, 40, 40, 7)
            labels = np.repeat(fin['target'], N)
            
            num_samples = labels.shape[0]
            if shuffling:
                indices = np.random.permutation(num_samples)
            else:
                indices = [i for i in range(num_samples)]

            for start_idx in range(0, num_samples, batch_size):
                end_idx = start_idx + batch_size
                batch_indices = indices[start_idx:end_idx]
                x = molecules[batch_indices]
                if cubic_rotations:
                    x = random_rotation(x)

                y = labels[batch_indices]

                yield (x, y)


def get_chunked_generator(files, batch_size, N, cubic_rotations, shuffling=False):


    output_types = (float, float)
    tfdata_generator = data.Dataset.from_generator(chunk_generator, args=[
                                                 files, batch_size, N, cubic_rotations, shuffling],
                                                 output_types=output_types)

    return tfdata_generator
    
def random_rotation(polycube):
    """List all 24 rotations of the given 3d array"""
    def rotations(polycube, i, axes):
        """List the four rotations of the given 3d array in the plane spanned by the given axes."""
        return rot90(polycube, i, axes)

    r = np.random.randint(6) #rotation type
    rot_degree = np.random.randint(4)

    if r == 0:
        return rotations(polycube,  rot_degree, (2,3))
    elif r == 1:
        return rotations(rot90(polycube, 2, axes=(1,3)), rot_degree, (2,3))
    elif r == 2:
        return rotations(rot90(polycube, axes=(1, 3)), rot_degree, (1,2))
    elif r == 3:
        return rotations(rot90(polycube, -1, axes=(1,3)), rot_degree, (1,2))
    elif r == 4:
        return rotations(rot90(polycube, axes=(1,2)), rot_degree, (1,3))
    elif r == 5:
        return rotations(rot90(polycube, -1, axes=(1,2)), rot_degree, (1,3))