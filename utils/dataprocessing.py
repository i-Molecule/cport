import numpy as np
np.random.seed(3)
from numpy import rot90
import gc

from tensorflow import data
from tensorflow import keras
import tensorflow as tf

def chunk_generator(files, batch_size, N, cubic_rotations):


    while True:
        np.random.shuffle(files)
        for f in files:
            fin = np.load(f, 'r')
            length = len(fin)*N
            molecules = fin['molecule'][:, :N, ... ].reshape(length, 40, 40, 40, 7) # make (length*N, 40, 40, 40, 7)
            labels = np.repeat(fin['target'], N)

            batches = int(np.ceil(length / batch_size))
            
            for i in range(batches):
                
                x = molecules[i*batch_size:(i+1)*batch_size]
                if cubic_rotations:
                    x = random_rotation(x)
                
                y = labels[i*batch_size:(i+1)*batch_size]
                    
                yield (x, y)

def get_chunked_generator(files, batch_size, N, cubic_rotations):


    output_types = (float, float)
    tfdata_generator = data.Dataset.from_generator(chunk_generator, args=[
                                                 files, batch_size, N, cubic_rotations],
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