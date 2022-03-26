from utils.resnet_model import get_model
from utils.dataprocessing import chunk_generator, get_chunked_generator
from utils.custom_callbacks import GarbageCollector, custom_mse

import tensorflow as tf

from glob import glob
import numpy as np
np.random.seed(0)
import argparse
from pathlib import Path
import json

gpus = tf.config.experimental.list_physical_devices("GPU")

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

parser = argparse.ArgumentParser(description='Train model with specific arguments from the grid')
parser.add_argument('--json_arguments', type=str, required=True,
                    help='json file with specified model`s parameters')
args = parser.parse_args()

with open(args.json_arguments, 'rt') as f:
    args = json.load(f)

N = args['N']
BATCH_SIZE = 128 
NAME = args['model_name']
THRESHOLD = args['thr']
MOL_DIR = Path(args['directory'])
N_epochs = args['epochs']
SCAFFOLD = args['scaffold']
try: #cubic augmentation randomly choose from 24 possible cubic orientations
    CUBIC_AUGMENTATION = args['cubic'] 
except:
    CUBIC_AUGMENTATION = True

###data preprocessing
train_files = list(map(str, list(MOL_DIR.joinpath("train").glob("*.npy"))))
test_files = list(map(str, list(MOL_DIR.joinpath("test").glob("*.npy"))))

print("train chunks: ", len(train_files))
print("test chunks: ", len(test_files))

steps_per_epoch = 0
for f in train_files:
    steps_per_epoch += len(np.load(f, 'r'))
steps_per_epoch = int(np.ceil(steps_per_epoch * N / BATCH_SIZE))
val_steps = 0
for f in test_files:
    val_steps += len(np.load(f, 'r'))
val_steps = int(np.ceil(val_steps / BATCH_SIZE))
del f
###generators
train_generator = get_chunked_generator(train_files, batch_size=BATCH_SIZE, N=N, cubic_rotations=CUBIC_AUGMENTATION,)
test_generator = get_chunked_generator(test_files, batch_size=BATCH_SIZE, N=1, cubic_rotations=False,)

###define loss function
if THRESHOLD:
    print("use custom loss")
    loss = custom_mse(thr=THRESHOLD)
else:
    loss = tf.keras.losses.MeanSquaredError()

###model
model = get_model()
model._average_over_rotations = False
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(0.01),  # Optimizer
    # Loss function to minimize
    loss=loss,
    # List of metrics to monitor
    metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanAbsolutePercentageError()],
)

###callbacks
script_path = Path(__file__).resolve().parent
subgrid_name = str(MOL_DIR).split("/")[-1]
if SCAFFOLD:
    subgrid_name += "_scaffold"
log_output = script_path.joinpath("supp/logs/{}".format(subgrid_name)).joinpath(NAME + ".csv")
log_output.resolve().parent.mkdir(parents=True, exist_ok=True)
log_output.touch(exist_ok=True)

lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='mean_absolute_percentage_error', factor=0.2, patience=int(16 / N), verbose=0, mode='auto',
    min_delta=0.5, cooldown=10, min_lr=1e-4
)
csv_logger = tf.keras.callbacks.CSVLogger(log_output, separator=',', append=False)
garbage_collector = GarbageCollector()

checkpoint_filepath = script_path.joinpath("supp/checkpoints/{}".format(subgrid_name)).joinpath(NAME)
checkpoint_filepath.mkdir(parents=True, exist_ok=True)
print('CHECKPOINT PATH: ', checkpoint_filepath)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_mean_absolute_error',
    mode='min',
    save_best_only=True)

#fit
model.fit(train_generator, epochs=N_epochs, steps_per_epoch=steps_per_epoch,\
    validation_data=test_generator, validation_steps=val_steps,
    callbacks=[lr_reduce, csv_logger, garbage_collector, checkpoint_callback],
    shuffle=False, 
    workers=4,
    use_multiprocessing=True,
    max_queue_size=20, 
    )
