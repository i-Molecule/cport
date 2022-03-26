from utils.resnet_model import get_model
from utils.custom_callbacks import GarbageCollector, custom_mse

import tensorflow as tf

from glob import glob
import pandas as pd
import numpy as np
np.random.seed(0)
import argparse
from pathlib import Path
import json

from scipy.stats import spearmanr

gpus = tf.config.experimental.list_physical_devices("GPU")

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

parser = argparse.ArgumentParser(description='Train model with specific arguments')
parser.add_argument('--json_arguments', type=str, required=True,
                    help='json file with specified model`s parameters')
args = parser.parse_args()

with open(args.json_arguments, 'rt') as f:
    args = json.load(f)

N = args['N']
BATCH_SIZE = 128 
NAME = args['model_name']
THRESHOLD = 0
MOL_DIR = Path(args['directory'])
N_epochs = 60
WEIGHTING = 0
PRETRAINED_MODEL = args['pretrained_model']
SCAFFOLD = 0

###data preprocessing
f = np.load(MOL_DIR.joinpath('train/chunk_0.npy'))
smiles_train = f['smile']
smiles_train = [s.decode('utf-8') for s in smiles_train]
try:
    table_idx_train = f['table_idx']
except:
    table_idx_train = []
    print('no table idx train')

X_train = f['molecule'].reshape(len(f)*4, 40, 40, 40, 7)
y_train = np.repeat(f['target'], N)

f = np.load(MOL_DIR.joinpath('test/chunk_0.npy'))
smiles_test = f['smile']
smiles_test = [s.decode('utf-8') for s in smiles_test]
try:
    table_idx_test = f['table_idx']
except:
    table_idx_test = []
    print('no table idx test')
X_test = f['molecule'].reshape(len(f)*4, 40, 40, 40, 7)
y_test = np.repeat(f['target'], N)

#free memory
del f

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
    optimizer=tf.keras.optimizers.RMSprop(1e-4),  # Optimizer
    # Loss function to minimize
    loss=loss,
    # List of metrics to monitor
    metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.RootMeanSquaredError()],
    #steps_per_execution=batch_accumulated
)

model.load_weights(PRETRAINED_MODEL)

#reinitialize last layer
last_layer_weights = np.random.normal(loc=0.0, scale=np.sqrt(2/100), size=(100, 1)) #he_normal initialization
last_layes_bias = np.array([1.0])
model.layers[-1].set_weights((last_layer_weights, last_layes_bias))

model.optimizer.learning_rate.assign(1e-2) #

###callbacks
script_path = Path(__file__).resolve().parent
log_output = script_path.joinpath("output/logs/transferlearning/").joinpath(NAME + "_{}.csv".format(N_epochs))
log_output.resolve().parent.mkdir(parents=True, exist_ok=True)
log_output.touch(exist_ok=True)
final_metrics_output = script_path.joinpath("output/logs/transferlearning/").joinpath(NAME + "_{}.json".format(N_epochs))
final_predictions_output = script_path.joinpath("output/logs/transferlearning/").joinpath(NAME + "_{}_predictions.csv".format(N_epochs))

csv_logger = tf.keras.callbacks.CSVLogger(log_output, separator=',', append=False)
garbage_collector = GarbageCollector()

checkpoint_filepath = script_path.joinpath("output/checkpoints/transferlearning").joinpath(NAME)
checkpoint_filepath.mkdir(parents=True, exist_ok=True)
print('CHECKPOINT PATH: ', checkpoint_filepath)

#lr_scheduler = CustomCosineAnnealingWithWarmUp(base_lr=0.01, end_lr=0.0001, warmup_steps=steps_per_epoch*2, decay_steps=(N_epochs-5)*steps_per_epoch)
lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='mean_absolute_error', factor=0.2, patience=5, verbose=0, mode='auto',
    min_delta=5, cooldown=1, min_lr=1e-4
)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_mean_absolute_error',
    mode='min',
    save_best_only=True)

#fit
model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=N_epochs, \
    validation_data=(X_test, y_test),
    callbacks=[lr_reduce, csv_logger, garbage_collector, checkpoint_callback],
    shuffle=False, 
    workers=4,
    use_multiprocessing=True,
    max_queue_size=20, 
    )


#check final results with averaging
model.load_weights(checkpoint_filepath)
model._average_over_rotations = True
df = pd.DataFrame(None)

#return averaged over 24 cube orientations
y_pred = model.predict(X_test)

#average over 4 conformations
y_pred = y_pred.reshape(-1, 4).mean(axis=1)
y_test = y_test.reshape(-1, 4).mean(axis=1)

errors = np.abs(y_pred - y_test)

test_mae = errors.mean()
test_medae = np.median(errors.flatten())
test_mape = np.mean(np.abs((y_pred - y_test) / y_test))
test_spr = spearmanr(y_pred, y_test)[0]

print('smile_test:', len(smiles_test))
print('table_idx_test:', len(table_idx_test))
print('unique in table idx: ', len(set(table_idx_test)))
print('y_pred:', len(y_pred))
print('y_test', len(y_test))

count = 0
for pred, label, smi in zip(y_pred, y_test, smiles_test):
    df.loc[count, 'y_true'] = label
    df.loc[count, 'y_pred'] = pred
    df.loc[count, 'smiles'] = smi
    df.loc[count, 'test'] = 1
    count += 1

print("after scoring test")
print(df.head())
print(df.shape)

#free memory
del X_test, y_test

#return NON-averaged(for the sake of speed) over 24 cube orientations
y_pred = model.predict(X_train)

#average over 4 conformations
y_pred = y_pred.reshape(-1, 4).mean(axis=1)
y_train = y_train.reshape(-1, 4).mean(axis=1)

errors = np.abs(y_pred - y_train)

train_mae = errors.mean()
train_medae = np.median(errors.flatten())
train_mape = np.mean(np.abs((y_pred - y_train) / y_train))
train_spr = spearmanr(y_pred, y_train)[0]

for pred, label, smi in zip(y_pred, y_train, smiles_train):
    df.loc[count, 'y_true'] = label
    df.loc[count, 'y_pred'] = pred
    df.loc[count, 'smiles'] = smi
    df.loc[count, 'test'] = 0
    count += 1

print("after scoring train")
print(df.head())
print(df.tail())
print(df.shape)
print("length of train predictions: ", len(y_train))

df.to_csv(final_predictions_output)
results = {'test_mae':str(test_mae), 'test_medae':str(test_medae), 'test_mape':str(test_mape), 'test_spearman':str(test_spr),
            'train_mae':str(train_mae), 'train_medae':str(train_medae), 'train_mape':str(train_mape), 'train_spearman':str(train_spr),
}

with open(final_metrics_output, 'w') as fout:
    json.dump(results, fout)
    print("final metrics saved at: ", str(final_metrics_output))
