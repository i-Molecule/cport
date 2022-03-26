#!/bin/sh

#how to train model on a random split of randomly rotated molecules from metlin

#generate conformations for molecules from the metlin dataset, voxelize them and save
#need ~300GB of disk be careful!
python3 prepare_metlin_conformations_and_tensors.py --rotation 1

#generate chunks from precomputed tensors with voxelized molecules for more efficient batching per training
#takes another 300GB
python3 stack_tensors.py --mol_dir data/precomputed_tensors/metlin --scaffold 0

#training on GPU
#takes ~two days on 1080Ti
python3 train.py --json_arguments args/random_model.json

#scoring
python3 test.py --weights weights/model_random --mol_dir precomputed_tensors/metlin




### scoring compounds from PredRet database
#preparing conformations for datasets from predret database
python3 prepare_predret_conformations_and_tensors.py --rotation 1

#score predret using CPORT model
python3 score_predret.py --weights weights/model_scaffold --mol_dir data/precomputed_tensors/predret/ --output predret_predictions_scaffold.csv
python3 score_predret.py --weights weights/model_random --mol_dir data/precomputed_tensors/predret/ --output predret_predictions_random.csv


### scoring inhouse dataset
python3 prepare_inhouse_conformations_and_tensors --rotation 1

python3 score_inhouse.py --weights weights/model_scaffold --mol_dir data/precomputed_tensors/inhouse/   --output inhouse_predictions_scaffold.csv
python3 score_inhouse.py --weights weights/model_random   --mol_dir data/precomputed_tensors/predret/   --output inhouse_predictions_random.csv


#transferlearning
#prepare conformations for transferlearning
python3 prepare_transferlearning_conformations.py

python3 chroma_transferlearning.py --json_arguments transfer_args/args_0.json
python3 chroma_transferlearning.py --json_arguments transfer_args/args_1.json
python3 chroma_transferlearning.py --json_arguments transfer_args/args_2.json
python3 chroma_transferlearning.py --json_arguments transfer_args/args_3.json
python3 chroma_transferlearning.py --json_arguments transfer_args/args_4.json
python3 chroma_transferlearning.py --json_arguments transfer_args/args_5.json
python3 chroma_transferlearning.py --json_arguments transfer_args/args_6.json
python3 chroma_transferlearning.py --json_arguments transfer_args/args_7.json
python3 chroma_transferlearning.py --json_arguments transfer_args/args_8.json
python3 chroma_transferlearning.py --json_arguments transfer_args/args_9.json
python3 chroma_transferlearning.py --json_arguments transfer_args/args_10.json
python3 chroma_transferlearning.py --json_arguments transfer_args/args_11.json
python3 chroma_transferlearning.py --json_arguments transfer_args/args_12.json
python3 chroma_transferlearning.py --json_arguments transfer_args/args_13.json
python3 chroma_transferlearning.py --json_arguments transfer_args/args_14.json
python3 chroma_transferlearning.py --json_arguments transfer_args/args_15.json
python3 chroma_transferlearning.py --json_arguments transfer_args/args_16.json





