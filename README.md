# CPORT

CPORT model predicts the retentition time of molecule using its 3D-conformers.
![CPORT](https://github.com/MarkZaretskiy/cport/blob/master/pics/CPORT_final_pic_v3.png)

# Setup
You need `python>3.6` to run this scripts.

This project depends upon the `TensorFlow 2.4`, `rdkit 2019.03.5`, `moleculekit 0.1.30` and other standard libraries like `pandas`, `numpy`, etc.

Install them with conda in virtual env:

`conda create --name cport_env --file requirements_conda.txt`

`conda activate cport_env`


For the installation recipe of rdkit please read the following instructions: https://www.rdkit.org/docs/Install.html

We tested our code on Ubuntu (16.04 and 18.04) with 2019.03.5 version of rdkit

# Preprocessing

To prepare a library of conformer for each molecule use the following script. Be careful, it needs ~300GB

`python3 prepare_conformations.py --dataset SMRT_set.csv --target_column RT --output metlin --rotation 1`

# Training

Before training it is essential to stack voxelized molecules into chunks to speed up the speed of data transferring to GPU during training

`python3 stack_tensors.py --mol_dir data/precomputed_tensors/metlin --scaffold 0`

`python3 stack_tensors.py --mol_dir data/precomputed_tensors/metlin --scaffold 1`

To launch trainings use the following script using best parameters from grid search

`python3 train.py --json_arguments args/random_model.json`

`python3 train.py --json_arguments args/scaffold_model.json`

# Pretrained weights

Pretrained weights of models trained on random and scaffold splits of the METLIN dataset are available at: 
https://drive.google.com/drive/folders/1baoG0etkFmX904T8LCSmkLejZhvOjQnu?usp=sharing

Put them into the weights/

# Scoring

For scoring metlin molecules run the following scripts:

`python3 test.py --weights weights/model_random --mol_dir precomputed_tensors/metlin --output random_predictions.csv`

`python3 test.py --weights weights/model_scaffold --mol_dir precomputed_tensors/metlin --output scaffold_predictions.csv`

You can also score your own .sdf file using the following command:

`python3 sdf2rt.py --weights weights/model_random --sdf your_sdf_file.sdf --output your_sdf_predictions.csv --augmentation 1 --robustness 1`

Be careful, molecules which could not be parsed with RDKit would be ommited in the output .csv 


# Fine-Tuning

For preparing conformations for transferlearning run the following:

`python3 prepare_conformations.py --dataset predret_filtered.csv --target_column RT --output predret --rotation 1`

`python3 prepare_conformations.py --dataset inhouse_set.csv --target_column RT_C8  --output inhouse_c8 --rotation 1`

`python3 prepare_conformations.py --dataset inhouse_set.csv --target_column RT_C18 --output inhouse_c18 --rotation 1`

To stack prepared tensors in chunks for effective training use the following script:

`python3 prepare_transferlearning_chunks.py`

`python3 stack_tensors.py --mol_dir data/precomputed_tensors/inhouse_c8 --scaffold 0`

`python3 stack_tensors.py --mol_dir data/precomputed_tensors/inhouse_c18 --scaffold 0`

To fine-tune model pretrained on random split for new chromatography conditions use:

`python3 chroma_transferlearning.py --json_arguments transfer_args/args_0.json`

`python3 chroma_transferlearning.py --json_arguments transfer_args/args_1.json`

`python3 chroma_transferlearning.py --json_arguments transfer_args/args_2.json`

`python3 chroma_transferlearning.py --json_arguments transfer_args/args_3.json`

`python3 chroma_transferlearning.py --json_arguments transfer_args/args_4.json`

`python3 chroma_transferlearning.py --json_arguments transfer_args/args_5.json`

`python3 chroma_transferlearning.py --json_arguments transfer_args/args_6.json`

`python3 chroma_transferlearning.py --json_arguments transfer_args/args_7.json`

`python3 chroma_transferlearning.py --json_arguments transfer_args/args_8.json`

`python3 chroma_transferlearning.py --json_arguments transfer_args/args_9.json`

`python3 chroma_transferlearning.py --json_arguments transfer_args/args_10.json`

`python3 chroma_transferlearning.py --json_arguments transfer_args/args_11.json`

`python3 chroma_transferlearning.py --json_arguments transfer_args/args_12.json`

`python3 chroma_transferlearning.py --json_arguments transfer_args/args_13.json`

`python3 chroma_transferlearning.py --json_arguments transfer_args/args_14.json`

`python3 chroma_transferlearning.py --json_arguments transfer_args/args_15.json`

`python3 chroma_transferlearning.py --json_arguments transfer_args/args_16.json`
