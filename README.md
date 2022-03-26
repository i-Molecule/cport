# CPORT

CPORT (Conformation-based Prediction Of Retention Time) is a deep learning model to predict the retentition time of molecules using its 3D-conformers.

Check our manuscript: M. Zaretskii, I. Bashkirova, S. Osipenko, Y. Kostyukevich, E. Nikolaev, and P. Popov, "3D chemical structures allow robust deep learning models for retention time prediction", 2022

![CPORT](https://github.com/i-Molecule/cport/blob/master/pics/CPORT_final_pic_v3.png)

# Setup

For proper installation of CPORT you need `python>=3.6` `TensorFlow 2.4`, `rdkit 2019.03.5`, `moleculekit 0.1.30` and other commonly used python packages listed in requirements_conda.txt

Use these commands to create a virtual environment with the required packages:

`conda create --name cport_env --file requirements_conda.txt`

`conda activate cport_env`

We tested our code on Ubuntu (16.04 and 18.04) with rdkit 2019.03.5

# Preprocessing

The following script will generated 3D conformers from the SMILES.
Note, that for SMRT dataset it will take ~300GB of the disk space.

`python3 prepare_conformations.py --dataset SMRT_set.csv --target_column RT --output metlin --rotation 1`

# Training

Before training it is essential to stack voxelized molecules into chunks to speed up data transferring to GPU during training

`python3 stack_tensors.py --mol_dir data/precomputed_tensors/metlin --scaffold 0`

`python3 stack_tensors.py --mol_dir data/precomputed_tensors/metlin --scaffold 1`

To launch trainings use the following script using best parameters from grid search

`python3 train.py --json_arguments args/random_model.json`

`python3 train.py --json_arguments args/scaffold_model.json`

# Pretrained weights

Pretrained weights of models trained on random and scaffold splits of the METLIN dataset are available at: 
https://drive.google.com/drive/folders/1baoG0etkFmX904T8LCSmkLejZhvOjQnu?usp=sharing

Put them into the `weights` directory within this repo.

# Scoring

To predict the retention times use the following scripts:

`python3 test.py --weights weights/model_random --mol_dir precomputed_tensors/metlin --output random_predictions.csv`

`python3 test.py --weights weights/model_scaffold --mol_dir precomputed_tensors/metlin --output scaffold_predictions.csv`

You can also screen an .sdf file using:

`python3 sdf2rt.py --weights weights/model_random --sdf your_sdf_file.sdf --output your_sdf_predictions.csv --augmentation 1 --robustness 1`

Note, that in rare cases molecules might not be processed with RDKit, such molecules will be ommited in the output .csv 


# Fine-Tuning

To prepare conformations for the transfer learning use the following scripts:

`python3 prepare_conformations.py --dataset predret_filtered.csv --target_column RT --output predret --rotation 1`

`python3 prepare_conformations.py --dataset inhouse_set.csv --target_column RT_C8  --output inhouse_c8 --rotation 1`

`python3 prepare_conformations.py --dataset inhouse_set.csv --target_column RT_C18 --output inhouse_c18 --rotation 1`

To stack prepared tensors in chunks for effective training use the following script:

`python3 prepare_transferlearning_chunks.py`

`python3 stack_tensors.py --mol_dir data/precomputed_tensors/inhouse_c8 --scaffold 0`

`python3 stack_tensors.py --mol_dir data/precomputed_tensors/inhouse_c18 --scaffold 0`

To fine-tune model pretrained on random split for new chromatography conditions use:

`python3 chroma_transferlearning.py --json_arguments transfer_args/args_0.json`

# Citation

If you use our article of this repository please cite:

M. Zaretskii, I. Bashkirova, S. Osipenko, Y. Kostyukevich, E. Nikolaev, and P. Popov, "3D chemical structures allow robust deep learning models for retention time prediction", 2022
