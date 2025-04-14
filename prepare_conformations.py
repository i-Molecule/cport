import multiprocessing
import os
from typing import Any, Union
import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse
from pathlib import Path

from rdkit import Chem

from utils.molprocessing import voxelize_molecule


def voxelize_and_save(
    m: Chem.Mol,
    l: float,
    idx: Any,
    rotation: Union[bool, int],
    mol_dir: Union[str, os.PathLike],
    tensor_dir: Union[str, os.PathLike],
    save_smiles: bool = False,
    save_mols: bool = False,
):

    # make and voxelize conformations for each molecule
    tp = np.dtype(
        [
            ("molecule", np.float16, (4, 40, 40, 40, 7)),
            ("idx", "S10"),
            ("target", np.float16),
            ("smile", np.int),
        ]
    )
    X = np.zeros(1, dtype=tp)

    canon_smi = Chem.MolToSmiles(m)
    voxelized_mols = [
        voxelize_molecule(m, box_size=[20, 20, 20], rotation=rotation) for j in range(4)
    ]
    vox_conformations = [conf[0] for conf in voxelized_mols]

    for conf in vox_conformations:
        if conf is None:
            print("No enough conformations for: ", canon_smi)
            continue

    molecules = [conf[2] for conf in voxelized_mols]

    vox_conformations = np.stack(
        vox_conformations, axis=0
    )  # (num_of_conf, h, w, d, channels)

    if save_smiles:
        X[0] = (vox_conformations, idx, l, canon_smi)
    else:
        X[0] = (vox_conformations, idx, l, 0)

    if save_mols:
        for count, m in enumerate(molecules):
            m.write(mol_dir.joinpath("{}_{}.mol2".format(idx, count)))

    np.save(tensor_dir.joinpath("{}.npy".format(idx)), X)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="SMRT_filtered.csv", type=str)
    parser.add_argument("--target_column", default="RETENTION_TIME", type=str)
    parser.add_argument("--output", default="metlin", type=str)
    parser.add_argument("--rotation", default=1, type=int)
    args = parser.parse_args()

    # load dataset
    script_path = Path(__file__).resolve().parent
    data_path = script_path.joinpath("data/")

    mol_dir = data_path.joinpath("precomputed_molecules/{}".format(args.output))
    mol_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir = data_path.joinpath("precomputed_tensors/{}".format(args.output))
    tensor_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path.joinpath(args.dataset), index_col=0)
    df["Molecule"] = df["SMILES"].apply(Chem.MolFromSmiles)

    argument_list = []
    for idx in tqdm(df.index.values):
        m = df.loc[idx, "Molecule"]
        l = df.loc[idx, args.target_column]
        argument_list.append((m, l, idx, args.rotation, mol_dir, tensor_dir))

    with multiprocessing.Pool(os.cpu_count()) as p:
        p.starmap(voxelize_and_save, argument_list)


if __name__ == "__main__":
    main()
