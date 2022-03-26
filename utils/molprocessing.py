import pandas as pd
from tqdm import tqdm
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, PandasTools
from rdkit.Chem.AllChem import UFFGetMoleculeForceField, Conformer
from rdkit.Chem.rdMolTransforms import TransformConformer, CanonicalizeMol, CanonicalizeConformer
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

from moleculekit.smallmol.smallmol import SmallMol
from moleculekit.tools.voxeldescriptors import getVoxelDescriptors, viewVoxelFeatures
from moleculekit.util import uniformRandomRotation

def conf_energies(mol):
    """
    Calculate conformer's energy
    """
    ff = UFFGetMoleculeForceField
    for conf in mol.GetConformers():
        id_ = conf.GetId()
        yield ff(mol, confId=id_).CalcEnergy(), id_

def generate_representation_v2(m, N=8, rotation=False):
    """
    Makes embeddings of Molecule.
    """
    try:
        mh = Chem.AddHs(m)
        AllChem.EmbedMultipleConfs(mh, numConfs=N, numThreads=0)
        # Get rid of redundant conformers.
        energy, conf_id = min(conf_energies(mh))
        #print(energy, conf_id)
        conf = mh.GetConformer(conf_id)
        conf = Conformer(conf)
        conf.SetId(0)
        mh.RemoveAllConformers()
        mh.AddConformer(conf)
        m = Chem.RemoveHs(mh)
        if rotation:
            R = uniformRandomRotation()
            A = np.zeros((4, 4))
            A[-1, -1] = 1.0
            A[:-1, :-1] = R
            TransformConformer(m.GetConformer(0), A)
        else:
            CanonicalizeConformer(m.GetConformer(0))

        mol = SmallMol(m)
        return mol, energy
    except:  # Rarely the conformer generation fails
        return None

def voxelize_molecule(molecule, box_size=None, center=None, voxel_size=0.5, rotation=False, method='C', show=False,):
    """Function which converts a small molecule into a tensor-representation with the following channels: hydrophobic, aromatic, hbond_acceptor, hbond_donor, positive_ionizable,
          negative_ionizable, metal, occupancies. 

    Args:
        molecule: rdkit mol object.
        
        box_size (list or int): size of 3D grid around the molecule.
        
        center (list): [x, y, z] of the box's center. If None the grid is constructed around the geometric center of
        the molecule. Use this only in combination with the `box_size` argument.
        
        voxel_size (float, optional): size of each voxel(in Angstrom). Default is 0.5 Angstrom.

        method (string): Can be 'C', 'CUDA' or 'NUMBA', if you have GPUs try 'CUDA' it can hugely increase performance especially for big molecules

        show(bool): If True, call VMD viewer with calculated grid(mesh representation) and molecule, works only if VMD is installed, else do nothing
    Returns:
        numpy array: An array with the following dimensions: (1, channels, box_size[1], box_size[2], box_size[3])
    """

    fail_count = 0
    while True:
        representation = generate_representation_v2(molecule, rotation=rotation)
        if representation is not None:
            break
        else:
            fail_count += 1 #conformer can no be generated
        
        if fail_count == 3:
            return None, None, None

    m = representation[0]
    energy = representation[1]
    
    if isinstance(box_size, int):
        box_size = [box_size, box_size, box_size]
    
    if center is None:
        center = m.getCenter()

    lig_vox, lig_centers, lig_N = getVoxelDescriptors(m, boxsize=box_size, center=center, voxelsize=voxel_size, method=method)
    
    #reshaping into multidimensional array
    lig_vox_t = lig_vox.reshape([lig_N[0], lig_N[1], lig_N[2], lig_vox.shape[1]]) # (height, width, depth, channels)
    lig_vox_t = lig_vox_t.astype(np.float32)
    lig_vox_t = np.concatenate([lig_vox_t[:,:, :, :-2], lig_vox_t[:, :, :, -1:]], axis=-1)

    return lig_vox_t, energy, m

def _generate_scaffold(smiles: str, include_chirality: bool = False):
    """
  Code is based on the DeepChem's implementation(https://github.com/deepchem/deepchem) 
  Compute the Bemis-Murcko scaffold for a SMILES string.
  Bemis-Murcko scaffolds are described in DOI: 10.1021/jm9602928.
  They are essentially that part of the molecule consisting of
  rings and the linker atoms between them.
  Paramters
  ---------
  smiles: str
    SMILES
  include_chirality: bool, default False
    Whether to include chirality in scaffolds or not.
  Returns
  -------
  str
    The MurckScaffold SMILES from the original SMILES
  References
  ----------
  .. [1] Bemis, Guy W., and Mark A. Murcko. "The properties of known drugs.
     1. Molecular frameworks." Journal of medicinal chemistry 39.15 (1996): 2887-2893.
  Note
  ----
    """

    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold

def generate_scaffold_split(smiles, frac_train):

    scaffolds = {}

    for i, smi in enumerate(smiles):

        scaffold = _generate_scaffold(smi)
        scaffolds.setdefault(scaffold, []).append(i)

    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]) , reverse=True)
    ]

    train_inds = []
    test_inds = []
    train_cutoff = frac_train * len(smiles)

    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            test_inds += scaffold_set
        else:
            train_inds += scaffold_set

    return train_inds, test_inds


#smiles = ["CC(C)Cl" , "CCC(C)CO" ,  "CCCCCCCO" , "CCCCCCCC(=O)OC" , "c3ccc2nc1ccccc1cc2c3" , "Nc2cccc3nc1ccccc1cc23" , "C1CCCCCC1" ]