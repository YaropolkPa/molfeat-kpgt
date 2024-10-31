import os
import random
import numpy as np
import torch
import dgl
from typing import Union
from typing import List
import datamol as dm

def set_random_seed(seed=22, n_threads=16):
    """Set random seed.

    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    dgl.random.seed(seed)
    dgl.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(n_threads)
    os.environ['PYTHONHASHSEED'] = str(seed)


from molfeat_kpgt.src.data.featurizer import smiles_to_graph_tune
from rdkit import Chem
from scipy import sparse as sp
from molfeat_kpgt.src.data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized


def smiles_to_input_parralel(smiles, n_jobs=4):
    from dgllife.utils.io import pmap
    from multiprocessing import Pool
    graphs = pmap(smiles_to_graph_tune,
                  smiless,
                  max_length=5,
                  n_virtual_nodes=2,
                  n_jobs=n_jobs)
    valid_ids = []
    valid_graphs = []
    for i, g in enumerate(graphs):
        if g is not None:
            valid_ids.append(i)
            valid_graphs.append(g)

    FP_list = []
    for smiles in smiless:
        mol = Chem.MolFromSmiles(smiles)
        FP_list.append(list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)))
    FP_arr = np.array(FP_list)
    FP_sp_mat = sp.csc_matrix(FP_arr)

    generator = RDKit2DNormalized()
    features_map = Pool(n_jobs).imap(generator.process, smiless)
    features_arr = np.array(list(features_map))

def smiles_to_input(smiles_list):
    # Constructing graphs
    graphs = map(lambda s: smiles_to_graph_tune(s, max_length=5), smiles_list)
    valid_ids = []
    valid_graphs = []
    for i, g in enumerate(graphs):
        if g is not None:
            valid_ids.append(i)
            valid_graphs.append(g)
    # TODO : LABELS ?!
    labels = [0 for _ in range(len(smiles_list))]
    labels = torch.tensor(labels, dtype=torch.float32)
    # _label_values = df[task_names].values
    # labels = F.zerocopy_from_numpy(
    # _label_values.astype(np.float32))[valid_ids]

    # Extracting fingerprints
    FP_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        FP_list.append(list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)))
    FP_arr = torch.tensor(FP_list, dtype=torch.float32)

    # Extracting descriptors
    generator = RDKit2DNormalized()
    descriptors_map = map(generator.process, smiles_list)
    descriptors_arr = torch.tensor(np.array(list(descriptors_map))[:,1:], dtype=torch.float32)

    samples = [(s, g, fp, d, l) for s, g, fp, d, l in zip(smiles_list, valid_graphs, FP_arr, descriptors_arr, labels)]
    return samples

def convert_smiles(
    inputs: List[Union[str, dm.Mol]], parallel_kwargs: dict, standardize: bool = False
):
    """Convert the list of input molecules into the proper format for embeddings
    Args:
        inputs: list of input molecules
        parallel_kwargs: kwargs for datamol parallelization
        standardize: whether to standardize the smiles
    """

    if isinstance(inputs, (str, dm.Mol)):
        inputs = [inputs]

    def _to_smiles(x):
        out = dm.to_smiles(x) if not isinstance(x, str) else x
        if standardize:
            with dm.without_rdkit_log():
                return dm.standardize_smiles(out)
        return out

    if len(inputs) > 1:
        smiles = dm.utils.parallelized(
            _to_smiles,
            inputs,
            **parallel_kwargs,
        )
    else:
        smiles = [_to_smiles(x) for x in inputs]

    return smiles

if __name__ == "__main__":
    import pandas as pd
    import time

    df = pd.read_csv(f"../../tests/datasets/freesolv/freesolv.csv")
    # cache_file_path = f"{dataset_path}/{dataset}/{dataset}_{path_length}.pkl"

    smiless = df.smiles.values.tolist()
    task_names = df.columns.drop(['smiles']).tolist()

    smiless = smiless[:32]

    start_time = time.time()
    (_, g, fps, d, _) = smiles_to_input(smiless)
    print("Nonparallel execution")
    print("--- %s seconds ---" % (time.time() - start_time))

    # start_time = time.time()
    # convert_smiles_parralel(smiless, n_jobs=4)
    # print("Parallel execution")
    # print("--- %s seconds ---" % (time.time() - start_time))


