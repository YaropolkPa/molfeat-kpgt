from molfeat.trans.pretrained.base import PretrainedMolTransformer

from molfeat_kpgt.src.model.light import LiGhTPredictor as LiGhT
from molfeat_kpgt.src.model_config import config_dict
from molfeat_kpgt.src.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES

import torch

class KPGTModel(PretrainedMolTransformer):
    def __init__(self,
                 model_path: str = 'models/pretrained/base/base.pth',
                 config: str = 'base',  # TODO : Proper config
                 batch_size: int = 32,
                 n_jobs: int = 4,
                 device: str = None
                 ):

        self.batch_size = batch_size
        self.n_jobs = n_jobs
        config = config_dict[config]

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
        self.model = LiGhT(
            d_node_feats=config['d_node_feats'],
            d_edge_feats=config['d_edge_feats'],
            d_g_feats=config['d_g_feats'],
            d_hpath_ratio=config['d_hpath_ratio'],
            n_mol_layers=config['n_mol_layers'],
            path_length=config['path_length'],
            n_heads=config['n_heads'],
            n_ffn_dense_layers=config['n_ffn_dense_layers'],
            input_drop=0,
            attn_drop=0,
            feat_drop=0,
            n_node_types=vocab.vocab_size
        ).to(device)
        self.model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path).items()})

        super().__init__()

    def _convert(self, inputs: list, **kwargs):
        return smiles_into_graphs(inputs, self.n_jobs)

    def _embed(self, smiles: str, **kwargs):
        pass