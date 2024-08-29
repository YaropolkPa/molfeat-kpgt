import numpy as np
import torch
from torch.utils.data import DataLoader

from molfeat_kpgt.src.model.light import LiGhT
from molfeat_kpgt.src.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES
from molfeat_kpgt.src.data.collator import Collator_tune
from molfeat_kpgt.src.model_config import config_dict

from molfeat.calc.base import SerializableCalculator

class KPGTDescriptors(SerializableCalculator):
    def __init__(self,
                 features: bool = True):

        self.features = features
        self.__model_initialization()

        self.config = config_dict['base']    # TODO : make the ablity to choose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, mol):    # TODO: MUST BE FOR A SINGLE MOLECULE


        collator = Collator_tune(self.config['path_length'])
        loader = DataLoader(mol_dataset, batch_size=32, shuffle=False, num_workers=4, drop_last=False,
                            collate_fn=collator)

        fps_list = []
        for batch_idx, batched_data in enumerate(loader):
            (_, g, ecfp, md, labels) = batched_data
            ecfp = ecfp.to(self.device)
            md = md.to(self.device)
            g = g.to(self.device)
            fps = self.model.generate_fps(g, ecfp, md)
            fps_list.extend(fps.detach().cpu().numpy().tolist())

        np.savez_compressed(f"{args.data_path}/{args.dataset}/kpgt_{args.config}.npz", fps=np.array(fps_list))
        print(f"The extracted features were saving at {args.data_path}/{args.dataset}/kpgt_{args.config}.npz")
        pass


    def __model_initialization(self):

        vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)

        self.model = LiGhT(
            d_node_feats=self.config['d_node_feats'],
            d_edge_feats=self.config['d_edge_feats'],
            d_g_feats=self.config['d_g_feats'],
            d_hpath_ratio=self.config['d_hpath_ratio'],
            n_mol_layers=self.config['n_mol_layers'],
            path_length=self.config['path_length'],
            n_heads=self.config['n_heads'],
            n_ffn_dense_layers=self.config['n_ffn_dense_layers'],
            input_drop=0,
            attn_drop=0,
            feat_drop=0,
            n_node_types=vocab.vocab_size
        ).to(self.device)

        self.model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(args.model_path).items()})


    def __preprocess(self):
        pass