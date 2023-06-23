import os
import esm
import time
import math
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils import data
from einops import rearrange, repeat
from esm.inverse_folding.util import load_coords
from src.utils.pdb import save_PDB, place_fourth_atom 
from torch_geometric.data import HeteroData


class GeoDockDataset(data.Dataset):
    def __init__(
        self, 
        save_dir: str,
        dataset: str = 'dips_test',
        out_pdb: bool = False,
        device: str = 'cpu',
    ):
        if dataset == 'dips_test':
            self.data_dir = "/home/lchu11/scr4_jgray21/lchu11/Docking-dev/data/equidock"
            self.file_list = [i[:-21] for i in os.listdir(self.data_dir) if i[-3:] == 'pdb'] 
            self.file_list = list(dict.fromkeys(self.file_list)) # remove duplicates

        elif dataset == 'db5_bound':
            self.data_dir = "/home/lchu11/scr4_jgray21/lchu11/Docking-dev/data/db5.5/structures"
            self.file_list = [i[0:4] for i in os.listdir(self.data_dir) if i[-3:] == 'pdb'] 
            self.file_list = list(dict.fromkeys(self.file_list)) # remove duplicates

        elif dataset == 'db5_unbound':
            self.data_dir = "/home/lchu11/scr4_jgray21/lchu11/Docking-dev/data/db5.5/structures"
            self.file_list = [i[0:4] for i in os.listdir(self.data_dir) if i[-3:] == 'pdb'] 
            self.file_list = list(dict.fromkeys(self.file_list)) # remove duplicates
        
        elif dataset == 'db5_unbound_flexible':
            self.data_dir = "/home/lchu11/scr4_jgray21/lchu11/ReplicaDock2/AF_RepD2_set/flexible_targets/unbound"
            self.partner_dir = "/home/lchu11/scr4_jgray21/lchu11/ReplicaDock2/AF_RepD2_set/flexible_targets/partners"
            self.file_list = [i[:4] for i in os.listdir(self.data_dir) if i[-3:] == 'pdb']

        elif dataset == 'db5_bound_flexible':
            self.data_dir = "/home/lchu11/scr4_jgray21/lchu11/ReplicaDock2/AF_RepD2_set/flexible_targets/bound"
            self.partner_dir = "/home/lchu11/scr4_jgray21/lchu11/ReplicaDock2/AF_RepD2_set/flexible_targets/partners"
            self.file_list = [i[:4] for i in os.listdir(self.data_dir) if i[-3:] == 'pdb']

        elif dataset == 'abag_test':
            self.data_dir = "/home/lchu11/scr4_jgray21/lchu11/Docking-dev/data/exp_aa/cleaned/"
            pdb_list = pd.read_csv("/home/lchu11/scr4_jgray21/lchu11/Docking-dev/data/exp_aa/pdb_list.csv")
            self.file_list = pdb_list['pdb_id'].to_list()
            self.partner1 = pdb_list['partner1'].to_list()
            self.partner2 = pdb_list['partner2'].to_list()
        
        self.dataset = dataset
        self.save_dir = save_dir 
        self.out_pdb = out_pdb
        self.device = device

        # Load esm
        esm_model, alphabet = esm.pretrained.load_model_and_alphabet('/home/lchu11/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt')
        self.batch_converter = alphabet.get_batch_converter()
        self.esm_model = esm_model.to(device).eval()

    def __getitem__(self, idx: int):
        if self.dataset == 'dips_test':
            _id = self.file_list[idx] 
            pdb_file_1 = os.path.join(self.data_dir, _id+".dill_r_b_COMPLEX.pdb")
            pdb_file_2 = os.path.join(self.data_dir, _id+".dill_l_b_COMPLEX.pdb")
            coords1, seq1 = load_coords(pdb_file_1, chain=None)
            coords2, seq2 = load_coords(pdb_file_2, chain=None)
            coords1 = torch.nan_to_num(torch.from_numpy(coords1))
            coords2 = torch.nan_to_num(torch.from_numpy(coords2))
        
        elif self.dataset == 'db5_bound':
            # Get info from file_list
            _id = self.file_list[idx] 
            pdb_file_1 = os.path.join(self.data_dir, _id+"_r_b.pdb")
            pdb_file_2 = os.path.join(self.data_dir, _id+"_l_b.pdb")
            coords1, seq1 = load_coords(pdb_file_1, chain=None)
            coords2, seq2 = load_coords(pdb_file_2, chain=None)
            coords1 = torch.nan_to_num(torch.from_numpy(coords1))
            coords2 = torch.nan_to_num(torch.from_numpy(coords2))

        elif self.dataset == 'db5_unbound':
            # Get info from file_list
            _id = self.file_list[idx] 
            pdb_file_1 = os.path.join(self.data_dir, _id+"_r_u.pdb")
            pdb_file_2 = os.path.join(self.data_dir, _id+"_l_u.pdb")
            coords1, seq1 = load_coords(pdb_file_1, chain=None)
            coords2, seq2 = load_coords(pdb_file_2, chain=None)
            coords1 = torch.nan_to_num(torch.from_numpy(coords1))
            coords2 = torch.nan_to_num(torch.from_numpy(coords2))

        elif self.dataset == 'db5_unbound_flexible':
            _id = self.file_list[idx] 
            print(_id)
            model_file = os.path.join(self.data_dir, _id+"_unbound.pdb")
            partner_file = os.path.join(self.partner_dir, _id+"_partners") 

            with open(partner_file, 'r') as f:
                partner = f.readline().strip().split()[1]
                model_partner1 = partner.split('_')[0]
                model_partner2 = partner.split('_')[1]
                
            coords1, seq1 = load_coords(model_file, chain=[*model_partner1])
            coords2, seq2 = load_coords(model_file, chain=[*model_partner2])

            coords1 = torch.nan_to_num(torch.from_numpy(coords1))
            coords2 = torch.nan_to_num(torch.from_numpy(coords2))

        elif self.dataset == 'db5_bound_flexible':
            _id = self.file_list[idx] 
            print(_id)
            model_file = os.path.join(self.data_dir, _id+"_bound.pdb")
            partner_file = os.path.join(self.partner_dir, _id+"_partners") 

            with open(partner_file, 'r') as f:
                partner = f.readline().strip().split()[1]
                model_partner1 = partner.split('_')[0]
                model_partner2 = partner.split('_')[1]
                
            coords1, seq1 = load_coords(model_file, chain=[*model_partner1])
            coords2, seq2 = load_coords(model_file, chain=[*model_partner2])

            coords1 = torch.nan_to_num(torch.from_numpy(coords1))
            coords2 = torch.nan_to_num(torch.from_numpy(coords2))

        elif self.dataset == 'abag_test':
            # Get info from file_list
            _id = self.file_list[idx] 
            partner1 = list(self.partner1[idx])
            partner2 = list(self.partner2[idx])
            pdb_file = os.path.join(self.data_dir, _id+".pdb")
            coords1, seq1 = load_coords(pdb_file, partner1)
            coords2, seq2 = load_coords(pdb_file, partner2)
            coords1 = torch.nan_to_num(torch.from_numpy(coords1))
            coords2 = torch.nan_to_num(torch.from_numpy(coords2))

        # ESM embedding
        esm_rep1 = self.get_esm_rep(seq1)
        esm_rep2 = self.get_esm_rep(seq2)

        if self.out_pdb:
            coords1 = self.get_full_coords(coords1)
            coords2 = self.get_full_coords(coords2)
            test_coords = torch.cat([coords1, coords2], dim=0)
            save_PDB(out_pdb='test.pdb', coords=test_coords, seq=seq1+seq2, delim=len(seq1)-1)

        # save data to a hetero graph 
        data = HeteroData()

        data['receptor'].x = esm_rep1
        data['receptor'].pos = coords1
        data['receptor'].seq = seq1
        data['ligand'].x = esm_rep2
        data['ligand'].pos = coords2
        data['ligand'].seq = seq2
        data.name = _id
        torch.save(data, os.path.join(self.save_dir, _id+'.pt'))

        return coords1

    def __len__(self):
        return len(self.file_list)

    def get_esm_rep(self, seq_prim):
        # Use ESM-1b format.
        # The length of tokens is:
        # L (sequence length) + 2 (start and end tokens)
        seq = [
            ("seq", seq_prim)
        ]
        out = self.batch_converter(seq)
        with torch.no_grad():
            results = self.esm_model(out[-1].to(self.device), repr_layers = [33])
            rep = results["representations"][33].cpu()
        
        return rep[0, 1:-1, :]

    def convert_to_torch_tensor(self, atom_coords):
        # Convert atom_coords to torch tensor.
        n_coords = torch.Tensor(atom_coords['N'])
        ca_coords = torch.Tensor(atom_coords['CA'])
        c_coords = torch.Tensor(atom_coords['C'])
        coords = torch.stack([n_coords, ca_coords, c_coords], dim=1)
        return coords

    def get_full_coords(self, coords):
        #get full coords
        N, CA, C = [x.squeeze(-2) for x in coords.chunk(3, dim=-2)]
        # Infer CB coordinates.
        b = CA - N
        c = C - CA
        a = b.cross(c, dim=-1)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
        
        O = place_fourth_atom(torch.roll(N, -1, 0),
                                        CA, C,
                                        torch.tensor(1.231),
                                        torch.tensor(2.108),
                                        torch.tensor(-3.142))
        full_coords = torch.stack(
            [N, CA, C, O, CB], dim=1)
        
        return full_coords


if __name__ == '__main__':
    name = 'db5_unbound_flexible'
    save_dir = '/home/lchu11/scr4_jgray21/lchu11/Docking-dev/data/pts/'+name 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    else:
        print(f"Directory {save_dir} already exists")

    dataset = GeoDockDataset(
        save_dir=save_dir,
        dataset=name,
        out_pdb=False,
    )

    dataloader = data.DataLoader(dataset, batch_size=1)

    for batch in tqdm(dataloader):
        pass
