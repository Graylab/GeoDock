import os
import torch
import pandas as pd
from torch.utils import data
from esm.inverse_folding.util import load_coords


class AFDataset(data.Dataset):
    def __init__(
        self,
        model_dir: str,
        native_dir: str,
        chain_file: str,
        prefix: str,
        sep: bool = False,
    ):
        self.model_dir = model_dir
        self.native_dir = native_dir
        self.chain_file = pd.read_csv(chain_file)
        self.prefix = prefix
        self.sep = sep
        self.file_list = [filename for filename in os.listdir(model_dir)] 

    def __getitem__(self, idx: int):
        model = self.file_list[idx] 
        pdb_id = model.split('_unrelaxed', 1)[0]
        rank = model.split('rank_', 1)[1][2]
        model = os.path.join(self.model_dir, model)

        if self.sep:
            native_rec = os.path.join(self.native_dir, pdb_id+".dill_r_b_COMPLEX.pdb")
            native_lig = os.path.join(self.native_dir, pdb_id+".dill_l_b_COMPLEX.pdb")
        else:
            native = os.path.join(self.native_dir, f"{pdb_id}{self.prefix}.pdb")
            chains = self.chain_file.loc[self.chain_file['id'] == pdb_id, 'chain'].values[0] 
            chain1, chain2 = chains.split("_")

        if self.sep:
            model_coords1, model_seq1 = load_coords(model, chain='A')
            model_coords2, model_seq2 = load_coords(model, chain='B')

        else:
            model_coords1, model_seq1 = load_coords(model, chain=list(chain1))
            model_coords2, model_seq2 = load_coords(model, chain=list(chain2))

        model_coords1 = torch.nan_to_num(torch.from_numpy(model_coords1))
        model_coords2 = torch.nan_to_num(torch.from_numpy(model_coords2))

        if self.sep:
            native_coords1, native_seq1 = load_coords(native_rec, chain=None)
            native_coords2, native_seq2 = load_coords(native_lig, chain=None)
        else:
            native_coords1, native_seq1 = load_coords(native, chain=list(chain1))
            native_coords2, native_seq2 = load_coords(native, chain=list(chain2))

        native_coords1 = torch.nan_to_num(torch.from_numpy(native_coords1))
        native_coords2 = torch.nan_to_num(torch.from_numpy(native_coords2))

        # Output
        output = {
            'id': pdb_id,
            'rank': rank,
            'model_coords1': model_coords1, 
            'model_coords2': model_coords2, 
            'native_coords1': native_coords1,
            'native_coords2': native_coords2,
        }
        
        return {key: value for key, value in output.items()}

    def __len__(self):
        return len(self.file_list)
    

if __name__ == '__main__':
    dataset = AFDataset(
        model_dir='/home/lchu11/scr4_jgray21/lchu11/my_repos/ColabFold/db5_bound_single/structures',
        native_dir='/home/lchu11/scr4_jgray21/lchu11/data/db5/structures',
        chain_file='/home/lchu11/scr4_jgray21/lchu11/other_repos/AlphaRED/partners.csv',
    )

    print(dataset[0])