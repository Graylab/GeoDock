import os
import torch
from torch.utils import data
from esm.inverse_folding.util import load_coords


class BoundDataset(data.Dataset):
    def __init__(
        self,
        dataset: str,
    ):
        self.dataset = dataset
        if self.dataset.split('_')[0] == 'dips':
            self.native_dir = "/home/lchu11/scr4_jgray21/lchu11/other_repos/equidock_public/test_sets_pdb/dips_test_random_transformed/complexes"
            self.file_list = [i[:-21] for i in os.listdir(self.native_dir) if i[-3:] == 'pdb'] 
            self.file_list = list(dict.fromkeys(self.file_list)) # remove duplicates
        elif self.dataset.split('_')[0] == 'db5':
            self.native_dir = "/home/lchu11/scr4_jgray21/lchu11/other_repos/equidock_public/test_sets_pdb/db5_test_random_transformed/complexes"
            self.file_list = [i[:4] for i in os.listdir(self.native_dir) if i[-3:] == 'pdb'] 
            self.file_list = list(dict.fromkeys(self.file_list)) # remove duplicates

        if self.dataset == 'dips_attract':
            self.model_dir = "/home/lchu11/scr4_jgray21/lchu11/other_repos/equidock_public/test_sets_pdb/dips_attract_results" 
        elif self.dataset == 'dips_cluspro':
            self.model_dir = "/home/lchu11/scr4_jgray21/lchu11/other_repos/equidock_public/test_sets_pdb/dips_cluspro_results" 
        elif self.dataset == 'dips_patchdock':
            self.model_dir = "/home/lchu11/scr4_jgray21/lchu11/other_repos/equidock_public/test_sets_pdb/dips_patchdock_results" 
        elif self.dataset == 'dips_equidock':
            self.model_dir = "/home/lchu11/scr4_jgray21/lchu11/other_repos/equidock_public/test_sets_pdb/dips_equidock_results" 
        elif self.dataset == 'dips_alphafold':
            self.model_dir = "/home/lchu11/scr4_jgray21/lchu11/my_repos/ColabFold/dips_test/structures"
        elif self.dataset == 'db5_attract':
            self.model_dir = "/home/lchu11/scr4_jgray21/lchu11/other_repos/equidock_public/test_sets_pdb/db5_attract_results" 
        elif self.dataset == 'db5_cluspro':
            self.model_dir = "/home/lchu11/scr4_jgray21/lchu11/other_repos/equidock_public/test_sets_pdb/db5_cluspro_results" 
        elif self.dataset == 'db5_patchdock':
            self.model_dir = "/home/lchu11/scr4_jgray21/lchu11/other_repos/equidock_public/test_sets_pdb/db5_patchdock_results" 
        elif self.dataset == 'db5_equidock':
            self.model_dir = "/home/lchu11/scr4_jgray21/lchu11/other_repos/equidock_public/test_sets_pdb/db5_equidock_results" 

    def __getitem__(self, idx: int):
        _id = self.file_list[idx] 

        dataset, method = self.dataset.split('_')

        if dataset == 'dips':
            native_rec = os.path.join(self.native_dir, _id+".dill_r_b_COMPLEX.pdb")
            native_lig = os.path.join(self.native_dir, _id+".dill_l_b_COMPLEX.pdb")
            if method == 'attract':
                model_rec = os.path.join(self.model_dir, _id+".dill_r_b_ATTRACT.pdb")
                model_lig = os.path.join(self.model_dir, _id+".dill_l_b_ATTRACT.pdb")
            elif method == 'cluspro':
                model_rec = os.path.join(self.native_dir, _id+".dill_r_b_COMPLEX.pdb")
                model_lig = os.path.join(self.model_dir, _id+".dill_l_b_CLUSPRO.pdb")
            elif method == 'patchdock':
                model_rec = os.path.join(self.native_dir, _id+".dill_r_b_COMPLEX.pdb")
                model_lig = os.path.join(self.model_dir, _id+".dill_l_b_PATCHDOCK.pdb")
            elif method == 'equidock':
                model_rec = os.path.join(self.native_dir, _id+".dill_r_b_COMPLEX.pdb")
                model_lig = os.path.join(self.model_dir, _id+".dill_l_b_EQUIDOCK.pdb")
            elif method == 'alphafold':
                model = os.path.join(self.model_dir, _id+"_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb")

        elif dataset == 'db5':
            native_rec = os.path.join(self.native_dir, _id+"_r_b_COMPLEX.pdb")
            native_lig = os.path.join(self.native_dir, _id+"_l_b_COMPLEX.pdb")
            if method == 'attract':
                model_rec = os.path.join(self.model_dir, _id+"_r_b_ATTRACT.pdb")
                model_lig = os.path.join(self.model_dir, _id+"_l_b_ATTRACT.pdb")
            elif method == 'cluspro':
                model_rec = os.path.join(self.native_dir, _id+"_r_b_COMPLEX.pdb")
                model_lig = os.path.join(self.model_dir, _id+"_l_b_CLUSPRO.pdb")
            elif method == 'patchdock':
                model_rec = os.path.join(self.native_dir, _id+"_r_b_COMPLEX.pdb")
                model_lig = os.path.join(self.model_dir, _id+"_l_b_PATCHDOCK.pdb")
            elif method == 'equidock':
                model_rec = os.path.join(self.native_dir, _id+"_r_b_COMPLEX.pdb")
                model_lig = os.path.join(self.model_dir, _id+"_l_b_EQUIDOCK.pdb")

        if method == 'alphafold':
            model_coords1, model_seq1 = load_coords(model, chain='A')
            model_coords2, model_seq2 = load_coords(model, chain='B')
        else:
            model_coords1, model_seq1 = load_coords(model_rec, chain=None)
            model_coords2, model_seq2 = load_coords(model_lig, chain=None)

        model_coords1 = torch.nan_to_num(torch.from_numpy(model_coords1))
        model_coords2 = torch.nan_to_num(torch.from_numpy(model_coords2))

        native_coords1, native_seq1 = load_coords(native_rec, chain=None)
        native_coords2, native_seq2 = load_coords(native_lig, chain=None)
        native_coords1 = torch.nan_to_num(torch.from_numpy(native_coords1))
        native_coords2 = torch.nan_to_num(torch.from_numpy(native_coords2))

        # Output
        output = {
            'id': _id,
            'model_coords1': model_coords1, 
            'model_coords2': model_coords2, 
            'native_coords1': native_coords1,
            'native_coords2': native_coords2,
        }
        
        return {key: value for key, value in output.items()}

    def __len__(self):
        return len(self.file_list)
    

if __name__ == '__main__':
    dataset = BoundDataset(
        dataset='dips_alphafold'
    )

    dataset[0]