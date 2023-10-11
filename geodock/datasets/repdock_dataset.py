import os
import torch
from torch.utils import data
from esm.inverse_folding.util import load_coords


class RepD2Dataset(data.Dataset):
    def __init__(
        self,
        dataset: str,
    ):
        self.dataset = dataset
        if self.dataset == 'RepD2':
            self.model_dir = "/home/lchu11/scr4_jgray21/lchu11/ReplicaDock2/flexible_set/top5" 
            self.native_dir = "/home/lchu11/scr4_jgray21/lchu11/ReplicaDock2/flexible_set/native"
            self.model_partner_dir = "/home/lchu11/scr4_jgray21/lchu11/ReplicaDock2/flexible_set/partners/model"
            self.native_partner_dir = "/home/lchu11/scr4_jgray21/lchu11/ReplicaDock2/flexible_set/partners/native"
            self.file_list = [i[:6] for i in os.listdir(self.model_dir) if i[-3:] == 'pdb']
            self.file_list = sorted(self.file_list)
        elif self.dataset == 'RepD2_global':
            self.model_dir = "/home/lchu11/scr4_jgray21/lchu11/ReplicaDock2/global/top1" 
            self.native_dir = "/home/lchu11/scr4_jgray21/lchu11/ReplicaDock2/flexible_set/native"
            self.model_partner_dir = "/home/lchu11/scr4_jgray21/lchu11/ReplicaDock2/flexible_set/partners/model"
            self.native_partner_dir = "/home/lchu11/scr4_jgray21/lchu11/ReplicaDock2/flexible_set/partners/native"
            self.file_list = [i[:4] for i in os.listdir(self.model_dir) if i[-3:] == 'pdb']
            self.file_list = sorted(self.file_list)
        elif self.dataset == 'RepD2_bound':
            self.model_dir = "/home/lchu11/scr4_jgray21/lchu11/ReplicaDock2/equidock_set/top1" 
            self.native_dir = "/home/lchu11/scr4_jgray21/lchu11/Docking-dev/data/db5.5/For_Ameya"
            self.model_partner_dir = "/home/lchu11/scr4_jgray21/lchu11/ReplicaDock2/equidock_set/partners"
            self.file_list = [i[:4] for i in os.listdir(self.model_dir) if i[-3:] == 'pdb']

    def __getitem__(self, idx: int):
        if self.dataset == 'RepD2':
            _id = self.file_list[idx] 
            model_file = os.path.join(self.model_dir, _id+".pdb")
            native_file = os.path.join(self.native_dir, _id[:4]+"_native.pdb")
            model_partner_file = os.path.join(self.model_partner_dir, _id[:4]) 
            native_partner_file = os.path.join(self.native_partner_dir, _id[:4]) 

            with open(model_partner_file, 'r') as f:
                partner = f.readline().strip()
                model_partner1 = partner.split('_')[0]
                model_partner2 = partner.split('_')[1]

            with open(native_partner_file, 'r') as f:
                partner = f.readline().strip()
                native_partner1 = partner.split('_')[0]
                native_partner2 = partner.split('_')[1]

        elif self.dataset == 'RepD2_global':
            _id = self.file_list[idx] 
            model_file = os.path.join(self.model_dir, _id+"_top_model.pdb")
            native_file = os.path.join(self.native_dir, _id[:4]+"_native.pdb")
            model_partner_file = os.path.join(self.model_partner_dir, _id[:4]) 
            native_partner_file = os.path.join(self.native_partner_dir, _id[:4]) 

            with open(model_partner_file, 'r') as f:
                partner = f.readline().strip()
                model_partner1 = partner.split('_')[0]
                model_partner2 = partner.split('_')[1]

            with open(native_partner_file, 'r') as f:
                partner = f.readline().strip()
                native_partner1 = partner.split('_')[0]
                native_partner2 = partner.split('_')[1]
                
            model_coords1, model_seq1 = load_coords(model_file, chain=[*model_partner1])
            model_coords2, model_seq2 = load_coords(model_file, chain=[*model_partner2])
            native_coords1, native_seq1 = load_coords(native_file, chain=[*native_partner1])
            native_coords2, native_seq2 = load_coords(native_file, chain=[*native_partner2])

            model_coords1 = torch.nan_to_num(torch.from_numpy(model_coords1))
            model_coords2 = torch.nan_to_num(torch.from_numpy(model_coords2))
            native_coords1 = torch.nan_to_num(torch.from_numpy(native_coords1))
            native_coords2 = torch.nan_to_num(torch.from_numpy(native_coords2))

        elif self.dataset == 'RepD2_bound':
            _id = self.file_list[idx] 
            model_file = os.path.join(self.model_dir, _id+".pdb")
            native_file_1 = os.path.join(self.native_dir, _id[:4]+"_r_b.pdb")
            native_file_2 = os.path.join(self.native_dir, _id[:4]+"_l_b.pdb")
            model_partner_file = os.path.join(self.model_partner_dir, _id[:4]+"_partners") 

            with open(model_partner_file, 'r') as f:
                partner = f.readline().strip().split()[1]
                model_partner1 = partner.split('_')[0]
                model_partner2 = partner.split('_')[1]

            model_coords1, model_seq1 = load_coords(model_file, chain=[*model_partner1])
            model_coords2, model_seq2 = load_coords(model_file, chain=[*model_partner2])
            native_coords1, native_seq1 = load_coords(native_file_1, chain=[*model_partner1])
            native_coords2, native_seq2 = load_coords(native_file_2, chain=[*model_partner1])

            model_coords1 = torch.nan_to_num(torch.from_numpy(model_coords1))
            model_coords2 = torch.nan_to_num(torch.from_numpy(model_coords2))
            native_coords1 = torch.nan_to_num(torch.from_numpy(native_coords1))
            native_coords2 = torch.nan_to_num(torch.from_numpy(native_coords2))

        # Output
        output = {
            'id': _id[:4],
            'model_coords1': model_coords1, 
            'model_coords2': model_coords2, 
            'native_coords1': native_coords1,
            'native_coords2': native_coords2,
        }
        
        return {key: value for key, value in output.items()}

    def __len__(self):
        return len(self.file_list)
    

if __name__ == '__main__':
    dataset = RepD2Dataset(
        dataset='RepD2_global'
    )
    dataset[0]

