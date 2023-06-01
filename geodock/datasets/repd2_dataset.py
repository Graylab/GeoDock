import os
import torch
from torch.utils import data
from esm.inverse_folding.util import load_coords


class RepD2Dataset(data.Dataset):
    def __init__(
        self, 
        model_dir: str,
        native_dir: str,
        model_partner_dir: str,
        native_partner_dir: str,
    ):
        self.model_dir = model_dir
        self.native_dir = native_dir
        self.model_partner_dir = model_partner_dir
        self.native_partner_dir = native_partner_dir
        self.file_list = [i[:6] for i in os.listdir(self.model_dir) if i[-3:] == 'pdb']
        self.file_list = sorted(self.file_list)

    def __getitem__(self, idx: int):

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

        model_coords1, model_seq1 = load_coords(model_file, chain=[*model_partner1])
        model_coords2, model_seq2 = load_coords(model_file, chain=[*model_partner2])
        native_coords1, native_seq1 = load_coords(native_file, chain=[*native_partner1])
        native_coords2, native_seq2 = load_coords(native_file, chain=[*native_partner2])

        model_coords1 = torch.nan_to_num(torch.from_numpy(model_coords1))
        model_coords2 = torch.nan_to_num(torch.from_numpy(model_coords2))
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
    model_dir = '/home/lchu11/scr4_jgray21/lchu11/RepD2/top5'
    native_dir = '/home/lchu11/scr4_jgray21/lchu11/RepD2/native'
    model_partner_dir = '/home/lchu11/scr4_jgray21/lchu11/RepD2/partners/model'
    native_partner_dir = '/home/lchu11/scr4_jgray21/lchu11/RepD2/partners/native'

    dataset = RepD2Dataset(
        model_dir=model_dir,
        native_dir=native_dir,
        model_partner_dir=model_partner_dir,
        native_partner_dir=native_partner_dir,
    )

    dataset[0]