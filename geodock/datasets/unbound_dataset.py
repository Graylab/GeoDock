import os
import torch
from torch.utils import data
from esm.inverse_folding.util import load_coords


class UnboundDataset(data.Dataset):
    def __init__(
        self,
        dataset: str,
    ):
        self.dataset = dataset
        if self.dataset == 'db5_test_flexible_unbound':
            self.model_dir = "/home/lchu11/scr4_jgray21/lchu11/GeoDock/geodock/benchmark/db5_test_flexible_unbound"
            self.unbound_dir = "/home/lchu11/scr4_jgray21/lchu11/ReplicaDock2/AF_RepD2_set/flexible_targets/unbound" 
            self.bound_dir = "/home/lchu11/scr4_jgray21/lchu11/ReplicaDock2/AF_RepD2_set/flexible_targets/bound"
            self.partner_dir = "/home/lchu11/scr4_jgray21/lchu11/ReplicaDock2/AF_RepD2_set/flexible_targets/partners"
            self.file_list = [i[:4] for i in os.listdir(self.model_dir) if i[-3:] == 'pdb']

    def __getitem__(self, idx: int):
        if self.dataset == 'db5_test_flexible_unbound':
            _id = self.file_list[idx]
            model_file = os.path.join(self.model_dir, _id+"_p.pdb")
            unbound_file = os.path.join(self.unbound_dir, _id+"_unbound.pdb")
            bound_file = os.path.join(self.bound_dir, _id+"_bound.pdb")
            native_partner_file = os.path.join(self.partner_dir, _id+"_partners") 

            model_partner1 = 'A'
            model_partner2 = 'B'
            with open(native_partner_file, 'r') as f:
                partner = f.readline().strip().split()[1]
                native_partner1 = partner.split('_')[0]
                native_partner2 = partner.split('_')[1]


            model_coords1, model_seq1 = load_coords(model_file, chain=[*model_partner1])
            model_coords2, model_seq2 = load_coords(model_file, chain=[*model_partner2])
            unbound_coords1, unbound_seq1 = load_coords(unbound_file, chain=[*native_partner1])
            unbound_coords2, unbound_seq2 = load_coords(unbound_file, chain=[*native_partner2])
            bound_coords1, bound_seq1 = load_coords(bound_file, chain=[*native_partner1])
            bound_coords2, bound_seq2 = load_coords(bound_file, chain=[*native_partner2])

            model_coords1 = torch.nan_to_num(torch.from_numpy(model_coords1))
            model_coords2 = torch.nan_to_num(torch.from_numpy(model_coords2))
            unbound_coords1 = torch.nan_to_num(torch.from_numpy(unbound_coords1))
            unbound_coords2 = torch.nan_to_num(torch.from_numpy(unbound_coords2))
            bound_coords1 = torch.nan_to_num(torch.from_numpy(bound_coords1))
            bound_coords2 = torch.nan_to_num(torch.from_numpy(bound_coords2))

            # Output
            output = {
                'id': _id[:4],
                'model_coords1': model_coords1, 
                'model_coords2': model_coords2, 
                'unbound_coords1': unbound_coords1,
                'unbound_coords2': unbound_coords2,
                'bound_coords1': bound_coords1,
                'bound_coords2': bound_coords2,
            }
        
        return {key: value for key, value in output.items()}

    def __len__(self):
        return len(self.file_list)
    

if __name__ == '__main__':
    dataset = UnboundDataset(
        dataset='db5_test_flexible_unbound'
    )

    dataset[0]