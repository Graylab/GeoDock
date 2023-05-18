import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from torch.utils import data
from dataclasses import dataclass
from einops import repeat, rearrange
from src.models.geodock.geodock import GeoDock
from src.utils.pdb import save_PDB, place_fourth_atom 
from src.models.geodock.datamodules.datasets.geodock_dataset import GeoDockDataset
from src.models.geodock.metrics.metrics import compute_metrics

@dataclass
class Input():
    rep1: torch.LongTensor
    rep2: torch.LongTensor
    pairs: torch.FloatTensor
    rel_pos: torch.FloatTensor


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dips_test')
    parser.add_argument('--count', type=int, default=0)
    parser.add_argument('--test_all', action='store_true', help='use the full testset')
    return parser.parse_args()

def _cli():
    #Get arguments
    args = _get_args()
    dataset = args.dataset
    count = args.count
    test_all = args.test_all

    #get dataset
    testset = GeoDockDataset(
        dataset=dataset,
        is_training=False,
        count=count,
    )

    if not test_all:
        subset_indices = [0]
        subset = data.Subset(testset, subset_indices)
        test_dataloader = data.DataLoader(subset, batch_size=1, num_workers=6)
    else:
        test_dataloader = data.DataLoader(testset, batch_size=1, num_workers=6)


    for batch in tqdm(test_dataloader):
        _id = batch['id'][0]
        seq1 = batch['seq1'][0]
        seq2 = batch['seq2'][0]
        label_coords = batch['label_coords']

        #get N, CA, C coords
        coords = label_coords.squeeze()
        coords1, coords2 = coords.split([len(seq1), len(seq2)], dim=0)
        full_coords = torch.cat([get_full_coords(coords1), get_full_coords(coords2)], dim=0)

        #get total len
        total_len = full_coords.size(0)

        #check seq len
        assert len(seq1) + len(seq2) == total_len

        # output dir
        out_dir = dataset
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        #get pdb
        out_pdb =  os.path.join(out_dir, _id + '_gt.pdb')

        if os.path.exists(out_pdb):
            os.remove(out_pdb)
            print(f"File '{out_pdb}' deleted successfully.")
        else:
            print(f"File '{out_pdb}' does not exist.") 
            
        save_PDB(out_pdb=out_pdb, coords=full_coords, seq=seq1+seq2, delim=len(seq1)-1)
    

def get_full_coords(coords):
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


def compute_plddt(logits: torch.Tensor) -> torch.Tensor:
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bounds = torch.arange(
        start=0.5 * bin_width, end=1.0, step=bin_width, device=logits.device
    )
    probs = F.softmax(logits, dim=-1)
    pred_lddt_ca = torch.sum(
        probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape),
        dim=-1,
    )
    return pred_lddt_ca * 100

if __name__ == '__main__':
    _cli()


