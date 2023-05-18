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
    parser.add_argument('--ckpt_file', type=str, default='best.ckpt')
    parser.add_argument('--dataset', type=str, default='dips_test')
    parser.add_argument('--count', type=int, default=0)
    parser.add_argument('--test_all', action='store_true', help='use the full testset')
    return parser.parse_args()

def _cli():
    #Get arguments
    args = _get_args()
    ckpt_file = args.ckpt_file
    dataset = args.dataset
    count = args.count
    test_all = args.test_all

    #get dataset
    testset = GeoDockDataset(
        dataset=dataset,
        is_training=False,
        count=count,
        use_Cb=True,
    )

    if not test_all:
        subset_indices = [0]
        subset = data.Subset(testset, subset_indices)
        test_dataloader = data.DataLoader(subset, batch_size=1, num_workers=6)
    else:
        test_dataloader = data.DataLoader(testset, batch_size=1, num_workers=6)


    #load dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = GeoDock.load_from_checkpoint(ckpt_file)
    model.eval()
    model.to(device)
    metrics_list = []

    for batch in tqdm(test_dataloader):
        _id = batch['id'][0]
        seq1 = batch['seq1'][0]
        seq2 = batch['seq2'][0]
        rep1 = batch['rep1'].to(device)
        rep2 = batch['rep2'].to(device)
        rel_pos = batch['rel_pos'].to(device)
        input_pairs = batch['input_pairs'].to(device)
        label_coords = batch['label_coords'].to(device)

        input = Input(
            rep1 = rep1,
            rep2 = rep2,
            pairs = input_pairs,
            rel_pos = rel_pos,
        )

        if len(seq1) + len(seq2) > 500:
            print(_id)
            continue

        # Get predicted coordinates 
        lddt_logit, dist_logit, pred_coords, pred_rotat, pred_trans = model(input)
        plddt = compute_plddt(lddt_logit).squeeze()

        pred = pred_coords.split([len(seq1), len(seq2)], dim=1)
        label = label_coords.split([len(seq1), len(seq2)], dim=1)

        metrics = {'id': _id}
        metrics.update(compute_metrics(pred, label))
        metrics.update({'plddt': plddt.mean().item()})
        metrics_list.append(metrics)

        #get N, CA, C coords
        coords = pred_coords.squeeze()
        coords1, coords2 = coords.split([len(seq1), len(seq2)], dim=0)
        full_coords = torch.cat([get_full_coords(coords1), get_full_coords(coords2)], dim=0)

        #get total len
        total_len = full_coords.size(0)

        #check seq len
        assert len(seq1) + len(seq2) == total_len

        # output dir
        out_dir = 'predictions/model3/' + dataset + '_' + str(count)
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        #get pdb
        out_pdb =  os.path.join(out_dir, _id + '_p.pdb')

        if os.path.exists(out_pdb):
            os.remove(out_pdb)
            print(f"File '{out_pdb}' deleted successfully.")
        else:
            print(f"File '{out_pdb}' does not exist.") 
            
        save_PDB(out_pdb=out_pdb, coords=full_coords, b_factors=plddt, seq=seq1+seq2, delim=len(seq1)-1)
    
    out_csv_dir = dataset + '_' + str(count) + '.csv'

    with open(out_csv_dir, 'w', newline='') as file:
        header = list(metrics_list[0].keys())
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        for row in metrics_list:
            row["c_rmsd"] = "{:.2f}".format(row["c_rmsd"])
            row["i_rmsd"] = "{:.2f}".format(row["i_rmsd"])
            row["l_rmsd"] = "{:.2f}".format(row["l_rmsd"])
            row["fnat"] = "{:.2f}".format(row["fnat"])
            row["DockQ"] = "{:.2f}".format(row["DockQ"])
            row["plddt"] = "{:.2f}".format(row["plddt"])
            writer.writerow(row)

        print(f"Results saved to {out_csv_dir}")


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


