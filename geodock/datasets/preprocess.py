import esm
import torch
import os.path as path
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import HeteroData
from esm.inverse_folding.util import load_coords


def get_esm_rep(seq_prim, batch_converter, esm_model, device):
    # Use ESM-1b format.
    # The length of tokens is:
    # L (sequence length) + 2 (start and end tokens)
    seq = [
        ("seq", seq_prim)
    ]
    out = batch_converter(seq)
    with torch.no_grad():
        results = esm_model(out[-1].to(device), repr_layers = [33])
        rep = results["representations"][33].cpu()
    
    return rep[0, 1:-1, :]

data_dir = "/home/lchu11/scr4_jgray21/lchu11/data/ab_ag/native_pdbs"
data_list = "/home/lchu11/scr4_jgray21/lchu11/data/ab_ag/af2.3_benchmark/af2.3_preds/2021_tmplts_5recycles/test.csv"
save_dir = "/home/lchu11/scr4_jgray21/lchu11/data/pt/ab_ag"

df = pd.read_csv(data_list)

# Load esm
esm_model, alphabet = esm.pretrained.load_model_and_alphabet('/home/lchu11/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt')
batch_converter = alphabet.get_batch_converter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
esm_model = esm_model.to(device).eval()

# save 
f = open('id_list.txt', 'w')

for index, row in tqdm(df.iterrows()):
    pdb_file = path.join(data_dir, row['case']+'.nat.chn.pdb')
    _id = row['id']

    n = len(row['receptor'] + row['ligand'])
    if n == 2:
        chain1 = ['B']
        chain2 = ['A']
    elif n == 3:
        chain1 = ['B', 'C']
        chain2 = ['A']
    elif n == 4:
        chain1 = ['C', 'D']
        chain2 = ['A', 'B']

    # get info from pdb
    coords1, seq1 = load_coords(pdb_file, chain1)
    coords2, seq2 = load_coords(pdb_file, chain2)
    coords1 = torch.nan_to_num(torch.from_numpy(coords1))
    coords2 = torch.nan_to_num(torch.from_numpy(coords2))

    # ESM embedding
    esm_rep1 = get_esm_rep(seq1, batch_converter, esm_model, device)
    esm_rep2 = get_esm_rep(seq2, batch_converter, esm_model, device)

    # save data to a hetero graph 
    data = HeteroData()

    data.name = _id
    data['receptor'].x = esm_rep1
    data['receptor'].pos = coords1
    data['receptor'].seq = seq1
    data['ligand'].x = esm_rep2
    data['ligand'].pos = coords2
    data['ligand'].seq = seq2

    if (esm_rep1.size(0) == coords1.size(0) == len(seq1)) and (esm_rep2.size(0) == coords2.size(0) == len(seq2)):
        torch.save(data, path.join(save_dir, _id+'.pt'))
    else:
        f.write('%s\n' % (_id))

f.close()


