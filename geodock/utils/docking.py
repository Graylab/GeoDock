import os
import torch
import torch.nn.functional as F
from geodock.utils.pdb import save_PDB, place_fourth_atom 


def dock(
    out_name,
    seq1, 
    seq2,
    model_in,
    model,
):

    model_out = model(model_in)

    coords = model_out.coords.squeeze()
    plddt = compute_plddt(model_out.lddt_logits).squeeze()
    coords1, coords2 = coords.split([len(seq1), len(seq2)], dim=0)
    full_coords = torch.cat([get_full_coords(coords1), get_full_coords(coords2)], dim=0)

    #get total len
    total_len = full_coords.size(0)

    #check seq len
    assert len(seq1) + len(seq2) == total_len

    # output dir
    out_dir = './'        
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    #get pdb
    out_pdb =  os.path.join(out_dir, f"{out_name}.pdb")

    if os.path.exists(out_pdb):
        os.remove(out_pdb)
        print(f"File '{out_pdb}' deleted successfully.")
    else:
        print(f"File '{out_pdb}' does not exist.") 
        
    save_PDB(out_pdb=out_pdb, coords=full_coords, b_factors=plddt, seq=seq1+seq2, delim=len(seq1)-1)


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
