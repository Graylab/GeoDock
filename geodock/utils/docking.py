import os
import torch
import torch.nn.functional as F
import numpy as np
from time import time
from geodock.utils.pdb import save_PDB_string, place_fourth_atom 


def dock(
    out_name,
    seq1, 
    seq2,
    model_in,
    model,
    do_refine=True,
    use_openmm=True,
):
    #-----Docking Start-----#
    start_time = time()

    # get prediction
    model_out = model(model_in)

    # coords and plddt
    coords = model_out.coords.squeeze()
    plddt = compute_plddt(model_out.lddt_logits).squeeze()
    coords1, coords2 = coords.split([len(seq1), len(seq2)], dim=0)
    full_coords = torch.cat([get_full_coords(coords1), get_full_coords(coords2)], dim=0)

    # seq
    seq_dict = {'A': seq1, 'B': seq2}
    chains = list(seq_dict.keys())
    delims = np.cumsum([len(s) for s in seq_dict.values()]).tolist()

    # get total len
    total_len = full_coords.size(0)

    # check seq len
    assert len(seq1) + len(seq2) == total_len

    # output dir
    out_dir = './'        
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # get pdb
    out_pdb =  os.path.join(out_dir, f"{out_name}.pdb")

    if os.path.exists(out_pdb):
        os.remove(out_pdb)
        print(f"File '{out_pdb}' deleted successfully.")
    else:
        print(f"File '{out_pdb}' does not exist.") 
        
    pdb_string = save_PDB_string(
        out_pdb=out_pdb, 
        coords=full_coords, 
        error=plddt, 
        seq=seq1+seq2,
        chains=chains,
        delims=delims
    )
    print(f"Completed docking in {time() - start_time:.2f} seconds.")
    #-----Docking end-----#

    #-----Refine start-----#
    if do_refine:
        start_time = time()
        if use_openmm:
            try:
                from geodock.refine.openmm_ref import refine
                refine_input = [out_pdb]
            except:
                exit("OpenMM not installed. Please install OpenMM to use refinement.")
        else:
            try:
                from geodock.refine.pyrosetta_ref import refine
                refine_input = [out_pdb, pdb_string]
            except:
                exit("PyRosetta not installed. Please install PyRosetta to use refinement.")

        refine(*refine_input)
        print(f"Completed refining in {time() - start_time:.2f} seconds.")
    #-----Refine end-----#

def get_full_coords(coords):
    # get full coords
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
