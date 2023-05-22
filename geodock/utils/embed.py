import esm
import torch
import torch.nn.functional as F
from esm.inverse_folding.util import load_coords
from geodock.utils.coords6d import get_coords6d
from geodock.model.interface import GeoDockInput

def embed(
    seq1,
    seq2,
    coords1,
    coords2,
    model,
    batch_converter,
    device,
):
    assert len(seq1) == coords1.size(0) and len(seq2) == coords2.size(0)

    # Get esm embeddings
    protein1_embeddings = get_esm_rep(seq1, model, batch_converter, device)
    protein2_embeddings = get_esm_rep(seq2, model, batch_converter, device)

    # Get pair embeddings
    coords = torch.cat([coords1, coords2], dim=0)
    input_pairs = get_pair_mats(coords, len(seq1))
    input_contact = torch.zeros(*input_pairs.shape[:-1])[..., None] 
    pair_embeddings = torch.cat([input_pairs, input_contact], dim=-1).to(device)
    
    # Get positional embeddings
    positional_embeddings = get_pair_relpos(len(seq1), len(seq2)).to(device)

    embeddings = GeoDockInput(
        protein1_embeddings=protein1_embeddings.unsqueeze(0),
        protein2_embeddings=protein2_embeddings.unsqueeze(0),
        pair_embeddings=pair_embeddings.unsqueeze(0),
        positional_embeddings=positional_embeddings.unsqueeze(0),
    )

    return embeddings
    
def get_esm_rep(seq_prim, model, batch_converter, device):
    # Use ESM-1b format.
    # The length of tokens is:
    # L (sequence length) + 2 (start and end tokens)
    seq = [
        ("seq", seq_prim)
    ]
    out = batch_converter(seq)
    with torch.no_grad():
        results = model(out[-1].to(device), repr_layers = [33])
        rep = results["representations"][33]
    
    return rep[0, 1:-1, :]

def get_pair_mats(coords, n, use_Cb=True):
    dist, omega, theta, phi = get_coords6d(coords, use_Cb=use_Cb)

    mask = dist < 22.0
    
    num_bins = 16
    dist_bin = get_bins(dist, 2.0, 22.0, num_bins)
    omega_bin = get_bins(omega, -180.0, 180.0, num_bins)
    theta_bin = get_bins(theta, -180.0, 180.0, num_bins)
    phi_bin = get_bins(phi, -180.0, 180.0, num_bins)

    def mask_mat(mat):
        mat[~mask] = num_bins - 1
        mat.fill_diagonal_(num_bins - 1)
        mat[:n, n:] = num_bins - 1
        mat[n:, :n] = num_bins - 1
        return mat

    dist_bin[:n, n:] = num_bins - 1
    dist_bin[n:, :n] = num_bins - 1
    omega_bin = mask_mat(omega_bin)
    theta_bin = mask_mat(theta_bin)
    phi_bin = mask_mat(phi_bin)

    # to onehot
    dist = F.one_hot(dist_bin, num_classes=num_bins).float() 
    omega = F.one_hot(omega_bin, num_classes=num_bins).float() 
    theta = F.one_hot(theta_bin, num_classes=num_bins).float() 
    phi = F.one_hot(phi_bin, num_classes=num_bins).float() 
    
    # test
    return torch.cat([dist, omega, theta, phi], dim=-1)

def get_pair_relpos(rec_len, lig_len):
    rmax = 32
    rec = torch.arange(0, rec_len)
    lig = torch.arange(0, lig_len) 
    total = torch.cat([rec, lig], dim=0)
    pairs = total[None, :] - total[:, None]
    pairs = torch.clamp(pairs, min=-rmax, max=rmax)
    pairs = pairs + rmax 
    pairs[:rec_len, rec_len:] = 2*rmax + 1
    pairs[rec_len:, :rec_len] = 2*rmax + 1 
    relpos = F.one_hot(pairs, num_classes=2*rmax+2).float()
    total_len = rec_len + lig_len
    chain_row = torch.cat([torch.zeros(rec_len, total_len), 
                            torch.ones(lig_len, total_len)], dim=0)  
    chain_col = torch.cat([torch.zeros(total_len, rec_len), 
                            torch.ones(total_len, lig_len)], dim=1)
    chains = F.one_hot((chain_row - chain_col + 1).long(), num_classes=3).float()

    pair_pos = torch.cat([relpos, chains], dim=-1)
    return pair_pos

def get_bins(x, min_bin, max_bin, num_bins):
    # Coords are [... L x 3 x 3], where it's [N, CA, C] x 3 coordinates.
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        num_bins - 1,
        device=x.device,
    )
    bins = torch.sum(x.unsqueeze(-1) > boundaries, dim=-1)  # [..., L, L]
    return bins