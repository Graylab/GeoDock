###
#   Modified from https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/loss.py
###

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from geodock.model.interface import GeoDockOutput

def get_fape(
    coords: torch.Tensor, 
    rotat: torch.Tensor, 
    trans: torch.Tensor,
) -> torch.Tensor:
    """
    """
    size = coords.size(1)
    assert size == trans.size(1) and size == rotat.size(1)
    coords = repeat(coords, 'b c a i -> b r c a i', r=size)
    trans = repeat(trans, 'b r i -> b r c () i', c=size)
    rotat = repeat(rotat, 'b r a i -> b r c a i', c=size)
    fape = torch.einsum('b r c a j, b r c i j -> b r c a i', coords-trans, rotat.transpose(-1, -2))
    return fape

def fape_loss(
    pred: torch.Tensor, 
    label: torch.Tensor, 
    mask: torch.Tensor = None, 
    d_clamp: float = None,
) -> torch.Tensor:
    """
    """
    loss = torch.sqrt(torch.sum((pred - label)**2.0, dim=-1) + 1e-4)
    if d_clamp is not None:
        loss = torch.clamp(loss, max=d_clamp)

    if mask is not None:
        mask = repeat(mask, 'b r c a -> b r c (repeat a)', repeat=loss.size(-1))
        loss = torch.sum(loss * mask) / (torch.sum(mask) + 1e-6)
    else: 
        loss = torch.mean(loss)

    return loss

def inter_fape_loss(
    pred: torch.Tensor, 
    label: torch.Tensor, 
    sep: int, 
    d_clamp: float = None
) -> torch.Tensor:
    """
    """
    loss = torch.sqrt(torch.sum((pred - label)**2.0, dim=-1) + 1e-4)

    if d_clamp is not None:
        loss = torch.clamp(loss, max=d_clamp)

    mask = torch.zeros_like(loss, device=pred.device)

    mask[:, :sep, sep:, :] = 1.0 
    mask[:, sep:, :sep, :] = 1.0

    loss = torch.sum(loss * mask) / torch.sum(mask)

    return loss

def intra_fape_loss(
    pred: torch.Tensor, 
    label: torch.Tensor, 
    sep: int, 
    d_clamp: float = None
) -> torch.Tensor:
    """
    """
    loss = torch.sqrt(torch.sum((pred - label)**2.0, dim=-1) + 1e-4)

    if d_clamp is not None:
        loss = torch.clamp(loss, max=d_clamp)

    mask = torch.zeros_like(loss, device=pred.device)

    mask[:, :sep, :sep, :] = 1.0 
    mask[:, sep:, sep:, :] = 1.0

    loss = torch.sum(loss * mask) / torch.sum(mask)

    return loss

def between_residue_bond_loss(
    pred_coords: torch.Tensor, 
    eps: float = 1e-6, 
    tolerance_factor_soft: float = 12.0
) -> torch.Tensor:
    """
    """
    this_ca_pos = pred_coords[:, :-1, 1]
    this_c_pos = pred_coords[:, :-1, 2]
    next_n_pos = pred_coords[:, 1:, 0]
    next_ca_pos = pred_coords[:, 1:, 1]
    c_n_bond_length = torch.sqrt(
        eps + torch.sum((this_c_pos - next_n_pos) ** 2, dim=-1)
    )
    ca_c_bond_length = torch.sqrt(
        eps + torch.sum((this_ca_pos - this_c_pos) ** 2, dim=-1)
    )
    n_ca_bond_length = torch.sqrt(
        eps + torch.sum((next_n_pos - next_ca_pos) ** 2, dim=-1)
    )
    c_ca_unit_vec = (this_ca_pos - this_c_pos) / ca_c_bond_length[..., None]
    c_n_unit_vec = (next_n_pos - this_c_pos) / c_n_bond_length[..., None]
    n_ca_unit_vec = (next_ca_pos - next_n_pos) / n_ca_bond_length[..., None]

    gt_length = 1.329
    gt_stddev = 0.014
    c_n_bond_length_error = torch.sqrt(eps + (c_n_bond_length - gt_length) ** 2)
    c_n_loss_per_residue = torch.nn.functional.relu(
        c_n_bond_length_error - tolerance_factor_soft * gt_stddev
    )
    c_n_loss = torch.mean(c_n_loss_per_residue)

    ca_c_n_cos_angle = torch.sum(c_ca_unit_vec * c_n_unit_vec, dim=-1)
    gt_angle = -0.4473
    gt_stddev = 0.014
    ca_c_n_cos_angle_error = torch.sqrt(
        eps + (ca_c_n_cos_angle - gt_angle) ** 2
    )
    ca_c_n_loss_per_residue = torch.nn.functional.relu(
        ca_c_n_cos_angle_error - tolerance_factor_soft * gt_stddev
    )
    ca_c_n_loss = torch.mean(ca_c_n_loss_per_residue)
    
    c_n_ca_cos_angle = torch.sum((-c_n_unit_vec) * n_ca_unit_vec, dim=-1)
    gt_angle = -0.5203 
    gt_stddev = 0.03
    c_n_ca_cos_angle_error = torch.sqrt(
        eps + torch.square(c_n_ca_cos_angle - gt_angle)
    )
    c_n_ca_loss_per_residue = torch.nn.functional.relu(
        c_n_ca_cos_angle_error - tolerance_factor_soft * gt_stddev
    )
    c_n_ca_loss = torch.mean(c_n_ca_loss_per_residue)

    loss = c_n_loss + ca_c_n_loss + c_n_ca_loss
    return loss

def violation_loss(
    pred_coords: torch.Tensor, 
    sep: int,
) -> torch.Tensor:
    """
    """
    pred_1 = pred_coords[:, :sep]
    pred_2 = pred_coords[:, sep:]
    loss_1 = between_residue_bond_loss(pred_1) 
    loss_2 = between_residue_bond_loss(pred_2)
    loss = (loss_1 + loss_2) * 0.5
    return loss

def lddt_loss(
    logits: torch.Tensor,
    pred_coords: torch.Tensor,
    label_coords: torch.Tensor,
    cutoff: float = 15.0,
    no_bins: int = 50,
    eps: float = 1e-10,
    **kwargs,
) -> torch.Tensor:
    """
    """
    pred_coords = pred_coords[..., 1, :]
    label_coords = label_coords[..., 1, :]

    score = lddt(
        pred_coords, 
        label_coords, 
        cutoff=cutoff, 
        eps=eps
    )

    score = score.detach()

    bin_index = torch.floor(score * no_bins).long()
    bin_index = torch.clamp(bin_index, max=(no_bins - 1))
    lddt_ca_one_hot = F.one_hot(
        bin_index, num_classes=no_bins
    )

    errors = softmax_cross_entropy(logits, lddt_ca_one_hot)

    # Average over the batch dimension
    loss = torch.mean(errors)

    return loss

def lddt(
    pred_coords: torch.Tensor,
    label_coords: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> torch.Tensor:
    """
    """
    n = pred_coords.size(1)

    dmat_true = torch.sqrt(
        eps
        + torch.sum(
            (
                label_coords[..., None, :]
                - label_coords[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    dmat_pred = torch.sqrt(
        eps
        + torch.sum(
            (
                pred_coords[..., None, :]
                - pred_coords[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )
    dists_to_score = (
        (dmat_true < cutoff)
        * (1.0 - torch.eye(n, device=pred_coords.device))
    )

    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25

    dims = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    score = norm * (eps + torch.sum(dists_to_score * score, dim=dims))

    return score

def distogram_loss(
    logits,
    coords,
    min_bin=2.3125,
    max_bin=21.6875,
    no_bins=64,
    eps=1e-6,
    **kwargs,
) -> torch.Tensor:
    """
    """
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        no_bins - 1,
        device=logits.device,
    )
    boundaries = boundaries ** 2

    N, CA, C = [x.squeeze(-2) for x in coords.chunk(3, dim=-2)]
    # Infer CB coordinates.
    b = CA - N
    c = C - CA
    a = b.cross(c, dim=-1)
    CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA

    dists = torch.sum(
        (CB[..., None, :] - CB[..., None, :, :]) ** 2,
        dim=-1,
        keepdims=True,
    )

    true_bins = torch.sum(dists > boundaries, dim=-1)

    errors = softmax_cross_entropy(
        logits,
        F.one_hot(true_bins, no_bins),
    )

    loss = torch.mean(errors)

    return loss

def softmax_cross_entropy(logits, labels):
    loss = -1 * torch.sum(
        labels * F.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss

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


class GeoDockLoss(nn.Module):
    """
    """
    def __init__(self):
        super(GeoDockLoss, self).__init__()

    def forward(self, out: GeoDockOutput, batch):
        pred_coords = out.coords
        pred_rotat = out.rotat
        pred_trans = out.trans
        label_coords = batch['label_coords']
        label_rotat = batch['label_rotat']
        label_trans = batch['label_trans']
        sep = batch['protein1_embeddings'].size(1)
        pred_fape = get_fape(pred_coords, pred_rotat, pred_trans)
        label_fape = get_fape(label_coords, label_rotat, label_trans)

        loss_fns = {
            "intra_loss": lambda: intra_fape_loss(
                pred=pred_fape,
                label=label_fape,
                sep=sep,
                d_clamp=10.0,
            ),
            "inter_loss": lambda: inter_fape_loss(
                pred=pred_fape,
                label=label_fape,
                sep=sep,
                d_clamp=30.0,
            ),
            "dist_loss": lambda: distogram_loss(
                logits=out.dist_logits,
                coords=label_coords,
            ),
            "lddt_loss": lambda: lddt_loss(
                logits=out.lddt_logits,
                pred_coords=pred_coords,
                label_coords=label_coords,
            ),
            "violation_loss": lambda: violation_loss(
                pred_coords=pred_coords,
                sep=sep,
            ),
        }

        losses = {}
        for loss_name, loss_fn in loss_fns.items():
            loss = loss_fn()
            losses[loss_name] = loss
        
        return losses
