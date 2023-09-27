import torch


def compute_metrics(model, native):
    # get inputs
    model_rec = model[0].squeeze()
    model_lig = model[1].squeeze()
    native_rec = native[0].squeeze()
    native_lig = native[1].squeeze()
    
    # calc metrics
    c_rmsd = get_c_rmsd(model_rec, model_lig, native_rec, native_lig)
    i_rmsd = get_i_rmsd(model_rec, model_lig, native_rec, native_lig)
    l_rmsd = get_l_rmsd(model_rec, model_lig, native_rec, native_lig)
    fnat = get_fnat(model_rec, model_lig, native_rec, native_lig)
    DockQ = get_DockQ(i_rmsd, l_rmsd, fnat)
    rec_bb_rmsd = get_bb_rmsd(model_rec, native_rec)
    lig_bb_rmsd = get_bb_rmsd(model_lig, native_lig)

    # get interface res
    res1, res2 = get_interface_res(native_rec, native_lig, cutoff=10.0)
    rec_bb_irmsd = get_bb_rmsd(model_rec[res1], native_rec[res1])
    lig_bb_irmsd = get_bb_rmsd(model_lig[res2], native_lig[res2])

    return {'cRMS': c_rmsd, 'iRMS': i_rmsd, 'LRMS': l_rmsd, 'Fnat': fnat, 'DockQ': DockQ, 'Rec_BB_RMS': rec_bb_rmsd, 'Lig_BB_RMS': lig_bb_rmsd, 'Rec_BB_iRMS': rec_bb_irmsd, 'Lig_BB_iRMS': lig_bb_irmsd}
    
def get_interface_res(x1, x2, cutoff=10.0):
    # Calculate pairwise distances
    dist = x1[..., None, :, None, :] - x2[..., None, :, None, :, :]
    dist = (dist ** 2).sum(dim=-1).sqrt().flatten(start_dim=-2)

    # Find minimum distance between each pair of residues
    min_dist, _ = torch.min(dist, dim=-1)

    # Find index < cutoff 
    index = torch.where(min_dist < cutoff)
    res1 = torch.unique(index[0])
    res2 = torch.unique(index[1])
    return res1, res2

def get_dist(x1, x2):
    # Calculate pairwise distances
    dist = x1[..., None, :, None, :] - x2[..., None, :, None, :, :]
    dist = (dist ** 2).sum(dim=-1).sqrt().flatten(start_dim=-2)

    # Find minimum distance between each pair of residues
    min_dist, _ = torch.min(dist, dim=-1)

    return min_dist

def get_c_rmsd(model_rec, model_lig, native_rec, native_lig):
    pred = torch.cat([model_rec, model_lig], dim=0).flatten(end_dim=1)
    label = torch.cat([native_rec, native_lig], dim=0).flatten(end_dim=1)
    R, t = find_rigid_alignment(pred, label)
    pred = (R.mm(pred.T)).T + t
    return get_rmsd(pred, label).item()

def get_i_rmsd(model_rec, model_lig, native_rec, native_lig, cutoff=10.0):
    res1, res2 = get_interface_res(native_rec, native_lig, cutoff=cutoff)
    pred = torch.cat([model_rec[res1], model_lig[res2]], dim=0).flatten(end_dim=1)
    label = torch.cat([native_rec[res1], native_lig[res2]], dim=0).flatten(end_dim=1)
    R, t = find_rigid_alignment(pred, label)
    pred = (R.mm(pred.T)).T + t
    return get_rmsd(pred, label).item()

def get_l_rmsd(model_rec, model_lig, native_rec, native_lig):
    model_rec = model_rec.flatten(end_dim=1)
    model_lig = model_lig.flatten(end_dim=1)
    native_rec = native_rec.flatten(end_dim=1)
    native_lig = native_lig.flatten(end_dim=1)
    R, t = find_rigid_alignment(model_rec, native_rec)
    model_lig = (R.mm(model_lig.T)).T + t
    return get_rmsd(model_lig, native_lig).item()

def get_bb_rmsd(model, native):
    pred = model.flatten(end_dim=1)
    label = native.flatten(end_dim=1)
    R, t = find_rigid_alignment(pred, label)
    pred = (R.mm(pred.T)).T + t
    return get_rmsd(pred, label).item()

def get_fnat(model_rec, model_lig, native_rec, native_lig, cutoff=6.0):
    ligand_receptor_distance = get_dist(native_rec, native_lig)
    positive_tuple = torch.where(ligand_receptor_distance < cutoff)
    active_receptor = positive_tuple[0]
    active_ligand = positive_tuple[1]
    assert len(active_ligand) == len(active_receptor)
    ligand_receptor_distance_pred = get_dist(model_rec, model_lig)
    selected_elements = ligand_receptor_distance_pred[active_receptor, active_ligand]
    count = torch.count_nonzero(selected_elements < cutoff)
    Fnat = round(count.item() / (len(active_ligand) + 1e-6), 6)
    return Fnat

def get_DockQ(i_rmsd, l_rmsd, fnat):
    i_rmsd_scaled = 1.0 / (1.0 + (i_rmsd/1.5)**2)
    l_rmsd_scaled = 1.0 / (1.0 + (l_rmsd/8.5)**2)
    return (fnat + i_rmsd_scaled + l_rmsd_scaled) / 3

def get_rmsd(pred, label):
    rmsd = torch.sqrt(torch.mean(torch.sum((pred - label) ** 2.0, dim=-1)))
    return rmsd

def find_rigid_alignment(A, B):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.
        Args:
        -    A: Torch tensor of shape (N,D) -- Point Cloud to Align (source)
        -    B: Torch tensor of shape (N,D) -- Reference Point Cloud (target)
        Returns:
        -    R: optimal rotation
        -    t: optimal translation
    Test on rotation + translation and on rotation + translation + reflection
    """
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.mm(B_c)
    U, S, Vt = torch.linalg.svd(H)
    # Rotation matrix
    R = Vt.T.mm(U.T)
    
    # special reflection case
    if torch.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        SS = torch.diag(torch.tensor([1.,1.,-1.], device=R.device))
        R = (Vt.T @ SS) @ U.T
    
    # Translation vector
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    t = t.T
    return R, t.squeeze()

