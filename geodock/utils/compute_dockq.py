import math
import numpy as np
import scipy.spatial as spa


# Input: expects 3xN matrix of points
# Returns such R, t so that rmsd(R @ A + t, B) is min
# Uses Kabsch algorithm (https://en.wikipedia.org/wiki/Kabsch_algorithm)
# R = 3x3 rotation matrix
# t = 3x1 column vector
# This already takes residue identity into account.
def rigid_transform_Kabsch_3D(A, B):
    assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")


    # find mean column wise: 3 x 1
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    # find rotation
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        SS = np.diag([1.,1.,-1.])
        R = (Vt.T @ SS) @ U.T
    assert math.fabs(np.linalg.det(R) - 1) < 1e-5

    t = -R @ centroid_A + centroid_B
    return R, t


def compute_rmsd(pred, true):
    return np.sqrt(np.mean(np.sum((pred - true) ** 2, axis=1)))


def compute_complex_rmsd(ligand_coors_pred, ligand_coors_true, receptor_coors):
    complex_coors_pred = np.concatenate((ligand_coors_pred, receptor_coors), axis=0)
    complex_coors_true = np.concatenate((ligand_coors_true, receptor_coors), axis=0)

    R,t = rigid_transform_Kabsch_3D(complex_coors_pred.T, complex_coors_true.T)
    complex_coors_pred_aligned = (R @ complex_coors_pred.T + t).T

    complex_rmsd = compute_rmsd(complex_coors_pred_aligned, complex_coors_true)

    return complex_rmsd


def compute_interface_rmsd(ligand_coors_pred, ligand_coors_true, receptor_coors):
    ligand_receptor_distance = spa.distance.cdist(ligand_coors_true, receptor_coors)
    positive_tuple = np.where(ligand_receptor_distance < 8.)
    
    active_ligand = positive_tuple[0]
    active_receptor = positive_tuple[1]
    
    ligand_coors_pred = ligand_coors_pred[active_ligand, :]
    ligand_coors_true = ligand_coors_true[active_ligand, :]
    receptor_coors = receptor_coors[active_receptor, :]

    complex_coors_pred = np.concatenate((ligand_coors_pred, receptor_coors), axis=0)
    complex_coors_true = np.concatenate((ligand_coors_true, receptor_coors), axis=0)

    R,t = rigid_transform_Kabsch_3D(complex_coors_pred.T, complex_coors_true.T)
    complex_coors_pred_aligned = (R @ complex_coors_pred.T + t).T

    interface_rmsd = compute_rmsd(complex_coors_pred_aligned, complex_coors_true)

    return interface_rmsd


def compute_ligand_rmsd(ligand_coors_pred, ligand_coors_true):
    ligand_rmsd = compute_rmsd(ligand_coors_pred, ligand_coors_true)
    
    return ligand_rmsd


def compute_Fnat(ligand_coors_pred, ligand_coors_true, receptor_coors):
    ligand_receptor_distance = spa.distance.cdist(ligand_coors_true, receptor_coors)
    positive_tuple = np.where(ligand_receptor_distance < 8.)
    active_ligand = positive_tuple[0]
    active_receptor = positive_tuple[1]
    assert len(active_ligand) == len(active_receptor)

    ligand_receptor_distance_pred = spa.distance.cdist(ligand_coors_pred, receptor_coors)
    selected_elements = ligand_receptor_distance_pred[active_ligand, active_receptor]

    count = np.count_nonzero(selected_elements < 8.)

    Fnat = round(count / (len(active_ligand) + 1e-6), 6)

    return Fnat


def compute_DockQ(i_rmsd, l_rmsd, fnat):
    i_rmsd_scaled = 1.0 / (1.0 + (i_rmsd/1.5)**2)
    l_rmsd_scaled = 1.0 / (1.0 + (l_rmsd/8.5)**2)
    return (fnat + i_rmsd_scaled + l_rmsd_scaled) / 3


def compute_metrics(ligand_coors_pred, ligand_coors_true, receptor_coors):
    complex_rmsd = compute_complex_rmsd(ligand_coors_pred, ligand_coors_true, receptor_coors)
    interface_rmsd = compute_interface_rmsd(ligand_coors_pred, ligand_coors_true, receptor_coors)
    ligand_rmsd = compute_ligand_rmsd(ligand_coors_pred, ligand_coors_true)
    Fnat = compute_Fnat(ligand_coors_pred, ligand_coors_true, receptor_coors)
    DockQ = compute_DockQ(interface_rmsd, ligand_rmsd, Fnat)

    return {"CRMS": complex_rmsd,"iRMS": interface_rmsd, "LRMS": ligand_rmsd, "Fnat": Fnat, "DockQ": DockQ}