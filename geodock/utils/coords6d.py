###
# Modified from https://github.com/RosettaCommons/trRosetta2/blob/main/trRosetta/coords6d.py
###

import math
import torch
from einops import repeat


def calc_dist(a_coords, b_coords):
    assert a_coords.shape == b_coords.shape
    mat_shape = list(a_coords.shape)
    mat_shape.insert(-1, mat_shape[-2])

    a_coords = a_coords.unsqueeze(-3).expand(mat_shape)
    b_coords = b_coords.unsqueeze(-2).expand(mat_shape)

    dist_mat = (a_coords - b_coords).norm(dim=-1)

    return dist_mat


def calc_dihedral(a_coords,
                  b_coords,
                  c_coords,
                  d_coords,
                  convert_to_degree=True):
    b1 = a_coords - b_coords
    b2 = b_coords - c_coords
    b3 = c_coords - d_coords

    n1 = torch.cross(b1, b2)
    n1 = torch.div(n1, n1.norm(dim=-1, keepdim=True))
    n2 = torch.cross(b2, b3)
    n2 = torch.div(n2, n2.norm(dim=-1, keepdim=True))
    m1 = torch.cross(n1, torch.div(b2, b2.norm(dim=-1, keepdim=True)))

    dihedral = torch.atan2((m1 * n2).sum(-1), (n1 * n2).sum(-1))

    if convert_to_degree:
        dihedral = dihedral * 180 / math.pi

    return dihedral


def calc_planar(a_coords, b_coords, c_coords, convert_to_degree=True):
    v1 = a_coords - b_coords
    v2 = c_coords - b_coords

    a = (v1 * v2).sum(-1)
    b = v1.norm(dim=-1) * v2.norm(dim=-1)

    planar = torch.acos(a / b)

    if convert_to_degree:
        planar = planar * 180 / math.pi

    return planar


# get 6d coordinates from x,y,z coords of N,Ca,C atoms
def get_coords6d(xyz, use_Cb=False):

    n = xyz.shape[0]

    # three anchor atoms
    N  = xyz[..., 0, :]
    Ca = xyz[..., 1, :]
    C  = xyz[..., 2, :]

    # recreate Cb given N,Ca,C
    b = Ca - N
    c = C - Ca
    a = torch.cross(b, c, dim=-1)
    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca

    if use_Cb:
        dist = calc_dist(Cb, Cb)
    else:
        dist = calc_dist(Ca, Ca)

    
    omega = calc_dihedral(
                repeat(Ca, 'r i -> r c i', c=n), 
                repeat(Cb, 'r i -> r c i', c=n), 
                repeat(Cb, 'c i -> r c i', r=n), 
                repeat(Ca, 'c i -> r c i', r=n),
            )

    theta = calc_dihedral(
                repeat(N, 'r i -> r c i', c=n), 
                repeat(Ca, 'r i -> r c i', c=n), 
                repeat(Cb, 'r i -> r c i', c=n), 
                repeat(Cb, 'c i -> r c i', r=n),
            )
    phi = calc_planar(
                repeat(Ca, 'r i -> r c i', c=n), 
                repeat(Cb, 'r i -> r c i', c=n), 
                repeat(Cb, 'c i -> r c i', r=n), 
            )


    return dist, omega, theta, phi

if __name__ == '__main__':
    coords = torch.randn(10, 3, 3)
    out = get_coords6d(coords)
    print(out)

