import torch
import numpy as np
from typing import List, Optional, Union


_aa_1_3_dict = {
    'A': 'ALA',
    'C': 'CYS',
    'D': 'ASP',
    'E': 'GLU',
    'F': 'PHE',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'K': 'LYS',
    'L': 'LEU',
    'M': 'MET',
    'N': 'ASN',
    'P': 'PRO',
    'Q': 'GLN',
    'R': 'ARG',
    'S': 'SER',
    'T': 'THR',
    'V': 'VAL',
    'W': 'TRP',
    'Y': 'TYR',
    '-': 'GAP',
    'X': 'URI',
}


def exists(x):
    return x is not None


def place_fourth_atom(a_coord: torch.Tensor, b_coord: torch.Tensor,
                      c_coord: torch.Tensor, length: torch.Tensor,
                      planar: torch.Tensor,
                      dihedral: torch.Tensor) -> torch.Tensor:
    """
    Given 3 coords + a length + a planar angle + a dihedral angle, compute a fourth coord
    """
    bc_vec = b_coord - c_coord
    bc_vec = bc_vec / bc_vec.norm(dim=-1, keepdim=True)

    n_vec = (b_coord - a_coord).expand(bc_vec.shape).cross(bc_vec)
    n_vec = n_vec / n_vec.norm(dim=-1, keepdim=True)

    m_vec = [bc_vec, n_vec.cross(bc_vec), n_vec]
    d_vec = [
        length * torch.cos(planar),
        length * torch.sin(planar) * torch.cos(dihedral),
        -length * torch.sin(planar) * torch.sin(dihedral)
    ]

    d_coord = c_coord + sum([m * d for m, d in zip(m_vec, d_vec)])
    return d_coord


def save_PDB(out_pdb: str,
             coords: torch.Tensor,
             seq: str,
             b_factors: torch.Tensor = None,
             delim: int = None) -> None:
    """
    Write set of N, CA, C, O, CB coords to PDB file
    """

    if type(delim) == type(None):
        delim = -1
    
    if b_factors is None:
        b_factors = torch.zeros(coords.size(0), device=coords.device)

    atoms = ['N', 'CA', 'C', 'O', 'CB']

    with open(out_pdb, "a") as f:
        k = 0
        for r, residue in enumerate(coords):
            AA = _aa_1_3_dict[seq[r]]
            for a, atom in enumerate(residue):
                if AA == "GLY" and atoms[a] == "CB": continue
                x, y, z = atom
                f.write(
                    "ATOM  %5d  %-2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f %4.2f\n"
                    % (k + 1, atoms[a], AA, "A" if r <= delim else "B", r + 1,
                       x, y, z, 1, b_factors[r]))
                k += 1
        f.close()


def save_PDB_string(
    out_pdb: str,
    coords: torch.Tensor,
    seq: str,
    chains: List[str] = None,
    error: torch.Tensor = None,
    delims: Union[int, List[int]] = None,
    atoms=['N', 'CA', 'C', 'O', 'CB'],
    write_pdb=True,
) -> None:
    """
    Write set of N, CA, C, O, CB coords to PDB file
    """

    if not exists(chains):
        chains = ["A", "B"]

    if type(delims) == type(None):
        delims = -1
    elif type(delims) == int:
        delims = [delims]

    if not exists(error):
        error = torch.zeros(len(seq))

    pdb_string = ""
    k = 0
    for r, residue in enumerate(coords):
        AA = _aa_1_3_dict[seq[r]]
        for a, atom in enumerate(residue):
            if AA == "GLY" and atoms[a] == "CB": continue
            x, y, z = atom
            chain_id = chains[np.where(np.array(delims) - r > 0)[0][0]]
            pdb_string += "ATOM  %5d  %-2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.2f\n" % (
                k + 1, atoms[a], AA, chain_id, r + 1, x, y, z, 1, error[r])
            k += 1

            if k in delims:
                pdb_string += "TER  %5d      %3s %s%4d\n" % (
                    k + 1, AA, chain_id, r + 1)
                
    pdb_string += "END\n"

    if write_pdb:
        with open(out_pdb, "w") as f:
            f.write(pdb_string)

    return pdb_string