###
#   Inspired by FoldingTrunk implementation from https://github.com/facebookresearch/esm/blob/main/esm/esmfold/v1/trunk.py
###

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from einops import repeat, rearrange
from geodock.model.modules.structure_module import StructureModule
from geodock.model.modules.graph_module import GraphModule 
from geodock.utils.coords6d import get_coords6d
from geodock.utils.transforms import quaternion_to_matrix


class IterativeTransformer(nn.Module):
    def __init__(
        self,
        *,
        node_dim,
        edge_dim,
        gm_depth,
        sm_depth,
        num_iter,
    ):
        super().__init__()
        self.num_iter = num_iter
        self.recycle_bins = 16
        self.graph_module = GraphModule(node_dim=node_dim, edge_dim=edge_dim, depth=gm_depth)
        self.structure_module = StructureModule(node_dim=node_dim, edge_dim=edge_dim, depth=sm_depth)
        self.recycled_node_norm = nn.LayerNorm(node_dim)
        self.recycled_edge_norm = nn.LayerNorm(edge_dim)
        self.recycled_disto = nn.Linear(self.recycle_bins*4, edge_dim)
        self.plddt = PerResidueLDDTCaPredictor(c_in=node_dim, c_hidden=128, no_bins=50)
        self.disto = DistogramHead(c_z=edge_dim, no_bins=64)

    def forward(
        self, 
        node, 
        edge,
        mask=None,
        rotat=None,
        trans=None,
    ):
        device = node.device
        is_grad_enabled = torch.is_grad_enabled()

        # sample the number iterations
        if self.training:
            num_iter = random.randint(1, self.num_iter)
        else:
            num_iter = self.num_iter
        
        # get initial graph features
        node_0 = node
        edge_0 = edge
        recycled_node = torch.zeros_like(node)
        recycled_edge = torch.zeros_like(edge)
        recycled_bins = torch.zeros((*edge.shape[:-1], self.recycle_bins*4), device=device, dtype=torch.float)

        for i in range(num_iter):
            
            is_final_iter = i == (num_iter - 1)
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                # recycling
                recycled_node = self.recycled_node_norm(recycled_node.detach())
                recycled_edge = self.recycled_edge_norm(recycled_edge.detach())
                recycled_edge += self.recycled_disto(recycled_bins.detach())

                # graph module
                node, edge = self.graph_module(node_0 + recycled_node, edge_0 + recycled_edge)

                # structure module
                feats, rotat, trans = self.structure_module(
                    node,
                    edge,
                    mask=mask,
                )

                coords = self.get_coords(rotat, trans)
                
                recycled_node = node
                recycled_edge = edge
        
                # orientogram
                recycled_bins = self.orientogram(
                    coords,
                    self.recycle_bins,
                )

        lddt_logits = self.plddt(feats)
        dist_logits = self.disto(edge)
                
        return lddt_logits, dist_logits, coords, rotat, trans
    
    def orientogram(self, coords, num_bins):
        dist, omega, theta, phi = get_coords6d(coords.squeeze(0), use_Cb=True)

        mask = dist < 22.0
        
        num_bins = 16
        dist_bin = self.get_bins(dist, 2.0, 22.0, num_bins)
        omega_bin = self.get_bins(omega, -180.0, 180.0, num_bins)
        theta_bin = self.get_bins(theta, -180.0, 180.0, num_bins)
        phi_bin = self.get_bins(phi, -180.0, 180.0, num_bins)

        def mask_mat(mat):
            mat[~mask] = num_bins - 1
            mat.fill_diagonal_(num_bins - 1)
            return mat

        omega_bin = mask_mat(omega_bin)
        theta_bin = mask_mat(theta_bin)
        phi_bin = mask_mat(phi_bin)

        # to onehot
        dist = F.one_hot(dist_bin, num_classes=num_bins).float() 
        omega = F.one_hot(omega_bin, num_classes=num_bins).float() 
        theta = F.one_hot(theta_bin, num_classes=num_bins).float() 
        phi = F.one_hot(phi_bin, num_classes=num_bins).float() 
        
        return torch.cat([dist, omega, theta, phi], dim=-1).unsqueeze(0)

    def get_coords(self, rotat, trans):
        # get batch size, total_len
        batch_size, total_len, *_ = rotat.shape

        # initialize residue frames
        input_coords = torch.tensor([[-0.527, 1.359, 0.0], 
                                    [0.0, 0.0, 0.0],
                                    [1.525, 0.0, 0.0]], device=rotat.device)

        input_coords = repeat(input_coords, 'i j -> b n i j', b=batch_size, n=total_len)

        # update coordinates according to predicted rotations and translations   
        coords = torch.einsum('b n a j, b n i j -> b n a i', input_coords, rotat) + trans.unsqueeze(-2)
        return coords

    def get_bins(self, x, min_bin, max_bin, num_bins):
        # Coords are [... L x 3 x 3], where it's [N, CA, C] x 3 coordinates.
        boundaries = torch.linspace(
            min_bin,
            max_bin,
            num_bins - 1,
            device=x.device,
        )
        bins = torch.sum(x.unsqueeze(-1) > boundaries, dim=-1)  # [..., L, L]

        return bins


class PerResidueLDDTCaPredictor(nn.Module):
    def __init__(self, c_in, c_hidden, no_bins):
        super(PerResidueLDDTCaPredictor, self).__init__()

        self.no_bins = no_bins
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.layer_norm = nn.LayerNorm(self.c_in)
        self.linear_1 = nn.Linear(self.c_in, self.c_hidden)
        self.linear_2 = nn.Linear(self.c_hidden, self.c_hidden)
        self.linear_3 = nn.Linear(self.c_hidden, self.no_bins)
        self.relu = nn.ReLU()

    def forward(self, s):
        s = self.layer_norm(s)
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        return s


class DistogramHead(nn.Module):
    """
    Computes a distogram probability distribution.

    For use in computation of distogram loss, subsection 1.9.8
    """

    def __init__(self, c_z, no_bins, **kwargs):
        """
        Args:
            c_z:
                Input channel dimension
            no_bins:
                Number of distogram bins
        """
        super(DistogramHead, self).__init__()

        self.c_z = c_z
        self.no_bins = no_bins
        self.linear = nn.Linear(self.c_z, self.no_bins)

    def forward(self, z):  # [*, N, N, C_z]
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N, N, no_bins] distogram probability distribution
        """
        # [*, N, N, no_bins]
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        
        return logits

# ===test=== 
if __name__ == '__main__':
    node = torch.rand(1, 100, 64)
    edge = torch.rand(1, 100, 100, 32)
    model = IterativeTransformer(node_dim=64, edge_dim=32, gm_depth=3, sm_depth=3, num_iter=2)
    out = model(node=node, edge=edge)
    print(out)
