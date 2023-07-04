###
#   Modified from IPA implementation from https://github.com/lucidrains/invariant-point-attention
###

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from torch import einsum
from geodock.utils.transforms import quaternion_to_matrix


# Helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


class InvariantPointAttention(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        heads=8,
        scalar_key_dim=16,
        scalar_value_dim=16,
        point_key_dim=4,
        point_value_dim=4,
        eps=1e-8,
    ):
        super().__init__()
        self.eps = eps
        self.heads = heads

        # Number of attentions
        num_attn_logits = 3 

        # QKV projection for scalar attention
        self.scalar_attn_logits_scale = (num_attn_logits * scalar_key_dim) ** -0.5
        self.to_scalar_q = nn.Linear(node_dim, scalar_key_dim * heads, bias=False)
        self.to_scalar_k = nn.Linear(node_dim, scalar_key_dim * heads, bias=False)
        self.to_scalar_v = nn.Linear(node_dim, scalar_value_dim * heads, bias=False)

        # QKV projection for point attention
        point_weight_init_value = torch.log(torch.exp(torch.full((heads,), 1.)) - 1.)
        self.point_weights = nn.Parameter(point_weight_init_value)
        self.point_attn_logits_scale = ((num_attn_logits * point_key_dim) * (9 / 2)) ** -0.5
        self.to_point_q = nn.Linear(node_dim, point_key_dim * heads * 3, bias=False)
        self.to_point_k = nn.Linear(node_dim, point_key_dim * heads * 3, bias=False)
        self.to_point_v = nn.Linear(node_dim, point_value_dim * heads * 3, bias=False)

        # Pairwise representation projection to attention bias
        self.pairwise_attn_logits_scale = num_attn_logits ** -0.5

        self.to_pairwise_attn_bias = nn.Sequential(
            nn.Linear(edge_dim, heads),
            Rearrange('b ... h -> (b h) ...')
        )

        # Combine out - scalar dim + pairwise dim + point dim * (3 for coordinates in R3 and then 1 for norm)
        self.to_out = nn.Linear(heads * (scalar_value_dim + edge_dim + point_value_dim * (3 + 1)), node_dim)

    def forward(
        self,
        node,
        edge,
        mask=None,
        *,
        rotations,
        translations,
    ):
        x, b, h, eps = node, node.size(0), self.heads, self.eps

        # Get queries, keys, values for scalar and point (coordinate-aware) attention pathways
        q_scalar, k_scalar, v_scalar = self.to_scalar_q(x), self.to_scalar_k(x), self.to_scalar_v(x)
        q_point, k_point, v_point = self.to_point_q(x), self.to_point_k(x), self.to_point_v(x)

        # Split out heads
        q_scalar, k_scalar, v_scalar = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q_scalar, k_scalar, v_scalar))
        q_point, k_point, v_point = map(lambda t: rearrange(t, 'b n (h d c) -> (b h) n d c', h=h, c=3), (q_point, k_point, v_point))
        rotations = repeat(rotations, 'b n r1 r2 -> (b h) n r1 r2', h=h)
        translations = repeat(translations, 'b n c -> (b h) n () c', h=h)

        # Rotate qkv points into global frame
        q_point = einsum('b n d c, b n r c -> b n d r', q_point, rotations) + translations
        k_point = einsum('b n d c, b n r c -> b n d r', k_point, rotations) + translations
        v_point = einsum('b n d c, b n r c -> b n d r', v_point, rotations) + translations

        # Derive attn logits for scalar and pairwise
        attn_logits_scalar = einsum('b i d, b j d -> b i j', q_scalar, k_scalar) * self.scalar_attn_logits_scale
        
        attn_logits_pairwise = self.to_pairwise_attn_bias(edge) * self.pairwise_attn_logits_scale
        
        # Derive attn logits for point attention
        point_qk_diff = rearrange(q_point, 'b i d c -> b i () d c') - rearrange(k_point, 'b j d c -> b () j d c')
        point_dist = (point_qk_diff ** 2.0).sum(dim=-2)
        point_weights = F.softplus(self.point_weights)
        point_weights = repeat(point_weights, 'h -> (b h) () () ()', b=b)
        attn_logits_points = -0.5 * (point_dist * point_weights * self.point_attn_logits_scale).sum(dim=-1)

        # Add look ahead mask for point attention
        if mask is not None:
            mask = repeat(mask, 'b w l -> (h b) w l', h=h)
            attn_logits_points += (mask * -1e9)

        # Combine attn logits
        attn_logits = attn_logits_scalar + attn_logits_points
        attn_logits = attn_logits + attn_logits_pairwise
        
        # Attention
        attn = attn_logits.softmax(dim=-1)

        # Aggregate values
        results_scalar = einsum('b i j, b j d -> b i d', attn, v_scalar)
        attn_with_heads = rearrange(attn, '(b h) i j -> b h i j', h=h)
        results_pairwise = einsum('b h i j, b i j d -> b h i d', attn_with_heads, edge)

        # Aggregate point values
        results_points = einsum('b i j, b j d c -> b i d c', attn, v_point)

        # Rotate aggregated point values back into local frame
        results_points = einsum('b n d r, b n c r -> b n d c', results_points - translations, rotations.transpose(-1, -2))
        results_points_norm = torch.sqrt(sum(map(torch.square, results_points.unbind(dim=-1))) + eps)

        # Merge back heads
        results_scalar = rearrange(results_scalar, '(b h) n d -> b n (h d)', h=h)
        results_points = rearrange(results_points, '(b h) n d c -> b n (h d c)', h=h)
        results_points_norm = rearrange(results_points_norm, '(b h) n d -> b n (h d)', h=h)

        results = (results_scalar, results_points, results_points_norm)

        results_pairwise = rearrange(results_pairwise, 'b h n d -> b n (h d)', h=h)
        results = (*results, results_pairwise)

        # Concat results and project out
        results = torch.cat(results, dim=-1)
        return self.to_out(results)


class Transition(nn.Module):
    """
    hidden_dim = node_dim
    """
    def __init__(
        self,
        *,
        dim,
    ):
        super().__init__() 
        self.layer1 = nn.Linear(dim, dim)
        self.layer2 = nn.Linear(dim, dim)
        self.layer3 = nn.Linear(dim, dim)
        self.act = nn.ReLU()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        x = self.act(x)
        x = self.layer3(x)
        return x


class IPABlock(nn.Module):
    """
    dropout = 0.1
    """
    def __init__(
        self,
        *,
        node_dim,
        edge_dim,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()
        self.attn_norm = nn.LayerNorm(node_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn = InvariantPointAttention(node_dim=node_dim, edge_dim=edge_dim, **kwargs)
        
        self.ff_norm = nn.LayerNorm(node_dim)
        self.ff_dropout = nn.Dropout(dropout)
        self.ff = Transition(dim=node_dim)

    def forward(self, node, edge, **kwargs):
        x = self.attn(node, edge, **kwargs) + node
        x = self.attn_dropout(x)
        x = self.attn_norm(x) 

        x = self.ff(x) + x
        x = self.ff_dropout(x)
        x = self.ff_norm(x) 
        return x


class StructureModule(nn.Module):
    def __init__(
        self,
        *,
        node_dim,
        edge_dim,
        depth,
    ):
        super().__init__()
        self.depth = depth
        self.IPA_block = IPABlock(
            node_dim=node_dim, 
            edge_dim=edge_dim
        )

        self.to_update = nn.Linear(node_dim, 6)

    def forward(
        self, 
        node, 
        edge,
        mask=None,
        rotations=None,
        translations=None,
    ):
        device = node.device
        b, n, *_ = node.shape

        # If no initial rotations passed in, start from identity
        if not exists(rotations):
            rotations = torch.eye(3, device=device) # initial rotations
            rotations = repeat(rotations, 'i j -> b n i j', b=b, n=n)

        # If no initial translations passed in, start from identity
        if not exists(translations):
            translations = torch.zeros((b, n, 3), device=device)

        for i in range(self.depth):
            # update nodes and backbone frames
            node = self.IPA_block(
                node,
                edge,
                mask=mask,
                rotations=rotations,
                translations=translations
            )

            quaternions_update, translations_update = self.to_update(node).split([3, 3], dim=-1)
            quaternions_update = F.pad(quaternions_update, (1, 0), value=1.)
            quaternions_update = quaternions_update / torch.linalg.norm(quaternions_update, dim=-1, keepdim=True)
            rotations_update = quaternion_to_matrix(quaternions_update)
            translations = einsum('b n i j, b n j -> b n i', rotations, translations_update) + translations
            rotations = einsum('b n i j, b n j k -> b n i k', rotations, rotations_update)
            
        return node, rotations, translations


# ===test=== 
if __name__ == '__main__':
    node = torch.rand(1, 10, 64)
    edge = torch.rand(1, 10, 10, 64)
    torch.manual_seed(0)
    model = StructureModule(node_dim=64, edge_dim=64, depth=3)
    out = model(node, edge)
    print(out)
