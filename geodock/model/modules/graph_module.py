###
#   Modified from triangle multiplicative update implementation from https://github.com/lucidrains/triangle-multiplicative-module
###

import torch
import torch.nn as nn
from torch import einsum
from einops.layers.torch import Rearrange
from einops import rearrange, repeat

# helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# classes
class NodeAttentionWithEdgeBias(nn.Module):
    """
    hidden_dim = 32
    """
    def __init__(
        self,
        *,
        node_dim,
        edge_dim,
        head_dim=32,
        heads=8,
    ):
        super().__init__()
        hidden_dim = head_dim * heads
        self.heads = heads
        self.node_norm = nn.LayerNorm(node_dim)
        self.edge_norm = nn.LayerNorm(edge_dim)

        self.q = nn.Linear(node_dim, hidden_dim, bias=False)
        self.k = nn.Linear(node_dim, hidden_dim, bias=False)
        self.v = nn.Linear(node_dim, hidden_dim, bias=False)
        self.b = nn.Linear(edge_dim, heads, bias=False)
        self.g = nn.Sequential(
                    nn.Linear(node_dim, hidden_dim),
                    nn.Sigmoid()
                )
        self.attn_logits_scale = head_dim ** -0.5
        self.to_out = nn.Linear(hidden_dim, node_dim)

    def forward(self, node, edge):
        h = self.heads
        # layer norm
        x = self.node_norm(node)
        y = self.edge_norm(edge)
        # get queries, keys, values, gated, pairwise
        q, k, v, g, b = self.q(x), self.k(x), self.v(x), self.g(x), self.b(y)
        # split out heads
        q, k, v, g = map(lambda t: rearrange(t, 'b i (h d) -> (b h) i d', h=h), (q, k, v, g))
        b = rearrange(b, 'b i j h -> (b h) i j', h=h)
        # derive attn logits
        attn_logits = einsum('b i d, b j d -> b i j', q, k) * self.attn_logits_scale + b
        # attention
        attn = attn_logits.softmax(dim=-1)
        # aggregate values
        results = einsum('b i j, b j d -> b i d', attn, v) * g
        # merge back heads
        results = rearrange(results, '(b h) i d -> b i (h d)', h=h)
        return self.to_out(results)


class Transition(nn.Module):
    """
    hidden_dim = 2 * node_dim
    """
    def __init__(
        self,
        *,
        dim,
        n=2,
    ):
        super().__init__() 
        self.norm = nn.LayerNorm(dim)
        self.to_act = nn.Linear(dim, dim * n)
        self.act = nn.ReLU()
        self.to_out = nn.Linear(dim * n, dim)
    
    def forward(self, x):
        x = self.norm(x)
        x = self.to_act(x)
        x = self.act(x)
        x = self.to_out(x)
        return x


class NodeUpdate(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
    ):
        super().__init__()
        self.attention = NodeAttentionWithEdgeBias(
            node_dim=node_dim, 
            edge_dim=edge_dim,
        )

        self.transition = Transition(
            dim=node_dim
        )

    def forward(self, node, edge):
        x = node + self.attention(node, edge)
        x = x + self.transition(x)
        return x


class TriangleMultiplicativeModule(nn.Module):
    """
    hidden_dim = edge_dim
    """
    def __init__(
        self,
        *,
        dim,
        hidden_dim=None,
        mix='ingoing'
    ):
        super().__init__()
        assert mix in {'ingoing', 'outgoing'}, 'mix must be either ingoing or outgoing'

        hidden_dim = default(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)

        self.left_gate = nn.Linear(dim, hidden_dim)
        self.right_gate = nn.Linear(dim, hidden_dim)
        self.out_gate = nn.Linear(dim, hidden_dim)

        # initialize all gating to be identity

        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.)
            nn.init.constant_(gate.bias, 1.)

        if mix == 'outgoing':
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == 'ingoing':
            self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, mask = None):
        assert x.shape[1] == x.shape[2], 'feature map must be symmetrical'
        if exists(mask):
            mask = rearrange(mask, 'b i j -> b i j ()')

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        if exists(mask):
            left = left * mask
            right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum(self.mix_einsum_eq, left, right)

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


class EdgeUpdate(nn.Module):
    """
    dropout = 0.1
    """
    def __init__(
        self,
        dim,
        dropout=0.1,
    ):
        super().__init__()
        self.triangle_multiply_outgoing = TriangleMultiplicativeModule(dim=dim, mix='outgoing')
        self.triangle_multiply_ingoing = TriangleMultiplicativeModule(dim=dim, mix='ingoing')
        self.transition = Transition(dim=dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x,
        mask=None,
    ):
        x = self.dropout(self.triangle_multiply_outgoing(x, mask=mask)) + x
        x = self.dropout(self.triangle_multiply_ingoing(x, mask=mask)) + x
        x = self.transition(x) + x
        return x


class NodeToEdge(nn.Module):
    """
    hidden_dim = 2 * node_dim
    """
    def __init__(
        self, 
        node_dim, 
        edge_dim,
        hidden_dim=None,
    ):
        super().__init__()
        hidden_dim = default(hidden_dim, node_dim)
        self.layernorm = nn.LayerNorm(node_dim)
        self.proj = nn.Linear(node_dim, hidden_dim * 2)
        self.o_proj = nn.Linear(2 * hidden_dim, edge_dim)

        torch.nn.init.zeros_(self.proj.bias)
        torch.nn.init.zeros_(self.o_proj.bias)

    def forward(self, node):
        assert len(node.shape) == 3

        s = self.layernorm(node)
        s = self.proj(s)
        q, k = s.chunk(2, dim=-1)

        prod = q[:, None, :, :] * k[:, :, None, :]
        diff = q[:, None, :, :] - k[:, :, None, :]

        x = torch.cat([prod, diff], dim=-1)
        x = self.o_proj(x)

        return x


class GraphBlock(nn.Module):
    def __init__(
        self,
        *,
        node_dim, 
        edge_dim,
    ):
        super().__init__()
        self.node_update = NodeUpdate(
            node_dim = node_dim,
            edge_dim = edge_dim
        )
        self.node_to_edge = NodeToEdge(
            node_dim = node_dim,
            edge_dim = edge_dim,
        )
        self.edge_update = EdgeUpdate(
            dim = edge_dim 
        )
    
    def forward(self, node, edge):
        node = self.node_update(node, edge)
        edge = edge + self.node_to_edge(node)
        edge = self.edge_update(edge)
        return node, edge


class GraphModule(nn.Module):
    def __init__(
        self,
        *,
        node_dim, 
        edge_dim,
        depth,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(depth):
            self.blocks.append(GraphBlock(node_dim=node_dim, edge_dim=edge_dim))

    def forward(self, node, edge):
        for block in self.blocks:
            node, edge = block(node, edge)
        return node, edge


# ===test=== 
if __name__ == '__main__':
    node = torch.rand(1, 10, 64)
    edge = torch.rand(1, 10, 10, 64)
    model = GraphModule(node_dim=64, edge_dim=64, depth=2)
    output = model(node, edge)
    print(output)
