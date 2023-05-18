from dataclasses import dataclass
from typing import List, Optional, Union
import torch


@dataclass
class GeoDockInput():
    protein1_embeddings: torch.FloatTensor
    protein2_embeddings: torch.FloatTensor
    pair_embeddings: torch.FloatTensor
    positional_embeddings: torch.FloatTensor


@dataclass
class GeoDockOutput():
    coords: torch.FloatTensor
    rotat: torch.FloatTensor
    trans: torch.FloatTensor
    lddt_logits: torch.FloatTensor
    dist_logits: torch.FloatTensor