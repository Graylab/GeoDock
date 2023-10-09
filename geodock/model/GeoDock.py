import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils import data
from einops import repeat, rearrange
from geodock.datasets.geodock_dataset import GeoDockDataset
from geodock.model.interface import GeoDockInput, GeoDockOutput
from geodock.model.modules.iterative_transformer import IterativeTransformer
from geodock.utils.loss import GeoDockLoss


class GeoDock(pl.LightningModule):
    def __init__(
        self,
        node_dim: int = 64,
        edge_dim: int = 64,
        gm_depth: int = 1,
        sm_depth: int = 1,
        num_iter: int = 1,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        # hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()

        # Embeddings
        esm_dim = 1280
        pair_dim = 65
        positional_dim = 69
        self.esm_to_node = nn.Linear(esm_dim, node_dim)
        self.pair_to_edge = nn.Linear(pair_dim, edge_dim)
        self.positional_to_edge = nn.Linear(positional_dim, edge_dim)

        # Networks
        self.net = IterativeTransformer(
            node_dim=node_dim, 
            edge_dim=edge_dim, 
            gm_depth=gm_depth,
            sm_depth=sm_depth,
            num_iter=num_iter,
        )

        # Loss
        self.loss = GeoDockLoss()

    def forward(self, input: GeoDockInput):
        # Inputs
        protein1_embeddings = input.protein1_embeddings
        protein2_embeddings = input.protein2_embeddings
        pair_embeddings = input.pair_embeddings
        positional_embeddings = input.positional_embeddings

        # Node embedding
        protein_embeddings = torch.cat([protein1_embeddings, protein2_embeddings], dim=1)
        nodes = self.esm_to_node(protein_embeddings)

        # Edge embedding
        edges = self.pair_to_edge(pair_embeddings) + self.positional_to_edge(positional_embeddings)

        # Networks
        lddt_logits, dist_logits, coords, rotat, trans = self.net(
            node=nodes, 
            edge=edges, 
        )

        # Outputs
        output = GeoDockOutput(
            coords=coords,
            rotat=rotat,
            trans=trans,
            lddt_logits=lddt_logits,
            dist_logits=dist_logits,
        )

        return output

    def step(self, batch, batch_idx):
        # Get info from the batch
        protein1_embeddings = batch['protein1_embeddings']
        protein2_embeddings = batch['protein2_embeddings']
        pair_embeddings = batch['pair_embeddings']
        positional_embeddings = batch['positional_embeddings']

        # Prepare GeoDock input
        input = GeoDockInput(
            protein1_embeddings=protein1_embeddings,
            protein2_embeddings=protein2_embeddings,
            pair_embeddings=pair_embeddings,
            positional_embeddings=positional_embeddings,
        )

        # Get GeoDock output
        output = self(input)

        # Loss
        losses = self.loss(output, batch)
        intra_loss = losses['intra_loss']
        inter_loss = losses['inter_loss']
        dist_loss = losses['dist_loss']
        lddt_loss = losses['lddt_loss']
        violation_loss = losses['violation_loss']

        if self.current_epoch < 5:
            loss = intra_loss + 0.3*dist_loss + 0.01*lddt_loss 
        elif self.current_epoch < 10:
            loss = intra_loss + inter_loss + 0.3*dist_loss + 0.01*lddt_loss
        else:
            loss = intra_loss + inter_loss + violation_loss + 0.3*dist_loss + 0.01*lddt_loss

        losses.update({'loss': loss})

        return losses

    def _log(self, losses, train=True):
        phase = "train" if train else "val"
        for loss_name, indiv_loss in losses.items():
            self.log(
                f"{phase}/{loss_name}", 
                indiv_loss, 
                on_step=train, on_epoch=(not train), logger=True,
            )

            if(train):
                self.log(
                    f"{phase}/{loss_name}_epoch",
                    indiv_loss,
                    on_step=False, on_epoch=True, logger=True,
                )

    def training_step(self, batch, batch_idx):
        losses = self.step(batch, batch_idx)
        self._log(losses, train=True)
        
        return losses['loss']

    def validation_step(self, batch, batch_idx):
        losses = self.step(batch, batch_idx)
        self._log(losses, train=False)
        
        return losses['loss']

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        return optimizer


if __name__ == '__main__':
    dataset = GeoDockDataset()

    subset_indices = [0]
    subset = data.Subset(dataset, subset_indices)

    #load dataset
    dataloader = data.DataLoader(subset, batch_size=1, num_workers=6)
    
    model = GeoDock()
    trainer = pl.Trainer()
    trainer.validate(model, dataloader)
