import esm
import torch
from time import time
from geodock.utils.embed import embed
from geodock.utils.docking import dock
from geodock.model.GeoDock import GeoDock
from esm.inverse_folding.util import load_coords


class GeoDockRunner():
    """
    Wrapper for GeoDock model predictions.
    """
    def __init__(self):

        # Load ESM-2 model
        self.esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = alphabet.get_batch_converter()
        self.esm_model.eval()  # disables dropout for deterministic results

        # Load GeoDock model
        self.model = GeoDock().eval()

    def embed(
        self, 
        seq1, 
        seq2,
        coords1,
        coords2,
    ):
        start_time = time()
        embeddings = embed(
            seq1, 
            seq2,
            coords1,
            coords2,
            self.esm_model,
            self.batch_converter,
        )

        print(f"Completed embedding in {time() - start_time:.2f} seconds.")

        return embeddings
    
    def dock(self, partner1, partner2):
        # Get seqs and coords
        coords1, seq1 = load_coords(partner1, chain=None)
        coords2, seq2 = load_coords(partner2, chain=None)
        coords1 = torch.nan_to_num(torch.from_numpy(coords1))
        coords2 = torch.nan_to_num(torch.from_numpy(coords2))

        # Get embeddings
        model_in = self.embed(
            seq1,
            seq2,
            coords1,
            coords2,
        )

        # Start docking
        start_time = time()
        dock(
            seq1,
            seq2,
            model_in,
            self.model,
        )

        print(f"Completed docking in {time() - start_time:.2f} seconds.")


if __name__ == '__main__':
    partner1="/home/lchu11/scr4_jgray21/lchu11/Docking-dev/data/equidock/a9_1a9x.pdb4_0.dill_r_b_COMPLEX.pdb"
    partner2="/home/lchu11/scr4_jgray21/lchu11/Docking-dev/data/equidock/a9_1a9x.pdb4_0.dill_l_b_COMPLEX.pdb"

    geodock = GeoDockRunner()
    pred = geodock.dock(
        partner1, 
        partner2,
    )