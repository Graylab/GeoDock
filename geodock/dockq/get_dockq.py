import csv
import torch
from tqdm import tqdm
from torch.utils import data
from geodock.utils.pdb import save_PDB, place_fourth_atom 
from geodock.datasets.repd2_dataset import RepD2Dataset
from geodock.utils.metrics import compute_metrics


def _cli():
    # get dataset
    model_dir = '/home/lchu11/scr4_jgray21/lchu11/RepD2/top5'
    native_dir = '/home/lchu11/scr4_jgray21/lchu11/RepD2/native'
    model_partner_dir = '/home/lchu11/scr4_jgray21/lchu11/RepD2/partners/model'
    native_partner_dir = '/home/lchu11/scr4_jgray21/lchu11/RepD2/partners/native'

    testset = RepD2Dataset(
        model_dir=model_dir,
        native_dir=native_dir,
        model_partner_dir=model_partner_dir,
        native_partner_dir=native_partner_dir,
    )

    """
    subset_indices = [0]
    subset = data.Subset(testset, subset_indices)
    test_dataloader = data.DataLoader(subset, batch_size=1, num_workers=6)
    """
    test_dataloader = data.DataLoader(testset, batch_size=1, num_workers=6)

    # load dataset
    device = 'cpu'

    # metrics
    metrics_list = []

    for batch in tqdm(test_dataloader):
        _id = batch['id'][0]
        model_coords1 = batch['model_coords1'].to(device)
        model_coords2 = batch['model_coords2'].to(device)
        native_coords1 = batch['native_coords1'].to(device)
        native_coords2 = batch['native_coords2'].to(device)

        pred = (model_coords1, model_coords2)
        label = (native_coords1, native_coords2)

        # get metrics
        metrics = {'id': _id}
        try:
            assert model_coords1.size(1) == native_coords1.size(1), "partner 1 mismatch"
            assert model_coords2.size(1) == native_coords2.size(1), "partner 2 mismatch"
            metrics.update(compute_metrics(pred, label))
        except Exception as e: 
            print(_id)
            print(e)
        metrics_list.append(metrics)

    
    out_csv_dir = 'RepD2.csv'

    with open(out_csv_dir, 'w', newline='') as file:
        header = list(metrics_list[0].keys())
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        for row in metrics_list:
            writer.writerow(row)

        print(f"Results saved to {out_csv_dir}")


def get_full_coords(coords):
    #get full coords
    N, CA, C = [x.squeeze(-2) for x in coords.chunk(3, dim=-2)]
    # Infer CB coordinates.
    b = CA - N
    c = C - CA
    a = b.cross(c, dim=-1)
    CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
    
    O = place_fourth_atom(torch.roll(N, -1, 0),
                                    CA, C,
                                    torch.tensor(1.231),
                                    torch.tensor(2.108),
                                    torch.tensor(-3.142))
    full_coords = torch.stack(
        [N, CA, C, O, CB], dim=1)
    
    return full_coords


def compute_plddt(logits: torch.Tensor) -> torch.Tensor:
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bounds = torch.arange(
        start=0.5 * bin_width, end=1.0, step=bin_width, device=logits.device
    )
    probs = F.softmax(logits, dim=-1)
    pred_lddt_ca = torch.sum(
        probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape),
        dim=-1,
    )
    return pred_lddt_ca

def compute_interface_lddt(plddt, coords, cutoff=10.0):
    x1 = coords[0]
    x2 = coords[1]
    # Calculate pairwise distances
    dist = x1[..., None, :, None, :] - x2[..., None, :, None, :, :]
    dist = (dist ** 2).sum(dim=-1).sqrt().flatten(start_dim=-2)

    # Find minimum distance between each pair of residues
    min_dist, _ = torch.min(dist, dim=-1)
   
    # Find index < cutoff 
    index = torch.where(min_dist < cutoff)
    res1 = torch.unique(index[1])
    res2 = torch.unique(index[2])
    interface_res = torch.cat([res1, res2+x1.size(1)])
    i_plddt = plddt[interface_res]

    return i_plddt


if __name__ == '__main__':
    _cli()


