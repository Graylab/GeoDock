import os
import torch
import random
import torch.nn.functional as F
from torch.utils import data
from tqdm import tqdm
from einops import repeat
from geodock.utils.pdb import save_PDB, place_fourth_atom 
from geodock.utils.coords6d import get_coords6d


class GeoDockDataset(data.Dataset):
    def __init__(
        self, 
        dataset: str = 'dips_train_500',
        out_pdb: bool = False,
        out_png: bool = False,
        is_training: bool = True,
        is_testing: bool = False,
        prob: float = 1.0,
        count: int = 0,
        use_Cb: bool = True,
    ):
        self.dataset = dataset 
        self.out_pdb = out_pdb
        self.out_png = out_png
        self.is_training = is_training
        self.is_testing = is_testing
        self.prob = prob
        self.count = count
        self.use_Cb = use_Cb

        if dataset == 'dips_train_0.3':
            self.data_dir = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/dips/pt_files"
            self.data_list = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/dips_equidock/train_list_0.3.txt" 
            with open(self.data_list, 'r') as f:
                lines = f.readlines()
            self.file_list = [line.strip() for line in lines] 

        if dataset == 'dips_val_0.3':
            self.data_dir = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/dips/pt_files"
            self.data_list = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/dips_equidock/val_list_0.3.txt" 
            with open(self.data_list, 'r') as f:
                lines = f.readlines()
            self.file_list = [line.strip() for line in lines] 
        if dataset == 'dips_train_0.4':
            self.data_dir = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/dips/pt_files"
            self.data_list = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/dips_equidock/train_list_0.4.txt" 
            with open(self.data_list, 'r') as f:
                lines = f.readlines()
            self.file_list = [line.strip() for line in lines] 

        if dataset == 'dips_val_0.4':
            self.data_dir = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/dips/pt_files"
            self.data_list = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/dips_equidock/val_list_0.4.txt" 
            with open(self.data_list, 'r') as f:
                lines = f.readlines()
            self.file_list = [line.strip() for line in lines] 

        if dataset == 'dips_train':
            self.data_dir = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/dips/pt_files"
            self.data_list = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/dips_equidock/train_list_lt_50.txt" 
            with open(self.data_list, 'r') as f:
                lines = f.readlines()
            self.file_list = [line.strip() for line in lines] 

        if dataset == 'dips_val':
            self.data_dir = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/dips/pt_files"
            self.data_list = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/dips_equidock/val_list_lt_50.txt" 
            with open(self.data_list, 'r') as f:
                lines = f.readlines()
            self.file_list = [line.strip() for line in lines] 

        if dataset == 'dips_train_500':
            self.data_dir = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/dips/pt_files"
            self.data_list = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/dips_equidock/train_list_500.txt" 
            with open(self.data_list, 'r') as f:
                lines = f.readlines()
            self.file_list = [line.strip() for line in lines] 

        if dataset == 'dips_val_500':
            self.data_dir = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/dips/pt_files"
            self.data_list = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/dips_equidock/val_list_500.txt" 
            with open(self.data_list, 'r') as f:
                lines = f.readlines()
            self.file_list = [line.strip() for line in lines] 

        if dataset == 'dips_test_500':
            self.data_dir = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/dips/pt_files"
            self.data_list = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/dips_equidock/test_list_500.txt" 
            with open(self.data_list, 'r') as f:
                lines = f.readlines()
            self.file_list = [line.strip() for line in lines] 
        
        elif dataset == 'dips_test':
            self.data_dir = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/pts/dips_test"
            self.file_list = [i[:-3] for i in os.listdir(self.data_dir)] 

        elif dataset == 'db5_train_bound':
            self.data_dir = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/pts/db5_bound"
            self.data_list = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/db5.5/train_list.txt"
            with open(self.data_list, 'r') as f:
                lines = f.readlines()
            self.file_list = [line.strip() for line in lines] 

        elif dataset == 'db5_val_bound':
            self.data_dir = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/pts/db5_bound"
            self.data_list = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/db5.5/val_list.txt"
            with open(self.data_list, 'r') as f:
                lines = f.readlines()
            self.file_list = [line.strip() for line in lines] 

        elif dataset == 'db5_test_bound':
            self.data_dir = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/pts/db5_bound"
            self.data_list = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/db5.5/bound_list.txt"
            with open(self.data_list, 'r') as f:
                lines = f.readlines()
            self.file_list = [line.strip() for line in lines] 

        elif dataset == 'db5_train_unbound':
            self.data_dir = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/pts/db5_unbound"
            self.data_list = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/db5.5/train_list.txt"
            with open(self.data_list, 'r') as f:
                lines = f.readlines()
            self.file_list = [line.strip() for line in lines] 

        elif dataset == 'db5_val_unbound':
            self.data_dir = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/pts/db5_unbound"
            self.data_list = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/db5.5/val_list.txt"
            with open(self.data_list, 'r') as f:
                lines = f.readlines()
            self.file_list = [line.strip() for line in lines] 

        elif dataset == 'db5_test_flexible_unbound':
            self.data_dir = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/pts/db5_unbound_flexible"
            self.data_list = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/db5.5/flexible_list.txt"
            with open(self.data_list, 'r') as f:
                lines = f.readlines()
            self.file_list = [line.strip() for line in lines] 

        elif dataset == 'db5_test_flexible_bound':
            self.data_dir = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/pts/db5_bound_flexible"
            self.data_list = "/home/lchu11/scr4_jgray21/lchu11/my_repos/Docking-dev/data/db5.5/unbound_list.txt"
            with open(self.data_list, 'r') as f:
                lines = f.readlines()
            self.file_list = [line.strip() for line in lines] 


    def __getitem__(self, idx: int):
        if self.dataset[:4] == 'dips' and self.dataset != 'dips_test':
            # Get info from file_list 
            _id = self.file_list[idx]
            split_string = _id.split('/')
            _id = split_string[0] + '_' + split_string[1].rsplit('.', 1)[0]
            data = torch.load(os.path.join(self.data_dir, _id+'.pt'))

        else:
            data = torch.load(os.path.join(self.data_dir, self.file_list[idx]+'.pt'))

        # get from data
        _id = data.name
        seq1 = data['receptor'].seq
        coords1 = data['receptor'].pos
        protein1_embeddings = data['receptor'].x
        seq2 = data['ligand'].seq
        coords2 = data['ligand'].pos
        protein2_embeddings = data['ligand'].x

        # crop > 500
        if not self.is_testing:
            crop_size = 500
            if len(seq1) + len(seq2) > crop_size:
                crop_size_per_chain = crop_size // 2
                if len(seq1) > len(seq2):
                    if len(seq2) < crop_size_per_chain:
                        crop_len = crop_size - len(seq2)
                        n = random.randint(0, len(seq1)-crop_len)
                        seq1 = seq1[n:n+crop_len]
                        coords1 = coords1[n:n+crop_len]
                        protein1_embeddings = protein1_embeddings[n:n+crop_len]
                    else:
                        n1 = random.randint(0, len(seq1)-crop_size_per_chain)
                        seq1 = seq1[n1:n1+crop_size_per_chain]
                        coords1 = coords1[n1:n1+crop_size_per_chain]
                        protein1_embeddings = protein1_embeddings[n1:n1+crop_size_per_chain]
                        n2 = random.randint(0, len(seq2)-crop_size_per_chain)
                        seq2 = seq2[n2:n2+crop_size_per_chain]
                        coords2 = coords2[n2:n2+crop_size_per_chain]
                        protein2_embeddings = protein2_embeddings[n2:n2+crop_size_per_chain]

                elif len(seq2) > len(seq1):
                    if len(seq1) < crop_size_per_chain:
                        crop_len = crop_size - len(seq1)
                        n = random.randint(0, len(seq2)-crop_len)
                        seq2 = seq2[n:n+crop_len]
                        coords2 = coords2[n:n+crop_len]
                        protein2_embeddings = protein2_embeddings[n:n+crop_len]
                    else:
                        n1 = random.randint(0, len(seq1)-crop_size_per_chain)
                        seq1 = seq1[n1:n1+crop_size_per_chain]
                        coords1 = coords1[n1:n1+crop_size_per_chain]
                        protein1_embeddings = protein1_embeddings[n1:n1+crop_size_per_chain]
                        n2 = random.randint(0, len(seq2)-crop_size_per_chain)
                        seq2 = seq2[n2:n2+crop_size_per_chain]
                        coords2 = coords2[n2:n2+crop_size_per_chain]
                        protein2_embeddings = protein2_embeddings[n2:n2+crop_size_per_chain]
                else:
                    n = random.randint(0, len(seq1)-crop_size_per_chain)
                    seq1 = seq1[n:n+crop_size_per_chain]
                    coords1 = coords1[n:n+crop_size_per_chain]
                    protein1_embeddings = protein1_embeddings[n:n+crop_size_per_chain]
                    seq2 = seq2[n:n+crop_size_per_chain]
                    coords2 = coords2[n:n+crop_size_per_chain]
                    protein2_embeddings = protein2_embeddings[n:n+crop_size_per_chain]

        try:
            assert len(seq1) == coords1.size(0) == protein1_embeddings.size(0)
            assert len(seq2) == coords2.size(0) == protein2_embeddings.size(0) 
        except:
            print(_id)
        
        # Get ground truth
        label_coords = torch.cat([coords1, coords2], dim=0)
        label_rotat = self.get_rotat(label_coords)
        label_trans = self.get_trans(label_coords)

        # Pair embedding
        input_pairs = self.get_pair_mats(label_coords, len(seq1))
        
        # Contact embedding
        if self.is_training:
            input_contact = self.get_pair_contact(label_coords, len(seq1), count=random.randint(0, 3))
        else:
            input_contact = self.get_pair_contact(label_coords, len(seq1), count=self.count)
            
        pair_embeddings = torch.cat([input_pairs, input_contact], dim=-1)

        # Pair positional embedding
        positional_embeddings = self.get_pair_relpos(len(seq1), len(seq2))

        try:
            assert positional_embeddings.size(0) == pair_embeddings.size(0)
            assert positional_embeddings.size(1) == pair_embeddings.size(1)
        except:
            print(_id)

        if self.out_pdb:
            print(_id)
            test_coords = self.get_full_coords(label_coords)
            out_file = _id + '.pdb'
            if os.path.exists(out_file):
                os.remove(out_file)
                print(f"File '{out_file}' deleted successfully.")
            else:
                print(f"File '{out_file}' does not exist.") 
            save_PDB(out_pdb=out_file, coords=test_coords, seq=seq1+seq2, delim=len(seq1)-1)

        # Output
        output = {
            'id': _id,
            'seq1': seq1,
            'seq2': seq2,
            'protein1_embeddings': protein1_embeddings, 
            'protein2_embeddings': protein2_embeddings, 
            'pair_embeddings': pair_embeddings,
            'positional_embeddings': positional_embeddings,
            'label_rotat': label_rotat,
            'label_trans': label_trans,
            'label_coords': label_coords,
        }
        
        return {key: value for key, value in output.items()}

    def __len__(self):
        return len(self.file_list)
    
    def get_rotat(self, coords):
        # Get backbone coordinates. 
        n_coords = coords[:, 0, :]
        ca_coords = coords[:, 1, :]
        c_coords = coords[:, 2, :]

        # Gram-Schmidt process.
        v1 = c_coords - ca_coords 
        v2 = n_coords - ca_coords
        e1 = F.normalize(v1) 
        u2 = v2 - e1 * (torch.einsum('b i, b i -> b', e1, v2).unsqueeze(-1))
        e2 = F.normalize(u2) 
        e3 = torch.cross(e1, e2, dim=-1)

        # Get rotations.
        rotations=torch.stack([e1, e2, e3], dim=-1)
        return rotations

    def get_trans(self, coords):
        return coords[:, 1, :]

    def get_pair_relpos(self, rec_len, lig_len):
        rmax = 32
        rec = torch.arange(0, rec_len)
        lig = torch.arange(0, lig_len) 
        total = torch.cat([rec, lig], dim=0)
        pairs = total[None, :] - total[:, None]
        pairs = torch.clamp(pairs, min=-rmax, max=rmax)
        pairs = pairs + rmax 
        pairs[:rec_len, rec_len:] = 2*rmax + 1
        pairs[rec_len:, :rec_len] = 2*rmax + 1 
        relpos = F.one_hot(pairs, num_classes=2*rmax+2).float()
        total_len = rec_len + lig_len
        chain_row = torch.cat([torch.zeros(rec_len, total_len), 
                               torch.ones(lig_len, total_len)], dim=0)  
        chain_col = torch.cat([torch.zeros(total_len, rec_len), 
                               torch.ones(total_len, lig_len)], dim=1)
        chains = F.one_hot((chain_row - chain_col + 1).long(), num_classes=3).float()

        pair_pos = torch.cat([relpos, chains], dim=-1)
        return pair_pos
    
    def get_pair_contact(self, coords, n, prob=None, count=None):
        assert (prob is None) ^ (count is None)
        
        if self.use_Cb:
            coords = self.get_full_coords(coords)[:, -1, :]
        else:
            coords = coords[:, 1, :]

        d = torch.norm(coords[:, None, :] - coords[None, :, :], dim=2)
        cutoff = 10.0
        mask = d <= cutoff
        
        mask[:n, :n] = False
        mask[n:, n:] = False

        rec = mask[:n, :n]
        lig = mask[n:, n:]
        inter = mask[:n, n:]

        if prob is not None:        
            random_tensor = torch.rand_like(inter, dtype=torch.float)
            inter = inter & (random_tensor > prob)
        
        elif count is not None:
            # get the indices of all the True values in the tensor
            true_indices = torch.nonzero(inter, as_tuple=False)

            # shuffle the indices
            shuffled_indices = torch.randperm(true_indices.shape[0])

            # make sure count <= # of true indices
            if count > true_indices.shape[0]:
                count = true_indices.shape[0]

            # pick the first shuffled index
            selected_indices = true_indices[shuffled_indices[:count]]

            # create a new tensor of the same shape as the original tensor
            mask = torch.zeros_like(inter)
            
            # set the randomly selected indices to True
            for i in range(selected_indices.shape[0]):
                mask[selected_indices[i, 0], selected_indices[i, 1]] = True

            inter = inter * mask

        upper = torch.cat([rec, inter], dim=1)
        lower = torch.cat([inter.T, lig], dim=1)
        contact = torch.cat([upper, lower], dim=0)
        
        return contact.unsqueeze(-1)

    def get_pair_dist(self, coords, n):
        num_bins = 16
        distogram = self.distogram(
            coords,
            2.0,
            22.0,
            num_bins,
        )

        distogram[:n, n:] = num_bins - 1
        distogram[n:, :n] = num_bins - 1

        # to onehot
        dist = F.one_hot(distogram, num_classes=num_bins).float() 
        
        # test
        if self.out_png:
            data = distogram.numpy()
            plt.imshow(data, cmap='hot', interpolation='nearest')
            plt.colorbar()

            # Save the plot as a PNG file
            plt.savefig('dist.png', dpi=300)
            plt.clf()

        return dist
    
    def get_pair_mats(self, coords, n):
        dist, omega, theta, phi = get_coords6d(coords, use_Cb=self.use_Cb)

        mask = dist < 22.0
        
        num_bins = 16
        dist_bin = self.get_bins(dist, 2.0, 22.0, num_bins)
        omega_bin = self.get_bins(omega, -180.0, 180.0, num_bins)
        theta_bin = self.get_bins(theta, -180.0, 180.0, num_bins)
        phi_bin = self.get_bins(phi, -180.0, 180.0, num_bins)

        def mask_mat(mat):
            mat[~mask] = num_bins - 1
            mat.fill_diagonal_(num_bins - 1)
            mat[:n, n:] = num_bins - 1
            mat[n:, :n] = num_bins - 1
            return mat

        dist_bin[:n, n:] = num_bins - 1
        dist_bin[n:, :n] = num_bins - 1
        omega_bin = mask_mat(omega_bin)
        theta_bin = mask_mat(theta_bin)
        phi_bin = mask_mat(phi_bin)

        # to onehot
        dist = F.one_hot(dist_bin, num_classes=num_bins).float() 
        omega = F.one_hot(omega_bin, num_classes=num_bins).float() 
        theta = F.one_hot(theta_bin, num_classes=num_bins).float() 
        phi = F.one_hot(phi_bin, num_classes=num_bins).float() 
        
        # test
        if self.out_png:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(nrows=2, ncols=2)
            ax[0, 0].imshow(dist_bin.numpy(), cmap='hot', interpolation='nearest')
            ax[0, 1].imshow(omega_bin.numpy(), cmap='hot', interpolation='nearest')
            ax[1, 0].imshow(theta_bin.numpy(), cmap='hot', interpolation='nearest')
            ax[1, 1].imshow(phi_bin.numpy(), cmap='hot', interpolation='nearest')
            
            # Set titles for each plot
            ax[0, 0].set_title('dist')
            ax[0, 1].set_title('omega')
            ax[1, 0].set_title('theta')
            ax[1, 1].set_title('phi')

            # Save the plot as a PNG file
            plt.savefig('dist_orient.png', dpi=300)
            plt.clf()

        return torch.cat([dist, omega, theta, phi], dim=-1)

    def get_full_coords(self, coords):
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

    def distogram(self, coords, min_bin, max_bin, num_bins, use_cb=False):
        # Coords are [... L x 3 x 3], where it's [N, CA, C] x 3 coordinates.
        boundaries = torch.linspace(
            min_bin,
            max_bin,
            num_bins - 1,
            device=coords.device,
        )
        boundaries = boundaries**2
        N, CA, C = [x.squeeze(-2) for x in coords.chunk(3, dim=-2)]

        if use_cb:
            # Infer CB coordinates.
            b = CA - N
            c = C - CA
            a = b.cross(c, dim=-1)
            CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
            dists = (CB[..., None, :, :] - CB[..., :, None, :]).pow(2).sum(dim=-1, keepdims=True)
        else:
            dists = (CA[..., None, :, :] - CA[..., :, None, :]).pow(2).sum(dim=-1, keepdims=True)

        bins = torch.sum(dists > boundaries, dim=-1)  # [..., L, L]
        return bins

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


if __name__ == '__main__':
    dataset = GeoDockDataset(
        dataset='db5_train_bound',
        out_pdb=True,
        out_png=False,
        is_training=False,
        count=0,
    )

    dataset[0]

    """
    dataloader = data.DataLoader(dataset, batch_size=1, num_workers=6)

    for batch in tqdm(dataloader):
        pass
    """
    
