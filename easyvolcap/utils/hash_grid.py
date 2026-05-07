import torch


class GaussianHashGrid:
    def __init__(self, min_cell_size, max_cell_size, num_levels=1, hashmap_size=2**23):
        self.hashmap_size = hashmap_size
        self.num_levels = num_levels


        self.cell_sizes = min_cell_size * (max_cell_size / min_cell_size) ** (
            torch.linspace(0, 1, num_levels, dtype=torch.float32)
        )


        self.primes = torch.tensor([1, 2654435761, 805459861], dtype=torch.long)

        self.levels_data = []
        self.sorted_indices = None
        self.flat_starts = None
        self.flat_ends = None
        self.flat_densities = None

    def _get_hash(self, points, cell_size):

        primes = self.primes.to(points.device)

        if not torch.is_tensor(cell_size):
            cell_size = torch.tensor(cell_size, dtype=points.dtype, device=points.device)
        else:
            cell_size = cell_size.to(device=points.device, dtype=points.dtype)

        grid_coords = torch.floor(points / cell_size).long()
        h = (grid_coords * primes).sum(dim=-1)
        return h % self.hashmap_size

    def update_grid(self, pcd):

        device = pcd.device

        self.levels_data = []


        cell_sizes = self.cell_sizes.to(device=device, dtype=pcd.dtype)

        for l in range(self.num_levels):
            cell_size = cell_sizes[l]

            h_pcd = self._get_hash(pcd, cell_size)

            self.sorted_indices = torch.argsort(h_pcd)
            sorted_hashes = h_pcd[self.sorted_indices]

            all_hashes = torch.arange(self.hashmap_size, device=device, dtype=sorted_hashes.dtype)

            starts = torch.searchsorted(sorted_hashes, all_hashes, side='left')
            ends = torch.searchsorted(sorted_hashes, all_hashes, side='right')

            self.levels_data.append({
                'starts': starts,
                'ends': ends,
                'grid_density': ends - starts,
                'cell_size': cell_size
            })

        starts_list = [d['starts'] for d in self.levels_data]
        ends_list = [d['ends'] for d in self.levels_data]
        densities_list = [d['grid_density'] for d in self.levels_data]

        self.flat_starts = torch.stack(starts_list).reshape(-1).float().to(device)
        self.flat_ends = torch.stack(ends_list).reshape(-1).float().to(device)
        self.flat_densities = torch.stack(densities_list).reshape(-1).float().to(device)

        return