import numpy as np
import torch
import math
import matplotlib.pyplot as plt


class GridCellCoding(torch.nn.Module):
    """
    Torch layer that takes x and y as global coordinates and returns activations of grid cells- Was not used for the final thesis
    """

    def __init__(self, number_frequencies=9, number_cells_frequency=100, device=None):
        super().__init__()
        self.number_frequencies = number_frequencies
        self.number_cells_frequency = number_cells_frequency
        self.b0 = torch.ones(2).to(device)
        self.b0[1] = 0
        # self.b0 = self.b0.view(1, 2)
        self.b1 = torch.ones(2).to(device)
        self.b1[0] = math.cos(math.pi / 3)
        self.b1[1] = math.sin(math.pi / 3)
        self.b2 = torch.ones(2).to(device)
        self.b2[0] = math.cos(2 * math.pi / 3)
        self.b2[1] = math.sin(2 * math.pi / 3)
        self.offsets = torch.rand((number_frequencies, number_cells_frequency, 2)).to(device)
        self.offsets[:, :, 0] *= 1 + math.cos(60/180 * math.pi)
        self.offsets[:, :, 1] *= math.sin(60 / 180 * math.pi)
        self.frequencies = torch.zeros(number_frequencies).to(device)
        for i in range(number_frequencies):
            frequency = 0.5 * 2 * math.pi * math.pow(math.sqrt(2), i)
            self.frequencies[i] = frequency
            self.offsets[i, :, :] *= math.pi
        self.frequencies = self.frequencies.repeat(number_cells_frequency, 1).T.view(1, self.number_frequencies, self.number_cells_frequency, 1)
        self.device = device



    def forward(self, x):
        pre_activations = self.frequencies @ x.reshape(x.shape[0], 1, 1, x.shape[1]).float()
        pre_activations = pre_activations + self.offsets
        z_0 = torch.tensordot(self.b0, pre_activations, dims=([0], [3]))
        z_1 = torch.tensordot(self.b1, pre_activations, dims=([0], [3]))
        z_2 = torch.tensordot(self.b2, pre_activations, dims=([0], [3]))
        activations = torch.maximum(torch.zeros((self.number_frequencies, self.number_cells_frequency)).to(self.device), (torch.cos(z_0) + torch.cos(z_1) + torch.cos(z_2)) / 3)
        activations = activations.flatten(start_dim=1)
        return activations

# # Code to visualize the grid cells
# number_frequencies = 6
# number_cells = 50
# grid_cells = GridCellCoding(number_frequencies=number_frequencies, number_cells_frequency=number_cells)
# image = np.zeros((number_frequencies, number_cells, 256, 256))
# images = []
# for x in range(256):
#     for y in range(256):
#         responses_cells = grid_cells(torch.from_numpy(np.array([x/128 - 1, y/128 - 1], dtype=float)).view(1, 2).repeat(32, 1))
#         image[:,:,x,y] = responses_cells[0].reshape(number_frequencies,number_cells)
# for i in range(number_frequencies):
#     for j in range(number_cells):
#         images.append(image[i, j])
# fig, _ = plot.display(images, lims=None, shape=(number_cells, number_frequencies), figsize=(number_cells*2, number_frequencies*2), axes_visible=False, layout='tight')
# plot.save_fig(fig, f'EmergentPredictiveCoding/Results/Fig2_mscoco/grid_cells', bbox_inches='tight')
