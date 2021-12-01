import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
#from .. import utils

class Resampler(nn.Module):

    def __init__(self, resolution, extents):
        super().__init__()

        # Store z positions of the near and far planes
        self.near = extents[1]
        self.far = extents[3]

        # Make a grid in the x-z plane
        self.grid = _make_grid(resolution, extents)


    def forward(self, features, calib):

        # Copy grid to the correct device
        self.grid = self.grid.to(features)
        
      
        calib = calib.view(-1, 1, 1, 3, 3)

        # Transform grid center locations into image u-coordinates
        cam_coords = torch.matmul(calib, self.grid.unsqueeze(-1)).squeeze(-1)

        # Apply perspective projection and normalize
        ucoords = cam_coords[..., 0] / cam_coords[..., -1]
        # logging.error('U COORDS ' + str(torch.min(ucoords)) +', '+ str(torch.max(ucoords)))
        
        
        
        ucoords = ucoords / features.size(-1) * 2 -1
        # logging.error('U COORDS ' + str(torch.min(ucoords)) +', '+ str(torch.max(ucoords)))
        
        # Normalize z coordinates
        zcoords = cam_coords[..., 1] / cam_coords[..., -1]
        
        # zcoords = zcoords + to_add
        zcoords = zcoords/ features.size(-2) * 2 - 1
        
        
        # Resample 3D feature map
        grid_coords = torch.stack([ucoords, zcoords], -1).clamp(-1.1, 1.1)
        return F.grid_sample(features, grid_coords)


def _make_grid(resolution, extents):

    # Create a grid of cooridinates in the birds-eye-view
    x1, z1, x2, z2 = extents
    z1 = z1 + 2
    z2 = z2 + 2
    # (z2 + z1) - 
    zz, xx = torch.meshgrid(
        (z2 + z1) -torch.arange(z1, z2, resolution), torch.arange(x1 , x2, resolution))
    
    
    return torch.stack([xx, 1.7*torch.ones_like(xx), zz], dim=-1)