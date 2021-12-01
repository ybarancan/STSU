# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from src.detr.util.misc import NestedTensor
import logging

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, split=False,temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        
        self.split = split
        if self.split:
            self.num_pos_feats = self.num_pos_feats/2
        
        self.x_limit = 40
        self.y_limit = 50
        
        self.img_size = [50,28]

    def forward(self,x,calib=None, bev=False, abs_bev=True):
       
        if bev:
            range_x = torch.arange(self.img_size[0]).cuda()
            range_y = torch.arange(self.img_size[1]).cuda()
            cur_y, cur_x = torch.meshgrid(range_y,range_x)
            cam_height = 1.7
         
            f = calib[0,0]
            y_center = calib[1,-1]
            y_embed = cam_height*f/(cur_y - y_center + 0.1)
            x_embed = (y_embed*cur_x - calib[0,-1]*y_embed)/f
            
            to_remove = (y_embed < 0) | (y_embed > self.y_limit)
            
            x_embed[y_embed < 0] = 0
            x_embed[y_embed > self.y_limit] = 0
            
            y_embed[y_embed < 0] = 0
            y_embed[y_embed > self.y_limit] = 0
         
            if abs_bev:
                
                                
                y_embed = y_embed.clamp(-self.y_limit,self.y_limit)/self.y_limit + 2
                x_embed = x_embed.clamp(-self.x_limit,self.x_limit)/self.x_limit + 2
                
                x_embed[to_remove] = 1
                x_embed[to_remove] = 1
                
                y_embed[to_remove] = 1
                y_embed[to_remove] = 1
                
                y_embed = torch.flip(y_embed,dims=[0])
          
                    
                x_embed = torch.log(x_embed)
                
                y_embed = torch.log(y_embed)
                
                y_embed = y_embed.unsqueeze(0).cumsum(1, dtype=torch.float32) 
                x_embed = x_embed.unsqueeze(0).cumsum(2, dtype=torch.float32)
                
                eps = 1e-6
                y_embed = torch.flip(y_embed,dims=[1])
                y_embed = y_embed / (y_embed[:,:1, :] + eps) 
                x_embed = x_embed / (x_embed[:,:, -1:] + eps) 
                
                x_embed[0,to_remove] = 1
                y_embed[0,to_remove] = 1
                
                x_embed = x_embed * self.scale
                y_embed = y_embed * self.scale

            else:
                                
                y_embed = y_embed.clamp(-self.y_limit,self.y_limit)/self.y_limit + 1
                x_embed = x_embed.clamp(-self.x_limit,self.x_limit)/self.x_limit + 1
                
                x_embed[to_remove] = 0
                x_embed[to_remove] = 0
                
                y_embed[to_remove] = 0
                y_embed[to_remove] = 0
                
                y_embed = torch.flip(y_embed,dims=[0])
                
     
                
                
                y_embed = y_embed.unsqueeze(0).cumsum(1, dtype=torch.float32) 
                x_embed = x_embed.unsqueeze(0).cumsum(2, dtype=torch.float32)
                
                eps = 1e-6
                y_embed = torch.flip(y_embed,dims=[1])
                y_embed = y_embed / (y_embed[:,:1, :] + eps) 
                x_embed = x_embed / (x_embed[:,:, -1:] + eps) 
                
                x_embed[0,to_remove] = 1
                y_embed[0,to_remove] = 1
                
                x_embed = x_embed * self.scale
                y_embed = y_embed * self.scale
    
        else:
            not_mask = torch.ones_like(x)
            not_mask = not_mask[:,0,...]
            y_embed = not_mask.cumsum(1, dtype=torch.float32)
            x_embed = not_mask.cumsum(2, dtype=torch.float32)
                
            if self.normalize:
                eps = 1e-6
                y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
                x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale



        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, split=args.split_pe, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
