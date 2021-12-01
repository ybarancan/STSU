# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np

import logging

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1,cost_visible: float = 1,cost_end: float = 1,
                 cost_obj_class=1, cost_obj_center=1, cost_obj_len=1, cost_obj_orient=1, polyline=False):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        
        self.cost_obj_class = cost_obj_class
        self.cost_obj_center = cost_obj_center
       
        self.cost_obj_len = cost_obj_len
        self.cost_obj_orient = cost_obj_orient
        
        
        self.cost_end = cost_end
        self.cost_giou = cost_giou
        self.cost_visible = cost_visible
        
        self.polyline = polyline
        
        
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, do_obj=True,val=False, thresh=0.5, pinet=False, only_objects=False):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        
        
        '''
        FIRST DEAL WITH ROADS
        '''
        if self.polyline:
            bs, num_queries = outputs["init_point_detection_softmaxed"].shape[:2]

        # We flatten to compute the cost matrices in a batch
            out_prob = outputs["init_point_detection_softmaxed"].flatten(0, 1)  # [batch_size * num_queries, num_classes]
            # out_bbox = outputs["pred_init_point_softmaxed"] # [batch_size * num_queries, 4]
       
    
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            
            est_loc = outputs['pred_init_point_softmaxed'].flatten(1)
        
            # logging.error('LOGITS ' + str(logits.shape))
            
            gt_centers = targets[0]['init_point_matrix']
            if gt_centers.size(0) > 30:
                gt_centers = gt_centers[:30]
            
            gt_centers = gt_centers.flatten(1)
            gt_center_locs = torch.argmax(gt_centers,dim=-1)
            
            est_dist = est_loc[:,gt_center_locs]
            
       
            # Final cost matrix
            # C = -self.cost_bbox * est_dist - self.cost_class * prob_dist
            C = -self.cost_bbox * est_dist 
            C =C.cpu()
    
            static_to_return = linear_sum_assignment(C)

            return static_to_return
        
        if pinet:
            tgt_bbox = torch.cat([v["control_points"] for v in targets])
            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(outputs, tgt_bbox, p=1)
       
            
       
            # Final cost matrix
            C = cost_bbox
            C = C.view(1, outputs.shape[0], -1).cpu()
    
            sizes = [len(v["control_points"]) for v in targets]
            static_indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            static_to_return = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in static_indices]
        
            return static_to_return
        
        if only_objects:
            if targets[0]['obj_exists']:
                bs, num_queries = outputs["obj_logits"].shape[:2]
    
                # We flatten to compute the cost matrices in a batch
                out_prob = outputs["obj_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
                out_bbox = outputs["obj_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
                
                # Also concat the target labels and boxes
                tgt_ids = torch.cat([v["obj_corners"][:,-1] for v in targets]).long()
                tgt_bbox = torch.cat([v['obj_converted'] for v in targets])
                
                 
                
                out_center = out_bbox[:,:2]
                out_lengths = out_bbox[:,2:4]
                out_angle = out_bbox[:,-1:]
                
                out_cos2 = torch.cos(2*out_angle)
                out_sin2 = torch.sin(2*out_angle)
                
                '''
                '''
                
                gt_center = tgt_bbox[:,:2]
                gt_lengths = tgt_bbox[:,2:4]
                gt_angle = tgt_bbox[:,-1:]
                
                gt_cos2 = torch.cos(2*gt_angle)
                gt_sin2 = torch.sin(2*gt_angle)
                
                
                # The 1 is a constant that doesn't change the matching, it can be ommitted.
                cost_class = -out_prob[:, tgt_ids]
        
               
                cost_center = torch.cdist(out_center, gt_center, p=1)
                
                cost_len = torch.cdist(out_lengths, gt_lengths, p=1)
                
                cost_orient1 = torch.cdist(out_cos2, gt_cos2, p=1)
                cost_orient2 = torch.cdist(out_sin2, gt_sin2, p=1)
                
                
                cost_orient = cost_orient1 + cost_orient2
                
                
        
                # Final cost matrix
               
                C = self.cost_obj_center * cost_center + self.cost_obj_len * cost_len + self.cost_obj_orient * cost_orient + self.cost_obj_class * cost_class 
                    
                C = C.view(bs, num_queries, -1).cpu()
        
                sizes = [len(v["obj_converted"]) for v in targets]
                obj_indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
                obj_to_return = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in obj_indices]
                
                # logging.error('OBJ TO RETURN')
                # logging.error(str(obj_to_return))
                
                '''
                '''
            else:
                obj_to_return = [0,0]
                
            return None, obj_to_return
                
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
      
        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["control_points"] for v in targets])
        # tgt_end = torch.cat([v["endpoints"] for v in targets])

    
        if val:
            cost_class = 5*(out_prob[:, tgt_ids] < thresh)
    
            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
       
            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class 
            C = C.view(bs, num_queries, -1).cpu()
    
            sizes = [len(v["control_points"]) for v in targets]
            static_indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            static_to_return = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in static_indices]
        
        
        else:
            
            cost_class = -out_prob[:, tgt_ids]
    
            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
       
            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class 
            C = C.view(bs, num_queries, -1).cpu()
    
            sizes = [len(v["control_points"]) for v in targets]
            static_indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            static_to_return = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in static_indices]
                
        '''
        NOW WITH OBJECTS
        '''
        
        if do_obj:
            
            if targets[0]['obj_exists']:
                bs, num_queries = outputs["obj_logits"].shape[:2]
    
                # We flatten to compute the cost matrices in a batch
                out_prob = outputs["obj_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
                out_bbox = outputs["obj_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
                
                # Also concat the target labels and boxes
                tgt_ids = torch.cat([v["obj_corners"][:,-1] for v in targets]).long()
                tgt_bbox = torch.cat([v['obj_converted'] for v in targets])
                
                    
                
                out_center = out_bbox[:,:2]
                out_lengths = out_bbox[:,2:4]
                out_angle = out_bbox[:,-1:]
                
                out_cos2 = torch.cos(2*out_angle)
                out_sin2 = torch.sin(2*out_angle)
                
                '''
                '''
                
                gt_center = tgt_bbox[:,:2]
                gt_lengths = tgt_bbox[:,2:4]
                gt_angle = tgt_bbox[:,-1:]
                
                gt_cos2 = torch.cos(2*gt_angle)
                gt_sin2 = torch.sin(2*gt_angle)
                
                
                cost_class = -out_prob[:, tgt_ids]
        
                cost_center = torch.cdist(out_center, gt_center, p=1)
                
                cost_len = torch.cdist(out_lengths, gt_lengths, p=1)
                
                cost_orient1 = torch.cdist(out_cos2, gt_cos2, p=1)
                cost_orient2 = torch.cdist(out_sin2, gt_sin2, p=1)
                
                
                cost_orient = cost_orient1 + cost_orient2
                
                
                C = self.cost_obj_center * cost_center + self.cost_obj_len * cost_len + self.cost_obj_orient * cost_orient + self.cost_obj_class * cost_class 
                    
                C = C.view(bs, num_queries, -1).cpu()
        
                sizes = [len(v["obj_converted"]) for v in targets]
                obj_indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
                obj_to_return = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in obj_indices]
                
          
                '''
                '''
            else:
                obj_to_return = [0,0]
        
            
                
            return static_to_return, obj_to_return
           
        else:
             return static_to_return, None

                
        
def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox,cost_end=args.set_cost_end, cost_giou=args.set_cost_giou,
                            cost_obj_class=args.set_obj_cost_class, cost_obj_center=args.set_obj_cost_center, cost_obj_len=args.set_obj_cost_len, 
                            cost_obj_orient=args.set_obj_cost_orient)
    
def build_polyline_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox,cost_end=args.set_cost_end, cost_giou=args.set_cost_giou,
                            cost_obj_class=args.set_obj_cost_class, cost_obj_center=args.set_obj_cost_center, cost_obj_len=args.set_obj_cost_len, 
                            cost_obj_orient=args.set_obj_cost_orient, polyline=True)
