# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F

from torch import nn
import numpy as np

from src.detr.deeplab_backbone import build_backbone
from src.detr.matcher import build_matcher

from src.detr.transformer import build_transformer

from ..nn.resampler import Resampler
from src.utils import bezier
import logging





class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride==1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
 
        if self.downsample is not None:
            x = self.downsample(x)
         
        return x + r 
    
    
class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        m = self.ResMM(m)
        return m
    
class Decoder(nn.Module):
    def __init__(self, indim, mdim, n_classes):
        super(Decoder, self).__init__()
        self.convFM = nn.Conv2d(indim, mdim, kernel_size=(3,3), padding=(1,1), stride=1)
        
        self.conv_inter = nn.Conv2d(258, mdim, kernel_size=(3,3), padding=(1,1), stride=1)
        
        self.ResMM1 = ResBlock(mdim, mdim)
        
        self.ResMM2 = ResBlock(mdim, mdim)
        
        self.Res_side = ResBlock(mdim, mdim)
        
        self.Res_tot = ResBlock(mdim, mdim)
        # self.RF3 = Refine(512, mdim) # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim) # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, n_classes+1, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, r4,  r2):
        m4 = self.ResMM2(self.ResMM1(self.convFM(r4)))
      
        r2 = self.Res_side(self.conv_inter(r2))
        
        m2 = self.Res_tot(self.RF2(r2, m4)) # out: 1/4, 256

    
        
        
        p2 = self.pred2(F.relu(m2))
        
        p2 = F.interpolate(p2, scale_factor=2, mode='bilinear', align_corners=False)

        return p2 #, p2, p3, p4

class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, warpers, num_classes, num_queries,args, aux_loss=False,parameter_setting=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        
        self.warper1, self.warper2 = warpers
        
        self.object_refinement = args.object_refinement
        
        
        self.num_control_points = args.num_spline_points
        self.num_coeffs = self.num_control_points*2
        hidden_dim = transformer.d_model
        
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.left_query_embed = nn.Embedding(num_queries, hidden_dim)
        
        self.split_pe = args.split_pe
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
            
        if self.object_refinement:
            self.coord_refiner = Decoder(hidden_dim + args.num_object_classes + 3,256, args.num_object_classes)
        
        self.bev_pe = args.bev_pe
        self.abs_bev = args.abs_bev
        
        self.only_bev = args.only_bev_pe
        
        self.backbone = backbone
        self.aux_loss = aux_loss
        
        
        self.num_object_queries = args.num_object_queries
        
        self.estimate_objects = args.objects
        
        if self.estimate_objects:
            
            self.object_embed = nn.Embedding(args.num_object_queries, hidden_dim)
            self.left_object_embed = nn.Embedding(args.num_object_queries, hidden_dim)
               
            self.object8_class_net = MLP(hidden_dim, 256, args.num_object_classes + 1, 3)
                
            self.object_bbox_net = MLP(hidden_dim, 256, 4, 3)
            self.object_slope_net = MLP(hidden_dim, 256, 1, 3)
        
            
        if parameter_setting is None:
        
            self.class_embed = MLP(hidden_dim, 128, num_classes + 1, 2)
            
            self.spline_embed = MLP(hidden_dim, 128, self.num_coeffs, 2)
            self.endpoint_embed = MLP(hidden_dim, 128, 4, 2)
            self.association_embed_maker = MLP(hidden_dim, 128, 64, 2)
            self.association_classifier = MLP(128, 64, 1, 2)
          
        
        else:
            
            self.class_embed = MLP(hidden_dim, parameter_setting['class_embed_dim'], num_classes + 1, parameter_setting['class_embed_num'])
            
            self.spline_embed = MLP(hidden_dim, parameter_setting['box_embed_dim'], self.num_coeffs, parameter_setting['box_embed_num'])
            self.endpoint_embed = MLP(hidden_dim, parameter_setting['endpoint_embed_dim'], 4, parameter_setting['endpoint_embed_num'])
            self.association_embed_maker = MLP(hidden_dim, parameter_setting['assoc_embed_dim'], parameter_setting['assoc_embed_last_dim'], parameter_setting['assoc_embed_num'])
            self.association_classifier = MLP(2*parameter_setting['assoc_embed_last_dim'], parameter_setting['assoc_classifier_dim'], 1, parameter_setting['assoc_classifier_num'])
            
          
        
    def thresh_and_assoc_estimates(self,  outputs,thresh=0.5):
#        _, idx = self._get_src_permutation_idx(indices)
        
#        _, target_ids = self._get_tgt_permutation_idx(indices)
        
        assoc_features = torch.squeeze(outputs['assoc_features'])
        
       
        out_logits = torch.squeeze(outputs['pred_logits'])
        
        prob = F.softmax(out_logits, -1)
        
        
        
        
        selected_features = assoc_features[prob[:,1] > thresh]
        
        reshaped_features1 = torch.unsqueeze(selected_features,dim=1).repeat(1,selected_features.size(0),1)
        reshaped_features2 = torch.unsqueeze(selected_features,dim=0).repeat(selected_features.size(0),1,1)
        
        total_features = torch.cat([reshaped_features1,reshaped_features2],dim=-1)
        
        est = torch.squeeze(self.association_classifier(total_features).sigmoid(),dim=-1)
        
        outputs['pred_assoc'] = torch.unsqueeze(est,dim=0)
        
   
        return outputs
    
    
    def forward(self, samples,calib=None,left_traffic=False):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
#        if isinstance(samples, (list, torch.Tensor)):
#            samples = nested_tensor_from_tensor_list(samples)
        
        # samples = torch.squeeze(samples,dim=0)
        calib_smallest = calib.clone()
        calib_smallest[:2,:] = calib_smallest[:2,:] / 16
        
        calib_big = calib.clone()
        calib_big[:2,:] = calib_big[:2,:] / 4
        

        features, low_features, pos, bev_pos = self.backbone(samples,calib_smallest, self.abs_bev)
        
        src = features[-1]
        
        
        
        # assert mask is not None
        mask=torch.zeros_like(src)
        mask = mask[:,0,...]
        mask=mask > 4
        
        if left_traffic:
#            logging.error('SELECTED LEFT TRAFFIC')
            selected_embed = self.left_query_embed.weight
            if self.estimate_objects:
                selected_object_embed = self.left_object_embed
        else:
#            logging.error('SELECTED RIGHT TRAFFIC')
            selected_embed = self.query_embed.weight
            if self.estimate_objects:
                selected_object_embed = self.object_embed
            
        if self.estimate_objects:
            selected_embed = torch.cat([selected_embed, selected_object_embed.weight],dim=0)
            
        if self.split_pe:
            hs, trans_memory = self.transformer(self.input_proj(src), mask, selected_embed, torch.cat([pos[-1], bev_pos[-1]],dim=1))
       
        
        elif self.bev_pe:
            if self.only_bev:
                hs, trans_memory = self.transformer(self.input_proj(src), mask, selected_embed,  bev_pos[-1])
            else:
                hs, trans_memory = self.transformer(self.input_proj(src), mask, selected_embed, pos[-1] + bev_pos[-1])
        else:
            hs, trans_memory = self.transformer(self.input_proj(src), mask, selected_embed, pos[-1] )
        

        
        static_hs = hs[:,:,:self.num_queries]
        
        outputs_class = self.class_embed(static_hs)
        outputs_coord = self.spline_embed(static_hs).sigmoid()
        outputs_endpoints = self.endpoint_embed(static_hs).sigmoid()
        
        assoc_features = self.association_embed_maker(static_hs[-1])
        
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_endpoints': outputs_endpoints[-1], 'assoc_features': assoc_features,
               }

        if self.estimate_objects:
        
            object_hs = hs[:,:,self.num_queries:self.num_object_queries + self.num_queries]
            
        
            outputs_obj_class = self.object8_class_net(object_hs)
            outputs_obj_coord = self.object_bbox_net(object_hs).sigmoid()
            
            outputs_obj_slope = self.object_slope_net(object_hs).sigmoid()* np.pi
            
            outputs_obj_coord = torch.cat([outputs_obj_coord, outputs_obj_slope],dim=-1)
            
            out['obj_logits'] = outputs_obj_class[-1]
            out['obj_boxes'] = outputs_obj_coord[-1]
            

    
        if self.object_refinement:
            out['src_features'] = src
            out['low_features'] = low_features[0]
        
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    
    def refine_obj_seg(self, out_raw, out_pp, calib):
        
        src = out_raw['src_features'].detach().data
        low_src = out_raw['low_features'].detach().data
        
    
        to_feed_polygons = out_pp['small_rendered_polygons']
        
        calib_smallest = calib.clone()
        calib_smallest[:2,:] = calib_smallest[:2,:] / 16
        
        calib_big = calib.clone()
        calib_big[:2,:] = calib_big[:2,:] / 4
        
        
        bev_feats = self.warper1(src, torch.unsqueeze(calib_smallest,0))
        mesh_y, mesh_x = torch.meshgrid(torch.linspace(-1,1,49),torch.linspace(-1,1,50))
        
        my_mesh = torch.unsqueeze(torch.stack([mesh_x,mesh_y],dim=0),0).cuda()
        
        bev_feats = torch.cat([bev_feats,my_mesh],dim=1).float()
        # bev_feats = bev_feats.repeat(to_feed_probs.size(0),1,1,1).float()
        
        large_bev_feats = self.warper2(low_src, torch.unsqueeze(calib_big,0))
        
        mesh_y, mesh_x = torch.meshgrid(torch.linspace(-1,1,98),torch.linspace(-1,1,100))
        
        my_mesh = torch.unsqueeze(torch.stack([mesh_x,mesh_y],dim=0),0).cuda()
        
        total_side = torch.cat([large_bev_feats,my_mesh],dim=1).float()
        
        total_in = torch.cat([bev_feats,to_feed_polygons.float()],dim=1).float()
        

        refined_seg = self.coord_refiner(total_in, total_side)
       
        return refined_seg, refined_seg.softmax(1)


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, num_object_classes, matcher, weight_dict, eos_coef,object_eos_coef,
                 losses,assoc_net,  apply_poly_loss, num_coeffs=3):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_object_classes = num_object_classes
        
        self.matcher = matcher
        
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        
        self.apply_poly_loss = apply_poly_loss
        self.object_eos_coef = object_eos_coef
#        
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        empty_weight_visible = torch.ones(2)
        empty_weight_visible[0] = 0.9
        self.register_buffer('empty_weight', empty_weight)
        
    
        object_empty_weight = torch.tensor(np.array([ 3.0,5.0,5.0,5.0,
                                10.0,12.0,12.0,
                                12.0, 1])).float()

        self.register_buffer('object_empty_weight', object_empty_weight)
        self.num_control_points = num_coeffs
        self.bezierA = bezier.bezier_matrix(n_control=self.num_control_points,n_int=50)
        self.bezierA = self.bezierA.cuda()
        self.assoc_net = assoc_net
#       
    def focal_loss(self, outputs, targets, indices, log=True):
        
        alpha = 0.8
        gamma = 2
        epsilon = 0.00001
        beta = 4
        
        idx = self._get_src_permutation_idx(indices)
        
        src_boxes = outputs['pred_boxes'][idx].view(-1,self.num_control_points,2)
        
        inter_points = torch.matmul(self.bezierA.expand(src_boxes.size(0),-1,-1),src_boxes)
        
        estimated_traj = bezier.gaussian_line_from_traj(inter_points)
        
        target_boxes = torch.cat([t['smoothed'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        target_boxes = torch.clamp(target_boxes,0,1)
        
        hard_labels = (target_boxes > 0.7).float()
        
        labels=target_boxes
        y_pred = estimated_traj
        
        
        L=-hard_labels*alpha*torch.pow((1-y_pred),gamma)*torch.log(y_pred + epsilon)-\
          (1-hard_labels)*(1-alpha)*torch.pow(1-labels,beta)*torch.pow(y_pred,gamma)*torch.log(1-y_pred + epsilon)

        
        losses = {}
        losses['focal'] = L.mean()
        
    
        return losses

     
    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        return losses
    
    
    def loss_object_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        
        obj_exists = targets[0]['obj_exists']
        
        if obj_exists:
            
            
            src_logits = outputs['obj_logits']
    
            idx = self._get_src_permutation_idx(indices)
            
            target_classes_o = torch.cat([t["obj_corners"][J][:,-1].long() for t, (_, J) in zip(targets, indices)])
            
            target_classes = torch.full(src_logits.shape[:2], self.num_object_classes,
                                        dtype=torch.int64, device=src_logits.device)
            
            target_classes[idx] = target_classes_o
    
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.object_empty_weight)
            losses = {'loss_obj_ce': loss_ce}

        
        else:
            src_logits = outputs['obj_logits']
    
            
            target_classes = torch.full(src_logits.shape[:2], self.num_object_classes,
                                        dtype=torch.int64, device=src_logits.device)
            
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.object_empty_weight)
            losses = {'loss_obj_ce': loss_ce}


        # if log:
        #     # TODO this should probably be a separate loss, not hacked in this one here
        #     losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses
    
    def loss_assoc(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        
        _, idx = self._get_src_permutation_idx(indices)
        
        _, target_ids = self._get_tgt_permutation_idx(indices)
        
        lab = targets[0]['con_matrix']
        lab = lab.float()
        lab = lab[target_ids,:]
        lab = lab[:,target_ids]
        
        est = outputs['pred_assoc']
        
        mask = lab*3 + (1-lab)
        
        loss_ce = torch.mean(F.binary_cross_entropy_with_logits(est.view(-1),lab.view(-1),weight=mask.float().view(-1)))
        losses = {'loss_assoc': loss_ce}
        

        src_boxes = outputs['pred_boxes'][0][idx]
        src_boxes = src_boxes.view(-1, int(src_boxes.shape[-1]/2), 2)
        
        start_points = src_boxes[:,0,:].contiguous()
        end_points = src_boxes[:,-1,:].contiguous()
        
        my_dist = torch.cdist(end_points, start_points, p=1)
        
        cost_end = 2*my_dist*lab - 3*torch.min(my_dist - 0.05,torch.zeros_like(my_dist).cuda())*(1-lab)
#        losses = {'loss_end_match': cost_end.sum()/(lab.sum() + 0.0001)}
        losses['loss_end_match']= cost_end.mean()

        # if log:
        #     # TODO this should probably be a separate loss, not hacked in this one here
        #     losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses



    def loss_object_boxes(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        
    
        obj_exists = targets[0]['obj_exists']
        
        if obj_exists:
            idx = self._get_src_permutation_idx(indices)
            src_boxes = outputs['obj_boxes'][idx]
#            src_slope = outputs['obj_slope'][idx]
            
            target_boxes = torch.cat([t['obj_converted'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            
            
            out_center = src_boxes[:,:2]
            out_lengths = src_boxes[:,2:4]
            out_angle = src_boxes[:,-1:]
            
            out_cos2 = torch.cos(2*out_angle)
            out_sin2 = torch.sin(2*out_angle)
            
            '''
            '''
            
            gt_center = target_boxes[:,:2]
            gt_lengths = target_boxes[:,2:4]
            gt_angle = target_boxes[:,-1:]
            
            gt_cos2 = torch.cos(2*gt_angle)
            gt_sin2 = torch.sin(2*gt_angle)
            
            
            
            center_loss = F.l1_loss(gt_center, out_center, reduction='none')
            len_loss = F.l1_loss(gt_lengths, out_lengths, reduction='none') 
            orient_loss1 = F.l1_loss(gt_cos2, out_cos2, reduction='none')
            orient_loss2 = F.l1_loss(gt_sin2, out_sin2, reduction='none')
            orient_loss = orient_loss1 + orient_loss2
    
            losses = {}
            losses['loss_obj_center'] = center_loss.mean()
    
            losses['loss_obj_len'] = len_loss.mean()
            losses['loss_obj_orient'] = orient_loss.mean()
            
            
            if 'obj_img_centers' in outputs:
                src_img_centers = outputs['obj_img_centers'][idx]
                target_img_centers = torch.cat([t['object_image_centers'][i] for t, (_, i) in zip(targets, indices)], dim=0)
                img_center_loss = F.l1_loss(target_img_centers, src_img_centers, reduction='none')
                losses['loss_obj_img_center'] = img_center_loss.mean()
    
        else:
            losses = {}
            losses['loss_obj_center'] =  torch.tensor(0).cuda()
    
            losses['loss_obj_len'] =  torch.tensor(0).cuda()
            if 'obj_img_centers' in outputs:
                losses['loss_obj_orient'] =  torch.tensor(0).cuda()
                losses['loss_obj_img_center'] =  torch.tensor(0).cuda()
    
        return losses
            
        
    def loss_polyline(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """

        
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx].contiguous().view(-1,self.num_control_points,2)
        target_boxes = torch.cat([t['control_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)
       
        target_boxes = target_boxes.contiguous().view(-1,self.num_control_points,2)
        
        inter_points = torch.matmul(self.bezierA.expand(src_boxes.size(0),-1,-1),src_boxes)
        
        target_points = torch.matmul(self.bezierA.expand(target_boxes.size(0),-1,-1),target_boxes)
        
        cost_bbox = torch.cdist(inter_points, target_points, p=1)
        
        min0 = torch.mean(torch.min(cost_bbox,dim=1)[0],dim=-1)
        min1 = torch.mean(torch.min(cost_bbox,dim=2)[0],dim=-1)
        
        
        losses = {}
        losses['loss_polyline'] = torch.mean(min0 + min1)

      
        return losses
        
        
    def loss_boxes(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        
        
    
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['control_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.mean()

      
        return losses
        
    
    
    def loss_endpoints(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        
    
        
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_endpoints'][idx]
        target_boxes = torch.cat([t['endpoints'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_endpoints'] = loss_bbox.mean()

      
        return losses
        

        
    def get_object_interpolated(self,outputs, targets,indices):
#     
        
        idx = self._get_src_permutation_idx(indices)
        
        target_ids = self._get_tgt_permutation_idx(indices)
        
        src_boxes = outputs['obj_boxes'][idx].view(-1,5)
        
        src_probs = outputs['obj_logits'][idx]
       
                
        
        centers = src_boxes[:,:2]
        angle = src_boxes[:,4]
        long_len = src_boxes[:,2]
        short_len = src_boxes[:,3]
        
        
        long_y = torch.abs(torch.sin(angle)*long_len)
        long_x = torch.cos(angle)*long_len
        
        short_x = -torch.sign(torch.cos(angle))*torch.sin(angle)*short_len
        short_y = torch.abs(torch.cos(angle)*short_len)
        
        corner_up = torch.stack([centers[:,0] + long_x/2 + short_x/2, centers[:,1] + long_y/2 + short_y/2],dim=-1)
        
        short_corner_up = corner_up - torch.stack([short_x,short_y],dim=-1)
        
        long_corner_up = corner_up - torch.stack([long_x,long_y],dim=-1)
        
        rest = long_corner_up - torch.stack([short_x,short_y],dim=-1)
        
        
        corners = torch.stack([corner_up, short_corner_up, rest, long_corner_up],dim=1)
        
        
        temp_arx = torch.ones_like(corners[...,0]).cuda()
        temp_ary = torch.ones_like(corners[...,0]).cuda()
        
        temp_arx = 2*temp_arx*corners[...,0]
        
        temp_ar = torch.stack([temp_arx, temp_ary],dim=-1)
            
        
        corners = temp_ar - corners
#        
        my_dict = dict()
        
        
        my_dict['interpolated'] = corners

        my_dict['src_boxes'] = src_boxes
        
        my_dict['src_probs'] = F.softmax(src_probs,-1)
        
        return my_dict, idx, target_ids
    
    
    def get_interpolated(self,outputs, targets,indices):
#        indices = self.matcher(outputs, targets)
        idx = self._get_src_permutation_idx(indices)
        
        target_ids = self._get_tgt_permutation_idx(indices)
        
        src_boxes = outputs['pred_boxes'][idx].view(-1,self.num_control_points,2)
        
        src_endpoints = outputs['pred_endpoints'][idx].view(-1,2,2)

        '''
        ASSOC
        '''        

        
        lab = targets[0]['con_matrix']
        lab = lab.long()
        lab = lab[target_ids[1],:]
        lab = lab[:,target_ids[1]]
             
        est = outputs['pred_assoc']
        

        
        inter_points = torch.matmul(self.bezierA.expand(src_boxes.size(0),-1,-1),src_boxes)
        my_dict = dict()
        my_dict['interpolated'] = inter_points
        my_dict['endpoints'] = src_endpoints
        my_dict['src_boxes'] = src_boxes
        my_dict['assoc_est'] = torch.squeeze(est,dim=0)
      
        my_dict['assoc_gt'] = lab
        
        return my_dict, idx, target_ids
        
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    
    
    def get_assoc_estimates(self,  outputs, indices):
        _, idx = self._get_src_permutation_idx(indices)
        
#        _, target_ids = self._get_tgt_permutation_idx(indices)
        
        assoc_features = torch.squeeze(outputs['assoc_features'])
        
        
            
        selected_features = assoc_features[idx]
        
        reshaped_features1 = torch.unsqueeze(selected_features,dim=1).repeat(1,selected_features.size(0),1)
        reshaped_features2 = torch.unsqueeze(selected_features,dim=0).repeat(selected_features.size(0),1,1)
        
        total_features = torch.cat([reshaped_features1,reshaped_features2],dim=-1)
        
        est = torch.squeeze(self.assoc_net(total_features),dim=-1)
        
        outputs['pred_assoc'] = torch.unsqueeze(est,dim=0)
        
       
        
        return outputs
    
    
    def get_loss(self, loss, outputs, targets, indices,  **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'assoc': self.loss_assoc,
        
            'boxes': self.loss_boxes,
            'loss_polyline': self.loss_polyline,
            'focal': self.focal_loss,
            'endpoints': self.loss_endpoints,
            
            'loss_obj_ce': self.loss_object_labels,
            'loss_obj_bbox': self.loss_object_boxes
            
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices,  **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices_static, indices_object = self.matcher(outputs_without_aux, targets,  do_obj=True)
        
        outputs = self.get_assoc_estimates(outputs,indices_static)
        
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if 'obj' in loss:
                
                losses.update(self.get_loss(loss, outputs, targets, indices_object))
                
                
            else:
                losses.update(self.get_loss(loss, outputs, targets, indices_static))
                
        if targets[0]['obj_exists']:
            losses['object_indices'] = (self._get_src_permutation_idx(indices_object)[1], self._get_tgt_permutation_idx(indices_object)[1])
        else:
            losses['object_indices'] = (0,0)
        
        losses['static_indices'] = (self._get_src_permutation_idx(indices_static)[1], self._get_tgt_permutation_idx(indices_static)[1])
        
        

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes, objects=True):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox, out_end, out_assoc = outputs['pred_logits'], outputs['pred_boxes'], outputs['pred_endpoints'], outputs['pred_assoc']
     
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob.max(-1)
        
        
        
        est = torch.reshape(out_bbox,(len(out_bbox),out_bbox.shape[1],-1,2))
        end_est = torch.reshape(out_end,(len(out_end),-1,2,2))
        
            
        results = [{'scores': s, 'labels': l, 'boxes': b,'probs': p,'endpoints': e,'assoc': a} for s, l, b, p, e, a in zip(scores, labels, est,prob,end_est,out_assoc)]
        
        
        if objects:
            obj_dict={}
            logits = outputs['obj_logits']
            
            prob = F.softmax(logits, -1).squeeze(0)
        
            
            src_boxes = outputs['obj_boxes'].squeeze(0)
            
            if 'obj_img_centers' in outputs:
                
                img_centers  = outputs['obj_img_centers'].squeeze(0)
                converted_img_centers = outputs['obj_converted_img_centers'].squeeze(0)
                obj_dict['obj_img_centers'] = img_centers 
                obj_dict['obj_converted_img_centers'] = converted_img_centers
              
                    
            centers = src_boxes[:,:2]
            angle = src_boxes[:,4]
            long_len = src_boxes[:,2]
            short_len = src_boxes[:,3]
            
            
            long_y = torch.abs(torch.sin(angle)*long_len)
            long_x = torch.cos(angle)*long_len
            
            short_x = -torch.sign(torch.cos(angle))*torch.sin(angle)*short_len
            short_y = torch.abs(torch.cos(angle)*short_len)
            
            corner_up = torch.stack([centers[:,0] + long_x/2 + short_x/2, centers[:,1] + long_y/2 + short_y/2],dim=-1)
            
            short_corner_up = corner_up - torch.stack([short_x,short_y],dim=-1)
            
            long_corner_up = corner_up - torch.stack([long_x,long_y],dim=-1)
            
            rest = long_corner_up - torch.stack([short_x,short_y],dim=-1)
            
            
            corners = torch.stack([corner_up, short_corner_up, rest, long_corner_up],dim=1)
            
            
            temp_arx = torch.ones_like(corners[...,0]).cuda()
            temp_ary = torch.ones_like(corners[...,0]).cuda()
            
            temp_arx = 2*temp_arx*corners[...,0]
            
            temp_ar = torch.stack([temp_arx, temp_ary],dim=-1)
                
            
            corners = temp_ar - corners
            
            obj_dict['corners'] = corners
            obj_dict['probs'] = prob
            
            if 'refine_out' in outputs:
                obj_dict['refine_out'] = outputs['refine_out']
            
                      
            
            
            return (results, obj_dict)
    
        else:
            return (results, {})
        
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args,config,params):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    # num_classes = 20 if args.dataset_file != 'coco' else 91
    num_classes = 2
    num_object_classes = args.num_object_classes
    device = torch.device(args.device)

    backbone = build_backbone(args)
    resampler1 = Resampler(4*config.map_resolution, config.map_extents)
    
    resampler2 = Resampler(2*config.map_resolution, config.map_extents)
    
    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        (resampler1, resampler2),
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        parameter_setting=params,
        args=args
    )
    
    
    if len(config.gpus) > 1:
        model = nn.DataParallel(model.cuda(), config.gpus)
    elif len(config.gpus) == 1:
        model.cuda()
    
    
    matcher = build_matcher(args)
    
    if args.objects:
        weight_dict = {'loss_ce': args.detection_loss_coef,'loss_obj_ce': args.object_detection_loss_coef, 'loss_obj_center': args.object_center_loss_coef,
                       'loss_obj_len': args.object_len_loss_coef, 'loss_obj_orient': args.object_orient_loss_coef, 'loss_bbox': args.bbox_loss_coef,'loss_polyline': args.polyline_loss_coef,
                       'loss_endpoints': args.endpoints_loss_coef,'loss_assoc': args.assoc_loss_coef ,
                
                       'loss_end_match': args.loss_end_match_coef,
                       'loss_refine': args.object_refine_loss_coef }


  
        losses = ['labels', 'boxes','loss_polyline', 'endpoints','assoc', 'loss_obj_ce','loss_obj_bbox' ]
  
  
    else:
        weight_dict = {'loss_ce': args.detection_loss_coef, 'loss_bbox': args.bbox_loss_coef,'loss_polyline': args.polyline_loss_coef,
                       'loss_endpoints': args.endpoints_loss_coef,'loss_assoc': args.assoc_loss_coef ,
                       
                       'loss_end_match': args.loss_end_match_coef,
                       }


  
        losses = ['labels', 'boxes','loss_polyline', 'endpoints','assoc' ]
  
    criterion = SetCriterion(num_classes,num_object_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, object_eos_coef=args.object_eos_coef, losses=losses, assoc_net=model.association_classifier,
           
        apply_poly_loss = args.apply_poly_loss,
                             num_coeffs=args.num_spline_points)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
   
    return model, criterion, postprocessors
