import matplotlib
matplotlib.use('Agg') 
from matplotlib.cm import get_cmap
import numpy as np
import torch
import logging
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import matplotlib.colors as colors
import random
from src.detr.util import box_ops
import scipy.ndimage as ndimage
from src.utils import bezier
import cv2 
#import networkx as nx

from scipy.spatial.distance import cdist, directed_hausdorff
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter

from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean

object_thresh_list = [0.5, 0.50,0.45,0.4,0.3,0.5,0.45,0.40]

DETECTION_COLOR_DICT = {'car': 'C0',
                    'truck': 'C1',
                    'bus': 'C2',
                    'trailer': 'C3',
                    'construction_vehicle': 'C4',
                    'pedestrian': 'C5',
                    'motorcycle': 'C6',
                    'bicycle': 'C7',
                    # 'traffic_cone': 'C8',
                    # 'barrier': 'C9'
                    }

TEMP_COLOR_LIST = ['C0','C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']


NUSC_COLOR_LIST = [np.array(colors.to_rgb(k)) for k in TEMP_COLOR_LIST]

TEMP_COLOR_LIST=[ [179, 0, 27],  [108, 13, 27],
                 [141, 2, 136],  [0,0,0],  [0,0,0],
                  [227, 216, 25],  [255, 255, 255], [255, 255, 255],
                  ]


COLOR_LIST=[]
for k in TEMP_COLOR_LIST:
    COLOR_LIST.append(np.array(k)/255)

COLOR_LIST[6] = NUSC_COLOR_LIST[6]
COLOR_LIST[7] = NUSC_COLOR_LIST[7]

# plt.imshow(np.ones((30,30,3))*np.expand_dims(np.expand_dims(COLOR_LIST[1],axis=0),axis=0))

COLOR_LIST.append(np.array([1,1,1]))

DETECTION_NAMES = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle',
                   ]



def convert_line_to_lane(coeffs, lane_width = 3.5):
    
    resolution = 0.25
    patch_size = (196,200)
    
    one_side = lane_width/2/resolution
    
    one_side_x = one_side/patch_size[1]
    one_side_y = one_side/patch_size[0]
    
    segments = len(coeffs) - 1
    
    new_coeffs_list1 = []
    new_coeffs_list2 = []
    
    for seg in range(segments):
        slope = (coeffs[seg+1,1] - coeffs[seg,1])/(coeffs[seg+1,0] - coeffs[seg,0] + 0.000001)
        
        inv_slope = -1/slope
        
        unit_vec_x = np.sqrt(1/(inv_slope**2 + 1))
        unit_vec_y = np.sqrt(1-unit_vec_x**2)*one_side_y
        unit_vec_x = unit_vec_x *one_side_x
        new_coeffs_list1.append(np.array([coeffs[seg,0] + unit_vec_x,coeffs[seg,1] + unit_vec_y]))
        new_coeffs_list1.append(np.array([coeffs[seg+1,0] + unit_vec_x,coeffs[seg+1,1] + unit_vec_y]))

        new_coeffs_list2.append(np.array([coeffs[seg,0] - unit_vec_x,coeffs[seg,1] - unit_vec_y]))
        new_coeffs_list2.append(np.array([coeffs[seg+1,0] - unit_vec_x,coeffs[seg+1,1] - unit_vec_y]))

    
    new_coeffs_list2_flipped = new_coeffs_list2[::-1]
    
    all_coeffs = new_coeffs_list1 + new_coeffs_list2_flipped
    all_coeffs = np.array(all_coeffs)
    
    return all_coeffs
    
    
def render_polygon(mask, polygon, shape, value=1):
    
#    logging.error('POLYGON ' + str(polygon.coords))
#    logging.error('EXTENTS ' + str(np.array(extents[:2])))
    to_mult = np.expand_dims(np.array([shape[1],shape[0]]),axis=0)
    polygon = polygon*to_mult
    polygon = np.ascontiguousarray(polygon).round().astype(np.int32)
    cv2.fillConvexPoly(mask, polygon, value)



def colorise(tensor, cmap, vmin=None, vmax=None):

    if isinstance(cmap, str):
        cmap = get_cmap(cmap)
    
    tensor = tensor.detach().cpu().float()

    vmin = float(tensor.min()) if vmin is None else vmin
    vmax = float(tensor.max()) if vmax is None else vmax

    tensor = (tensor - vmin) / (vmax - vmin)
    return cmap(tensor.numpy())[..., :3]



def _pascal_color_map(N=256, normalized=False):
    """
    Python implementation of the color map function for the PASCAL VOC data set.
    Official Matlab version can be found in the PASCAL VOC devkit
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap

def overlay_semantic_mask(im, ann, alpha=0.5, colors=None, contour_thickness=None):
    im, ann = np.asarray(im, dtype=np.uint8), np.asarray(ann, dtype=np.int)
    if im.shape[:-1] != ann.shape:
        raise ValueError('First two dimensions of `im` and `ann` must match')
    if im.shape[-1] != 3:
        raise ValueError('im must have three channels at the 3 dimension')

    colors = colors or _pascal_color_map()
    colors = np.asarray(colors, dtype=np.uint8)

    mask = colors[ann]
    fg = im * alpha + (1 - alpha) * mask

    img = im.copy()
    img[ann > 0] = fg[ann > 0]

    if contour_thickness:  # pragma: no cover
        import cv2
        for obj_id in np.unique(ann[ann > 0]):
            contours = cv2.findContours((ann == obj_id).astype(
                np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            cv2.drawContours(img, contours[0], -1, colors[obj_id].tolist(),
                             contour_thickness)
    return img


def my_line_maker(points,size=(196,200)):
    
    res = np.zeros(size)
    for k in range(len(points)):
        res[np.min([int(points[k][1]*size[0]),int(size[0]-1)]),np.min([int(points[k][0]*size[1]),int(size[1]-1)])] = 1
    return np.uint8(255*res)


def my_color_line_maker(points,endpoints,size=(196,200)):
    if len(endpoints) == 4:
        endpoints = np.reshape(endpoints,[2,2])
    res = np.zeros((size[0],size[1],3))
    for k in range(len(points)):
        res[np.min([int(points[k][1]*size[0]),int(size[0]-1)]),np.min([int(points[k][0]*size[1]),int(size[1]-1)])] = 1
    
    base_start = np.zeros((res.shape[0],res.shape[1]))
    base_start[np.min([int(endpoints[0,1]*size[0]),int(size[0]-1)]),np.min([int(endpoints[0,0]*size[1]),int(size[1]-1)])] = 1
    struct = ndimage.generate_binary_structure(2, 2)
    # struct = ndimage.generate_binary_structure(5, 2)
    
    # logging.error('STRUCT ' + str(struct))
    # logging.error('BASE START ' + str(base_start.shape))
    
    dilated = ndimage.binary_dilation(base_start>0, structure=struct)
    
    res[dilated,0] = 0
    res[dilated,1] = 1
    res[dilated,2] = 0
    
    base_end = np.zeros((res.shape[0],res.shape[1]))
    base_end[np.min([int(endpoints[1,1]*size[0]),int(size[0]-1)]),np.min([int(endpoints[1,0]*size[1]),int(size[1]-1)])] = 1
    
    # struct = ndimage.generate_binary_structure(2, 1)
    dilated = ndimage.binary_dilation(base_end>0, structure=struct)
    
    res[dilated,0] = 1
    res[dilated,1] = 0
    res[dilated,2] = 0
    
    
    return np.uint8(255*res)


def my_float_line_maker(points,size=(196,200)):
    
    res = np.zeros(size)
    for k in range(len(points)):
        res[np.min([int(points[k][1]*size[0]),int(size[0]-1)]),np.min([int(points[k][0]*size[1]),int(size[1]-1)])] = 1
    return res

def prepare_refine_inputs(object_post, targets, crit_indices_object, transforms):
    
    source_ids, target_ids = crit_indices_object
    
    occ_mask = targets['bev_mask'][-1].cpu().numpy()
    true_segs = targets['bev_mask']
    
    my_shape = [occ_mask.shape[0]//4,occ_mask.shape[1]//4]
    
    estimates = object_post['corners'].cpu().numpy()
    # logging.error(str(estimates.shape))
    
    estimates = estimates[source_ids]
    
    probs = object_post['probs'].cpu().numpy()
    
    
    probs = probs[source_ids]
    
    if len(estimates.shape) == 2:
        estimates = np.expand_dims(estimates,axis=0)
        
    if len(probs.shape) == 1:
        probs = np.expand_dims(probs,axis=0)
        
    # gt_rendered_polygons_list=[]
    # for k in range(len(orig_corners)):
    #     obj_seg_out = np.zeros((196,200))    
        
    #     render_polygon(obj_seg_out, np.reshape(orig_corners[k][:8],(4,2)), [196,200])
    #     gt_rendered_polygons_list.append(np.copy(obj_seg_out))
        
    # ref_target = torch.tensor(np.array(gt_rendered_polygons_list)).cuda()
        
    ref_target = true_segs[:probs.shape[1] - 1]
    
    bg_label = 1 - torch.sum(ref_target,dim=0).unsqueeze(0).clamp(0,1)
    
    ref_target = torch.cat([ref_target,bg_label],dim=0).long().cpu().numpy()
    ref_target = torch.tensor(np.expand_dims(np.argmax(ref_target,axis=0),axis=0)).cuda()
    # vis_mask = ref_target
    
    expanded_probs = np.expand_dims(np.expand_dims(probs,axis=-1),axis=-1)
     
    rendered_polygons_list=[]
    feed_poly = np.zeros((probs.shape[1],my_shape[0],my_shape[1]))
    
    for k in range(len(estimates)):
     
        
        obj_seg_out = np.zeros((49,50))    
        
        render_polygon(obj_seg_out, estimates[k], [my_shape[0],my_shape[1]])
        
        feed_poly = feed_poly + np.expand_dims(np.copy(obj_seg_out),axis=0)*expanded_probs[k]
        
        # rendered_polygons_list.append(np.copy(obj_seg_out))
    feed_poly = np.clip(feed_poly,0,1)
    
    ref_in = dict()    
   
    ref_in['anything_to_feed'] =True
    
    orig_feed_polygons = torch.tensor(feed_poly).unsqueeze(0).cuda()
    
    # to_feed_probs = torch.tensor(probs).cuda().unsqueeze(-1).unsqueeze(-1).repeat(1,1,49,50)
    
    
    
    # orig_feed_polygons = torch.tensor(np.array(rendered_polygons_list)).cuda()
            
    # to_feed_polygons = transforms(orig_feed_polygons).unsqueeze(1)
    
    
    
    # ref_in['orig_feed_polygons'] = orig_feed_polygons
    ref_in['small_rendered_polygons'] = orig_feed_polygons
        
    # ref_in['to_feed_probs'] = to_feed_probs
    
    return ref_in, ref_target


def get_selected_objects(targets, thresh = 0.5):
    
    torch_probs = targets['probs']
    torch_corners = targets['corners']
    
    
    temp_scores = torch_probs[...,:-1].contiguous()
    temp_max_scores, temp_max_scores_ind = temp_scores.max(-1)
    
    temp_dict={}
    
    probs = targets['probs'].detach().cpu().numpy()
    corners = targets['corners'].detach().cpu().numpy()
    
    
    # max_prob = np.argmax(probs,axis=-1)
    # crit = max_prob < (probs.shape[-1] - 1)
#    logging.error('CRIT ' + str(crit.shape))
    max_prob_v = np.max(probs[:,:-1],axis=-1)
    
    
    crit = max_prob_v >= thresh
    
    selected_corners = corners[crit]
    selected_probs = probs[crit]
    
    keep_ind = np.arange(len(selected_corners))
    
     
    
    temp_dict['nms_keep_ind'] = keep_ind
    
    temp_dict['corners'] = selected_corners
    temp_dict['probs'] = selected_probs
    
    if  'refine_out' in targets:
        temp_dict['refine_out'] = targets['refine_out']
        
    else:
           
#        logging.error('GET SELECTED OBJECTS ')
        if np.sum(crit) > 0:
            expanded_probs = np.expand_dims(np.expand_dims(selected_probs,axis=-1),axis=-1)
             
            rendered_polygons_list=[]
            feed_poly = np.zeros((probs.shape[1],49,50))
            
            for k in range(len(selected_corners)):
             
                
                obj_seg_out = np.zeros((49,50))    
                
                render_polygon(obj_seg_out, selected_corners[k], [49,50])
                
                feed_poly = feed_poly + np.expand_dims(np.copy(obj_seg_out),axis=0)*expanded_probs[k]
                
                # rendered_polygons_list.append(np.copy(obj_seg_out))
            feed_poly = np.clip(feed_poly,0,1)
            orig_feed_polygons = torch.tensor(feed_poly).unsqueeze(0).cuda()
    
            # temp_dict['small_rendered_polygons'] = to_feed_polygons
            temp_dict['small_rendered_polygons'] = orig_feed_polygons
            # temp_dict['to_feed_probs'] = to_feed_probs
                
            temp_dict['anything_to_feed'] = True
            
        else:
            temp_dict['small_rendered_polygons'] = None
            # temp_dict['to_feed_probs'] = None
            temp_dict['anything_to_feed'] = False
            
        
    
    return temp_dict
   
def hausdorff_match(out, target,pinet=False):
    
    # res_coef_list = out['interpolated_points']
    est_coefs = out['boxes']
    
    orig_coefs = target['control_points'].cpu().numpy()
    orig_coefs = np.reshape(orig_coefs, (-1, int(orig_coefs.shape[-1]/2),2))
    
    interpolated_origs = []
    
    for k in range(len(orig_coefs)):
        inter = bezier.interpolate_bezier(orig_coefs[k],100)
        interpolated_origs.append(np.copy(inter))
    
    if len(est_coefs) == 0:
        return None,None, interpolated_origs
    
    
    
    dist_mat = np.mean(np.sum(np.square(np.expand_dims(est_coefs,axis=1) - np.expand_dims(orig_coefs,axis=0)),axis=-1),axis=-1)
    
    if pinet:
        second_dist_mat = np.mean(np.sum(np.square(np.expand_dims(est_coefs[:,::-1,:],axis=1) - np.expand_dims(orig_coefs,axis=0)),axis=-1),axis=-1)
        dist_mat = np.min(np.stack([dist_mat,second_dist_mat],axis=0),axis=0)
    
    ind = np.argmin(dist_mat, axis=-1)
    min_vals = np.min(dist_mat,axis=-1)
    
  
        
        
    return min_vals, ind, interpolated_origs 

    
def merged_hausdorff_match(out, target):
    
    # res_coef_list = out['interpolated_points']
    est_coefs = out['merged_coeffs']
    
    # est_coefs = np.reshape(est_coefs,(est_coefs.shape[0],-1))
    
    orig_coefs = target['control_points'].cpu().numpy()
    orig_coefs = np.reshape(orig_coefs, (-1, int(orig_coefs.shape[-1]/2),2))
    
    interpolated_origs = []
    
    for k in range(len(orig_coefs)):
        inter = bezier.interpolate_bezier(orig_coefs[k],100)
        interpolated_origs.append(np.copy(inter))
    
    if len(est_coefs) == 0:
        return None,None, interpolated_origs
    dist_mat = np.mean(np.sum(np.square(np.expand_dims(est_coefs,axis=1) - np.expand_dims(orig_coefs,axis=0)),axis=-1),axis=-1)
    
    ind = np.argmin(dist_mat, axis=-1)
    min_vals = np.min(dist_mat,axis=-1)
    
  
        
        
    return min_vals, ind, interpolated_origs 


def get_selected_estimates(targets, thresh = 0.5):
    
    res = []
    for b in range(len(targets)):
        
        temp_dict = dict()
        
        scores = targets[b]['scores'].detach().cpu().numpy()
        probs = targets[b]['probs'].detach().cpu().numpy()
        labels = targets[b]['labels'].detach().cpu().numpy()
        coeffs = targets[b]['boxes'].detach().cpu().numpy()
        endpoints = targets[b]['endpoints'].detach().cpu().numpy()
        assoc = targets[b]['assoc'].detach().cpu().numpy()
        
        
        selecteds = probs[:,1] > thresh
        
        detected_scores = probs[selecteds,1]
        detected_coeffs = coeffs[selecteds,...]
        detected_endpoints = endpoints[selecteds,...]
        
        all_roads = np.zeros((196,200,3),np.float32)
        coef_all_roads = np.zeros((196,200,3),np.float32)
        if len(detected_scores) > 0:
            
            temp_dict['scores'] = detected_scores
            temp_dict['boxes'] = detected_coeffs
            temp_dict['endpoints'] = detected_endpoints
            temp_dict['assoc'] = assoc
#          
            to_merge = {'assoc': assoc,'boxes':detected_coeffs}
            merged = get_merged_coeffs(to_merge)
            temp_dict['merged_coeffs'] = merged
        
            res_list = []
            res_coef_list=[]
            
            res_interpolated_list=[]
            # res_assoc_list = []
            
            for k in range(len(detected_scores)):
                

                control = detected_coeffs[k]
                
                coef_endpoints = np.concatenate([control[0:1,:],control[-1:,:]],axis=0)
                
                interpolated = bezier.interpolate_bezier(control,100)
                
                res_interpolated_list.append(np.copy(interpolated))
                
                line = my_color_line_maker(interpolated,detected_endpoints[k],size=(196,200))
                line2 = my_color_line_maker(interpolated,coef_endpoints,size=(196,200))
                res_list.append(line)
                res_coef_list.append(line2)
                all_roads = all_roads + np.float32(line)
                coef_all_roads = coef_all_roads + np.float32(line2)
            
            temp_dict['lines'] = res_list
            temp_dict['coef_lines'] = res_coef_list
            
            temp_dict['interpolated_points'] = res_interpolated_list
            
            temp_dict['all_roads'] = all_roads
            temp_dict['coef_all_roads'] = coef_all_roads
            temp_dict['labels'] = labels[selecteds]
            
            merged_interpolated_list=[]
            for k in range(len(merged)):
                

                control = merged[k]
                
                coef_endpoints = np.concatenate([control[0:1,:],control[-1:,:]],axis=0)
                
                merged_interpolated = bezier.interpolate_bezier(control,100)
                
                merged_interpolated_list.append(np.copy(merged_interpolated))
            
            
            temp_dict['merged_interpolated_points'] = merged_interpolated_list
            
            
            
        else:
            
            logging.error('DETECTED NOTHING')
            temp_dict['scores'] = []
            temp_dict['interpolated_points'] = []
            temp_dict['scores'] = []
            temp_dict['boxes'] = []
            temp_dict['lines'] = []
            temp_dict['coef_lines'] = []
            temp_dict['all_roads'] = []
            temp_dict['coef_all_roads'] = []
            temp_dict['labels'] = []
            temp_dict['assoc'] = []
 
            temp_dict['merged_interpolated_points'] = []
            temp_dict['merged_coeffs'] = []
        
        res.append(temp_dict)
        
    return res
            

def get_vertices(adj):

    ins = []
    outs = []
    
    for k in range(len(adj)):
    #for k in range(7):
        for m in range(len(adj)):
        
            if adj[k,m] > 0.5:
                if len(ins) > 0:
                    ins_exists = False
                    out_exists = False
    
                    for temin in range(len(ins)):
                        if k in ins[temin]:
                            if not (m in outs[temin]):
                                outs[temin].append(m)
                            ins_exists=True
                            break
                    
                    if not ins_exists:
                        for temin in range(len(outs)):
                            if m in outs[temin]:
                                if not (k in ins[temin]):
                                    ins[temin].append(k)
                                out_exists=True
                                break
                        
                        if not out_exists:
                            ins.append([k])
                            outs.append([m])
                            
                else:
                    ins.append([k])
                    outs.append([m])
                            
    
    return ins, outs                    
   

def gather_all_ends(adj):

    clusters = []
    
    for k in range(len(adj)):
    #for k in range(7):
        for m in range(len(adj)):
        
            if adj[k,m] > 0.5:
                if len(clusters) > 0:
                    ins_exists = False
                    out_exists = False
    
                    for temin in range(len(clusters)):
                        if k in clusters[temin]:
                            if not (m in clusters[temin]):
                                clusters[temin].append(m)
                            ins_exists=True
                            break
                    
                    if not ins_exists:
                        for temin in range(len(clusters)):
                            if m in clusters[temin]:
                                if not (k in clusters[temin]):
                                    clusters[temin].append(k)
                                out_exists=True
                                break
                        
                        if not out_exists:
                            clusters.append([k,m])
                            
                else:
                    clusters.append([k,m])
                            
    
    return clusters             
#

def get_merged_coeffs(targets):
    
    coeffs = targets['boxes']
     
    assoc = targets['assoc'] 
           
    diag_mask = np.eye(len(assoc))
    
    diag_mask = 1 - diag_mask
    assoc = assoc*diag_mask
    
    corrected_coeffs = np.copy(coeffs)
    
    ins, outs = get_vertices(assoc)
    
    
    
    for k in range(len(ins)):
        all_points=[]
        for m in ins[k]:
            all_points.append(corrected_coeffs[m,-1])
            
        for m in outs[k]:
            all_points.append(corrected_coeffs[m,0])
            
        
        av_p = np.mean(np.stack(all_points,axis=0),axis=0)
        
        for m in ins[k]:
            corrected_coeffs[m,-1] = av_p
            
        for m in outs[k]:
            corrected_coeffs[m,0] = av_p
    
    
    return corrected_coeffs


def get_merged_network(targets):
    
    coeffs = targets['boxes']
    
    assoc = targets['assoc'] 
            
    diag_mask = np.eye(len(assoc))
    
    diag_mask = 1 - diag_mask
    assoc = assoc*diag_mask
    
    corrected_coeffs = np.copy(coeffs)
    
    ins, outs = get_vertices(assoc)
    
    
    
    for k in range(len(ins)):
        all_points=[]
        for m in ins[k]:
            all_points.append(corrected_coeffs[m,-1])
            
        for m in outs[k]:
            all_points.append(corrected_coeffs[m,0])
            
        
        av_p = np.mean(np.stack(all_points,axis=0),axis=0)
        
        for m in ins[k]:
            corrected_coeffs[m,-1] = av_p
            
        for m in outs[k]:
            corrected_coeffs[m,0] = av_p
    
    lines=[]
            
    for k in range(len(corrected_coeffs)):        
        control = corrected_coeffs[k]
        coef_endpoints = np.concatenate([control[0:1,:],control[-1:,:]],axis=0)
        interpolated = bezier.interpolate_bezier(control)
        line = np.float32(my_color_line_maker(interpolated,coef_endpoints,size=(196,200)))/255
        lines.append(line)    
        
    return lines
        
    
def get_merged_lines(coeffs1,coeffs2):
    
    interp_list = []

        
    control = coeffs1

    interpolated = bezier.interpolate_bezier(control)
    
    interp_list.append(interpolated)
    
    control = coeffs2

    interpolated = bezier.interpolate_bezier(control)
    
    interp_list.append(interpolated)
    
        
    all_points = np.concatenate(interp_list,axis=0)
        
    
    new_coeffs = bezier.fit_bezier(all_points, len(coeffs1))[0]
        
    
    return None, new_coeffs

def visual_est(images,targets,save_path,name=None):
 
    
    for b in range(len(targets)):
        
        
        merged = get_merged_network(targets[b])
        
        
        
        if len(merged) > 0:
            
            merged = np.sum(np.stack(merged,axis=0),axis=0)
            merged = np.uint8(np.clip(merged, 0, 1)*255)
            res = Image.fromarray(merged)
            
            if name==None:
                res.save(os.path.join(save_path,'batch_'+str(b) + '_merged_road.jpg'))
                
            else:
                res.save(os.path.join(save_path,name + '_merged_road.jpg'))
        else:
            logging.error('EMPTY MERGED')
        
        scores = targets[b]['scores']
        labels = targets[b]['labels']
        coeffs = targets[b]['boxes']
        
        res_list = targets[b]['lines'] 
        res_coef_list = targets[b]['coef_lines'] 
        all_roads = targets[b]['all_roads'] 
        coef_all_roads = targets[b]['coef_all_roads'] 
        assoc = targets[b]['assoc'] 
            
        # logging.error('VIS EST '+ str(assoc.shape))
        if len(res_list) > 0:
            all_lanes = np.zeros((196,200))
            for k in range(len(res_list)):
                
                
                res = Image.fromarray(res_list[k])
                res_coef = Image.fromarray(res_coef_list[k])
                if name==None:
                    res.save(os.path.join(save_path,'batch_'+str(b) + '_est_interp_road_'+str(k)+'.jpg'))
                    res_coef.save(os.path.join(save_path,'batch_'+str(b) + '_est_coef_interp_road_'+str(k)+'.jpg'))
  
                
                else:
                    res.save(os.path.join(save_path,name + '_est_interp_road_'+str(k)+'.jpg'))
                    res_coef.save(os.path.join(save_path,name + '_est_coef_interp_road_'+str(k)+'.jpg'))
                    
#                plt.figure()
#                fig, ax = plt.subplots(1, figsize=(196,200))
##                axes = plt.gca()
#                ax.set_xlim([0,1])
#                ax.set_ylim([0,1])
#                plt.plot(interpolated[:,0],interpolated[:,1])
#                
#                plt.savefig(os.path.join(save_path,'batch_'+str(b) + '_est_interp_road_'+str(k)+'.jpg'), bbox_inches='tight', pad_inches=0.0)   
#                plt.close()  
            
                # merged, merged_coeffs = get_merged_lines(coeffs,assoc,k)
                # for m in range(len(assoc[k])):
                #     if assoc[k][m] > 0:
                #         first_one = np.float32(res_coef_list[k])/255
                #         second_one = np.float32(res_coef_list[m])/255
                        
                #         tot = np.clip(first_one + second_one,0,1)
                #         temp_img = Image.fromarray(np.uint8( tot*255))
                        
                #         if name==None:
                #             temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_est_assoc_from_'+str(k)+'to_'+str(m)+'.jpg'))
                        
                #         else:
                #             temp_img.save(os.path.join(save_path,name + '_est_assoc_from_'+str(k)+'to_'+str(m)+'.jpg'))
                     
            all_lanes = np.uint8(np.clip(all_lanes,0,1)*255)
            if name==None:
                
                
                all_roads = np.uint8(np.clip(all_roads,0,1)*255)
                temp_img = Image.fromarray(all_roads)
                temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_est_all_roads.jpg' ))       
                
                coef_all_roads = np.uint8(np.clip(coef_all_roads,0,1)*255)
                temp_img = Image.fromarray(coef_all_roads)
                temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_est_coef_all_roads.jpg' ))       
            else:
                
                all_roads = np.uint8(np.clip(all_roads,0,1)*255)
                temp_img = Image.fromarray(all_roads)
                temp_img.save(os.path.join(save_path,name + '_est_all_roads.jpg' ))    
                
                coef_all_roads = np.uint8(np.clip(coef_all_roads,0,1)*255)
                temp_img = Image.fromarray(coef_all_roads)
                temp_img.save(os.path.join(save_path,name + '_est_coef_all_roads.jpg' ))    

def plot_obj_img_centers(images,obj_img_centers):
    
    images = np.squeeze(images)
#    logging.error('IMAGES ' + str(images.shape))
    out_ar = np.zeros_like(images)[...,0]
#
    
    
    for k in range(len(obj_img_centers)):
        cur_cen = obj_img_centers[k]
        
        out_ar[int(cur_cen[1]*447),int(cur_cen[0]*799)] = 1
        
            
    struct = np.ones((7,7)) > 0
    
    dilated = ndimage.binary_dilation(out_ar, structure=struct)
    
#    img = np.squeeze(images)
#    logging.error('DILATED GET')
    overlayed = overlay_semantic_mask(images,dilated)
    
#    logging.error('OVERLAY GOT')
    return overlayed

def visual_masks_gt(images,targets,save_path,name=None):
 
    
    for b in range(len(targets)):
        img_centers = targets[b]['center_img']
        
        img_centers = img_centers.cpu().numpy()
        
        orig_img_centers = targets[b]['orig_center_img']
        
        orig_img_centers = orig_img_centers.cpu().numpy()
        
        roads = targets[b]['roads'].cpu().numpy()
        
        all_endpoints = targets[b]['endpoints'].cpu().numpy()
        
        true_assoc = targets[b]['con_matrix'].cpu().numpy()
        
        all_roads = np.zeros((img_centers.shape[0],img_centers.shape[1],3))
        coef_all_roads = np.zeros((img_centers.shape[0],img_centers.shape[1],3))
        
        if targets[b]['obj_exists']:
            orig_corners = targets[b]['obj_corners'].cpu().numpy()
          
            segwise_objs = np.sum(targets[b]['bev_mask'].cpu().numpy()[:-1],axis=0)
            
            
            obj_seg_list = []
            
            for k in range(len(orig_corners)):
                
                
                
                
                obj_seg_out = np.zeros((196,200))
                cur_est = np.reshape(orig_corners[k][:-1],(4,2))
                gt_label = int(orig_corners[k,-1])
                render_polygon(obj_seg_out, cur_est, [196,200])
                
                obj_seg_out = np.stack([obj_seg_out,obj_seg_out,obj_seg_out],axis=-1)
                obj_seg_out = obj_seg_out * COLOR_LIST[gt_label]
                
                obj_seg_list.append(np.copy(obj_seg_out))
            
            seg_img = Image.fromarray(np.uint8(np.clip(segwise_objs,0,1)*255))
            
            temp_img = Image.fromarray(np.uint8( np.clip(np.sum(np.stack(obj_seg_list,axis=0),axis=0),0,1)*255))
            # obj_centers_img = Image.fromarray(out_ar)
            if name==None:
                temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_all_gt_objects.jpg'))
                # obj_centers_img.save(os.path.join(save_path,'batch_'+str(b) + '_all_gt_image_centers.jpg'))
                seg_img.save(os.path.join(save_path,'batch_'+str(b) + '_all_gt_seg_objs.jpg'))
            
            else:
                temp_img.save(os.path.join(save_path,name + '_all_gt_objects.jpg'))
                # obj_centers_img.save(os.path.join(save_path,name + '_all_gt_image_centers.jpg'))
                seg_img.save(os.path.join(save_path,name + '_all_gt_seg_objs.jpg'))
     
        # orig_five_params = targets[b]['obj_converted'].cpu().numpy()
    
        
#        all_orig_roads = np.zeros((img_centers.shape[0],img_centers.shape[1],3))
#        
#        all_smoothed = targets[b]['smoothed'].cpu().numpy()
##        logging.error('SMOOTHED ' + str(all_smoothed.shape))
#        all_dilated = targets[b]['dilated'].cpu().numpy()
##        logging.error('DILATED ' + str(all_dilated.shape))
#
#        all_origs = targets[b]['origs']
        
        orig_coefs = targets[b]['control_points'].cpu().numpy()
    
        coef_endpoints = get_endpoints_from_coeffs(orig_coefs)
    
        # all_masks = targets[b]['mask'].cpu().numpy()
        occ_img = Image.fromarray(np.uint8(255*targets[b]['bev_mask'].cpu().numpy()[-1]))
        # vis_img = Image.fromarray(np.uint8(255*all_masks[0]))
        
        all_lanes = np.zeros((196,200))
        
        # drivable_area = targets[b]['bev_mask'].cpu().numpy()[0]
        # driv_img = Image.fromarray(np.uint8(255*drivable_area))
        
        # lane_area = targets[b]['static_mask'].cpu().numpy()[4]
        # lane_img = Image.fromarray(np.uint8(255*lane_area))
        
        # van_occ = targets[b]['bev_mask'].cpu().numpy()[-1]
        # van_occ = Image.fromarray(np.uint8(255*van_occ))
        if name==None:
            
        #     lane_img.save(os.path.join(save_path,'batch_'+str(b) + '_lane_layer.jpg'))
            
            # van_occ.save(os.path.join(save_path,'batch_'+str(b) + '_van_occ.jpg'))
        #     driv_img.save(os.path.join(save_path,'batch_'+str(b) + '_drivable.jpg'))
            occ_img.save(os.path.join(save_path,'batch_'+str(b) + '_occ.jpg'))
        #     vis_img.save(os.path.join(save_path,'batch_'+str(b) + '_vis.jpg'))
        else:
        #     lane_img.save(os.path.join(save_path,name + '_lane_layer.jpg'))
            
            # van_occ.save(os.path.join(save_path,name + '_van_occ.jpg'))
        #     driv_img.save(os.path.join(save_path,name + '_drivable.jpg'))
            occ_img.save(os.path.join(save_path,name + '_occ.jpg'))
        #     vis_img.save(os.path.join(save_path,name + '_vis.jpg'))
            
        for k in range(len(roads)):
            cur_full = add_endpoints_to_line(np.float32(img_centers == roads[k]),all_endpoints[k])
            cur_coef_full = add_endpoints_to_line(np.float32(img_centers == roads[k]),coef_endpoints[k])
            
#            all_roads[img_centers == roads[k]] = 1
            temp_img = Image.fromarray(np.uint8(cur_full*255))
            
            temp_coef_img = Image.fromarray(np.uint8(cur_coef_full*255))
            
            all_roads = all_roads + cur_full
            coef_all_roads = coef_all_roads + cur_coef_full
            
            lane_poly = convert_line_to_lane(np.reshape(orig_coefs[k],(-1,2)), lane_width = 3.5)
            can = np.zeros((196,200))
    
            render_polygon(can, lane_poly, shape=(196,200), value=1)
            
            all_lanes = all_lanes + can
            lane_img = Image.fromarray(np.uint8(can*255))
            
            # for m in range(len(true_assoc[k])):
            #     if true_assoc[k][m] > 0.5:
            #         first_one = add_endpoints_to_line(np.float32(img_centers == roads[k]),coef_endpoints[k])
            
            #         second_one = add_endpoints_to_line(np.float32(img_centers == roads[m]),coef_endpoints[m])
            
            #         tot = np.clip(first_one + second_one,0,1)
            #         temp_img = Image.fromarray(np.uint8( tot*255))
                    
            #         if name==None:
            #             temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_true_assoc_from_'+str(k)+'to_'+str(m)+'.jpg'))
                    
            #         else:
            #             temp_img.save(os.path.join(save_path,name + '_true_assoc_from_'+str(k)+'to_'+str(m)+'.jpg'))
                 
                    
          
                 
                    # temp_img.save(os.path.join(save_path,'gt_assoc_from_'+str(k)+'to_'+str(m)+'.jpg' ))
        

            
            if name==None:
 #                orig_img.save(os.path.join(save_path,'batch_'+str(b) + '_gt_orig_road_'+str(k)+'.jpg' ))
                
                 temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_gt_visible_road_'+str(k)+'.jpg' ))
                 temp_coef_img.save(os.path.join(save_path,'batch_'+str(b) + '_gt_coef_visible_road_'+str(k)+'.jpg' ))
                
#                 lane_img.save(os.path.join(save_path,'batch_'+str(b) + '_gt_lane_'+str(k)+'.jpg' ))

            else:

                 temp_img.save(os.path.join(save_path,name + '_visible_road_'+str(k)+'.jpg' ))
                 temp_coef_img.save(os.path.join(save_path,name + '_coef_visible_road_'+str(k)+'.jpg' ))
#                 lane_img.save(os.path.join(save_path,name + '_gt_lane_'+str(k)+'.jpg' ))

            
                
        all_roads = np.clip(all_roads,0,1)
        coef_all_roads = np.clip(coef_all_roads,0,1)
        
        all_lanes = np.clip(all_lanes,0,1)
        
        
        if name==None:
        
            temp_img = Image.fromarray(np.uint8(all_roads*255))
            temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_gt_visible_all_roads.jpg' ))
            
            temp_img = Image.fromarray(np.uint8(coef_all_roads*255))
            temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_gt_coef_visible_all_roads.jpg' ))
            
            
            temp_img = Image.fromarray(np.uint8(all_lanes*255))
            temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_gt_all_lanes.jpg' ))
            
            # temp_img = Image.fromarray(np.uint8(all_orig_roads*255))
            # temp_img.save(os.path.join(save_path,'batch_'+str(b)  + '_gt_comp_all_roads.jpg' ))
            
        else:
            temp_img = Image.fromarray(np.uint8(all_roads*255))
            temp_img.save(os.path.join(save_path,name + '_gt_visible_all_roads.jpg' ))
            
            temp_img = Image.fromarray(np.uint8(coef_all_roads*255))
            temp_img.save(os.path.join(save_path,name + '_gt_coef_visible_all_roads.jpg' ))
            
            temp_img = Image.fromarray(np.uint8(all_lanes*255))
            temp_img.save(os.path.join(save_path,name + '_gt_all_lanes.jpg' ))
            
            # temp_img = Image.fromarray(np.uint8(all_orig_roads*255))
            # temp_img.save(os.path.join(save_path,name + '_gt_comp_all_roads.jpg' ))
     
     
def process_image(image):
    
    image = np.transpose(image,(0,2,3,1))
    
    image = (image + 1)/2*255
    return image
 
    
def add_endpoints_to_line(ar,endpoints):
    if len(endpoints) == 4:
        endpoints = np.reshape(endpoints,[2,2])
    size = ar.shape
    res = np.zeros((ar.shape[0],ar.shape[1],3))
    res[ar > 0] = 1
    
    # logging.error('AR SHAPE ' + str(ar.shape))
    # logging.error('ENDPOINTS SHAPE ' + str(endpoints.shape))
    # logging.error('ENDPOINTS ' + str(endpoints))
    
    base_start = np.zeros((ar.shape[0],ar.shape[1]))
    base_start[np.min([int(endpoints[0,1]*size[0]),int(size[0]-1)]),np.min([int(endpoints[0,0]*size[1]),int(size[1]-1)])] = 1
    
    # struct = ndimage.generate_binary_structure(5, 2)
    struct = ndimage.generate_binary_structure(2, 2)
    dilated = ndimage.binary_dilation(base_start>0, structure=struct)
    
    res[dilated,0] = 0
    res[dilated,1] = 1
    res[dilated,2] = 0
    
    base_end = np.zeros((ar.shape[0],ar.shape[1]))
    base_end[np.min([int(endpoints[1,1]*size[0]),int(size[0]-1)]),np.min([int(endpoints[1,0]*size[1]),int(size[1]-1)])] = 1
    
    # struct = ndimage.generate_binary_structure(2, 1)
    dilated = ndimage.binary_dilation(base_end>0, structure=struct)
    
    res[dilated,0] = 1
    res[dilated,1] = 0
    res[dilated,2] = 0
    
    # res[int(endpoints[0,1]*size[0]),int(endpoints[0,0]*size[1]),0] = 1
    # res[int(endpoints[0,1]*size[0]),int(endpoints[0,0]*size[1]),1] = 0
    # res[int(endpoints[0,1]*size[0]),int(endpoints[0,0]*size[1]),2] = 0
    
    # res[int(endpoints[1,1]*size[0]),int(endpoints[1,0]*size[1]),0] = 0
    # res[int(endpoints[1,1]*size[0]),int(endpoints[1,0]*size[1]),1] = 0
    # res[int(endpoints[1,1]*size[0]),int(endpoints[1,0]*size[1]),2] = 1
    
    return res
    

def get_endpoints_from_coeffs(coeffs):
    
    start = coeffs[:,:2]
    end = coeffs[:,-2:]
    
    return np.concatenate([start,end],axis=-1)
    


def five_params_to_corners(src_boxes):
    

    
    centers = src_boxes[:,:2]
    angle = src_boxes[:,4]
    long_len = src_boxes[:,2]
    short_len = src_boxes[:,3]
    
    
    long_y = np.abs(np.sin(angle)*long_len)
    long_x = np.cos(angle)*long_len
    
    short_x = -np.sign(np.cos(angle))*np.sin(angle)*short_len
    short_y = np.abs(np.cos(angle)*short_len)
    
    corner_up = np.stack([centers[:,0] + long_x/2 + short_x/2, centers[:,1] + long_y/2 + short_y/2],axis=-1)
    
    short_corner_up = corner_up - np.stack([short_x,short_y],axis=-1)
    
    long_corner_up = corner_up - np.stack([long_x,long_y],axis=-1)
    
    rest = long_corner_up - np.stack([short_x,short_y],axis=-1)
    
    
    corners = np.stack([corner_up, short_corner_up, rest, long_corner_up],axis=1)
    
    corners[...,-1] = 1 - corners[...,-1]
#    
#    centers = src_boxes[:,:2]
#    egim = src_boxes[:,4]
#    long_len = src_boxes[:,2]
#    short_len = src_boxes[:,3]
#    
#    
#    long_y = np.abs(egim*long_len)
#    long_x = np.sign(egim)*np.sqrt(1-np.square(egim))*long_len
#    
#    short_x = -egim*short_len
#    short_y = np.sqrt(1-np.square(egim))*short_len
#    
#    corner_up = np.stack([centers[:,0] + long_x/2 + short_x/2, centers[:,1] + long_y/2 + short_y/2],axis=-1)
#    
#    short_corner_up = corner_up - np.stack([short_x,short_y],axis=-1)
#    
#    long_corner_up = corner_up - np.stack([long_x,long_y],axis=-1)
#    
#    rest = long_corner_up - np.stack([short_x,short_y],axis=-1)
#    
#    
#    corners = np.stack([corner_up, short_corner_up, rest, long_corner_up],axis=1)

    return corners

def visual_object_est(images,targets,save_path,vis_mask = None,name=None):
#    raw_corners = targets['raw_corners']
#    raw_probs = targets['raw_probs']
    vis_mask = np.expand_dims(vis_mask,axis=-1)
    nms_ind = targets['nms_keep_ind']
    corners = targets['corners']
    probs = targets['probs']
    if  'refine_out' in targets:
        refined = targets['refine_out'].detach().cpu().numpy().squeeze(0)
        all_refineds = np.zeros((196,200,3))
        thereisrefined=True
    else:
        thereisrefined=False
        
    obj_estimated = len(probs) > 0
    if 'obj_img_centers' in targets:
    
        logging.error('OBJ IMG CENTER')
        img_centers = targets['obj_img_centers']
        if len(img_centers) > 0:
            
            overlayed = plot_obj_img_centers(images,img_centers)
            
            converted_img_centers = np.clip(targets['obj_converted_img_centers'],0,1)
                    
        
        else:
            obj_estimated = False
    else:
        overlayed = np.zeros((50,50),np.uint8)
            
#    logging.error('CONVERTED IMG CENTERS ' + str(converted_img_centers))
    
#    logging.error('VISUAL NMS IND' + str(nms_ind))
    all_converted_obj_centers = np.zeros((196,200))
    all_estimateds = np.zeros((196,200,3))
    all_estimateds_nms = np.zeros((196,200,3))
    
    if obj_estimated:
        nms_selected_boxes = corners[nms_ind]
        nms_selected_probs = probs[nms_ind]
        
    #    logging.error('VISUAL NMS PROBS ' + str(len(nms_selected_probs)))
        
        for k in range(len(nms_selected_probs)):
            # logging.error(str(nms_selected_probs[k]))
            if ((np.argmax(nms_selected_probs[k]) != (len(nms_selected_probs[k]) - 1)) &
                (np.argmax(nms_selected_probs[k]) != 3) &
                (np.argmax(nms_selected_probs[k]) != (len(nms_selected_probs[k]) - 1))):
                
                # logging.error('NMS VIS RES')
                obj_seg_out = np.zeros((196,200))    
                sel_class = np.argmax(nms_selected_probs[k])
                render_polygon(obj_seg_out, nms_selected_boxes[k], [196,200])
                
                obj_seg_out = np.stack([obj_seg_out,obj_seg_out,obj_seg_out],axis=-1)
                obj_seg_out = obj_seg_out * COLOR_LIST[sel_class]
                
                all_estimateds_nms= all_estimateds_nms + obj_seg_out
                
                # temp_img = Image.fromarray(np.uint8( obj_seg_out*255))
                # if name==None:
                #     temp_img.save(os.path.join(save_path,'nms_estimated_object_'+str(k)+'.jpg'))
                
                # else:
                #     temp_img.save(os.path.join(save_path,name + '_nms_estimated_object_'+str(k)+'.jpg'))
            
    all_estimateds_nms = np.clip(all_estimateds_nms,0,1)*vis_mask
    temp_img = Image.fromarray(np.uint8( all_estimateds_nms*255))
    
    
    if name==None:
        temp_img.save(os.path.join(save_path,'nms_all_estimated_objects.jpg'))
    
    else:
        temp_img.save(os.path.join(save_path,name + '_nms_all_estimated_objects.jpg'))
    
    for k in range(len(probs)):
        obj_seg_out = np.zeros((196,200))    
        sel_class = np.argmax(probs[k])
        render_polygon(obj_seg_out, corners[k], [196,200])
        
        obj_seg_out = np.stack([obj_seg_out,obj_seg_out,obj_seg_out],axis=-1)
        obj_seg_out = obj_seg_out * COLOR_LIST[sel_class]
        
        
        
        all_estimateds = all_estimateds + np.copy(obj_seg_out)
        
        
        if 'obj_img_centers' in targets:
            cur_center = converted_img_centers[k]
            cen_seg_out = np.zeros((196,200))    
            cen_seg_out[int(cur_center[1]*195), int(cur_center[0]*199)] = 1
            struct = np.ones((5,5)) > 0
        
            dilated = ndimage.binary_dilation(cen_seg_out, structure=struct)
            
            all_converted_obj_centers = all_converted_obj_centers + np.copy(dilated)
    if thereisrefined:   
        for k in range(len(refined)-1): 
            if ((k == 3) | (k == 4) ):
                continue
            cur_refined = refined[k] > object_thresh_list[k]
            
            cur_refined = np.stack([cur_refined,cur_refined,cur_refined],axis=-1)
            cur_refined = cur_refined * COLOR_LIST[k]
            all_refineds = all_refineds + np.copy(cur_refined)
            # cur_refined = refined[k] > 0.5
            temp_img = Image.fromarray(np.uint8( cur_refined*255))
            if name==None:
                temp_img.save(os.path.join(save_path,'refined_object_'+str(k)+'.jpg'))
            
            else:
                temp_img.save(os.path.join(save_path,name + '_refined_object_'+str(k)+'.jpg'))
         
        
        # temp_img = Image.fromarray(np.uint8( obj_seg_out*255))
        # if name==None:
        #     temp_img.save(os.path.join(save_path,'estimated_object_'+str(k)+'.jpg'))
        
        # else:
        #     temp_img.save(os.path.join(save_path,name + '_estimated_object_'+str(k)+'.jpg'))
     
        
        
    all_estimateds = np.clip(all_estimateds,0,1)*vis_mask
    temp_img = Image.fromarray(np.uint8( all_estimateds*255))
    over_img = Image.fromarray(np.uint8( overlayed))
    
    if 'obj_img_centers' in targets:
        if obj_estimated:
            er = np.copy(all_estimateds)
            er[...,0] = er[...,0]*(1 - all_converted_obj_centers) + all_converted_obj_centers
            er[...,1] = er[...,1]*(1 - all_converted_obj_centers) + all_converted_obj_centers
            er[...,2] = er[...,2]*(1 - all_converted_obj_centers) + all_converted_obj_centers
            
            er = np.clip(er,0,1)
        else:
            er = np.zeros_like(all_estimateds)
            
    else:
        er = np.zeros_like(all_estimateds)
        
    if thereisrefined:
        
        all_refineds = np.clip(all_refineds,0,1)*vis_mask
        all_refined_img = Image.fromarray(np.uint8( all_refineds*255))
        if name==None:
            all_refined_img.save(os.path.join(save_path,'all_refined_objects.jpg'))
            
        else:
            all_refined_img.save(os.path.join(save_path,name + '_all_refined_objects.jpg'))
         
        
    con_img = Image.fromarray(np.uint8( er*255))
    if name==None:
        temp_img.save(os.path.join(save_path,'all_estimated_objects.jpg'))
        over_img.save(os.path.join(save_path,'all_estimated_img_centers.jpg'))
        con_img.save(os.path.join(save_path,'all_estimated_converted_img_centers.jpg'))
    
    else:
        temp_img.save(os.path.join(save_path,name + '_all_estimated_objects.jpg'))
        over_img.save(os.path.join(save_path,name + '_all_estimated_img_centers.jpg'))
        over_img.save(os.path.join(save_path,name + '_all_estimated_converted_img_centers.jpg'))
 
    
    
def save_matched_objects(inter_dict,targets,target_ids,config,save_path):
    _, target_ids = target_ids
    
    
    orig_corners = targets['obj_corners'].cpu().numpy()
    orig_five_params = targets['obj_converted'].cpu().numpy()
    
    
    
    converted = five_params_to_corners(orig_five_params)
    
        
    orig_corners = orig_corners[target_ids]
    orig_five_params = orig_five_params[target_ids]
    converted = converted[target_ids]
    if len(orig_corners.shape) == 1:
        # dilated = np.expand_dims(dilated,axis=0)
        orig_corners = np.expand_dims(orig_corners,axis=0)
        orig_five_params = np.expand_dims(orig_five_params,axis=0)
        converted = np.expand_dims(converted,axis=0)
   
        
    inter_points = inter_dict['interpolated'].detach().cpu().numpy()
    probs = inter_dict['src_probs'].detach().cpu().numpy()
  
    
    if 'refine_out' in inter_dict:
        refined = inter_dict['refine_out'].detach().cpu().numpy().squeeze(0)
        for k in range(len(refined)):
            
            det = refined[k]
            
            temp_img = Image.fromarray(np.uint8( det*255))
            temp_img.save(os.path.join(save_path,'matched_refined_object_'+str(k)+'.jpg' ))
            
            
    for k in range(len(orig_corners)):
            
        
        
        sel_class = np.argmax(probs[k])
        
#        if sel_class == (probs[k].shape[-1] - 1):
#            continue
        
        '''
        PLOT ESTIMATED OBJECTS
        '''
        obj_seg_out = np.zeros((196,200))    
        cur_est = inter_points[k,...]
        render_polygon(obj_seg_out, cur_est, [196,200])
        
        obj_seg_out = np.stack([obj_seg_out,obj_seg_out,obj_seg_out],axis=-1)
        obj_seg_out = obj_seg_out * COLOR_LIST[sel_class]
        
        temp_img = Image.fromarray(np.uint8( obj_seg_out*255))
        temp_img.save(os.path.join(save_path,'matched_est_object_'+str(k)+'.jpg' ))
        
        '''
        PLOT REAL OBJECT CORNERS
        '''
        
        obj_seg_out = np.zeros((196,200))    
        cur_est = np.reshape(orig_corners[k][:-1],(4,2))
        gt_label = int(orig_corners[k,-1])
        render_polygon(obj_seg_out, cur_est, [196,200])
        
        obj_seg_out = np.stack([obj_seg_out,obj_seg_out,obj_seg_out],axis=-1)
        obj_seg_out = obj_seg_out * COLOR_LIST[gt_label]
        
        temp_img = Image.fromarray(np.uint8( obj_seg_out*255))
        temp_img.save(os.path.join(save_path,'matched_gt_object_'+str(k)+'.jpg' ))
        
        '''
        SANITY CHECK FOR PARAM CONVERSION
        '''
        
        # obj_seg_out = np.zeros((196,200))    
        # gt_label = int(orig_corners[k,-1])
        # render_polygon(obj_seg_out, converted[k], [196,200])
        
        # obj_seg_out = np.stack([obj_seg_out,obj_seg_out,obj_seg_out],axis=-1)
        # obj_seg_out = obj_seg_out * COLOR_LIST[gt_label]
        
        # temp_img = Image.fromarray(np.uint8( obj_seg_out*255))
        # temp_img.save(os.path.join(save_path,'matched_converted_object_'+str(k)+'.jpg' ))
        
def save_matched_results(inter_dict,targets,target_ids,config,save_path):
    _, target_ids = target_ids
    
    origs = targets['origs'].cpu().numpy()
    true_ends = targets['endpoints'].cpu().numpy()
    orig_coefs = targets['control_points'].cpu().numpy()
    
    coef_endpoints = get_endpoints_from_coeffs(orig_coefs)
    
    
    true_assoc = inter_dict['assoc_gt']
    # true_start = inter_dict['start_gt']
    # true_fin = inter_dict['fin_gt']
        
    origs = origs[target_ids]
    true_ends = true_ends[target_ids]
    coef_endpoints = coef_endpoints[target_ids]
    if len(origs.shape) == 2:
        # dilated = np.expand_dims(dilated,axis=0)
        origs = np.expand_dims(origs,axis=0)
        true_ends = np.expand_dims(true_ends,axis=0)
        coef_endpoints = np.expand_dims(coef_endpoints,axis=0)
        
    inter_points = inter_dict['interpolated'].detach().cpu().numpy()
    est_endpoints = inter_dict['endpoints'].detach().cpu().numpy()
    est_coeffs = inter_dict['src_boxes'].detach().cpu().numpy()
    
    est_assoc = inter_dict['assoc_est'] 
    # start_assoc = inter_dict['start_est'] 
    # fin_assoc = inter_dict['fin_est'] 
    # logging.error('SAVE MATCHED ASSOC ' + str(est_assoc))
    # logging.error('SAVE MATCHED GT ASSOC ' + str(true_assoc))
    
    est_coef_endpoints = np.concatenate([est_coeffs[:,0],est_coeffs[:,-1]],axis=-1)
    
    for k in range(len(origs)):
            
        cur_est = inter_points[k,...]
        cur_est = my_line_maker(cur_est)
    
        temp_img = Image.fromarray(np.uint8( add_endpoints_to_line(origs[k],true_ends[k])*255))
        temp_img.save(os.path.join(save_path,'gt_road_'+str(k)+'.jpg' ))
        
        temp_img = Image.fromarray(np.uint8( add_endpoints_to_line(cur_est,est_endpoints[k])*255))
        temp_img.save(os.path.join(save_path,'matched_road_'+str(k) +'.jpg' ))
        
        temp_img = Image.fromarray(np.uint8( add_endpoints_to_line(origs[k],coef_endpoints[k])*255))
        temp_img.save(os.path.join(save_path,'gt_coef_road_'+str(k)+'.jpg' ))
        
        temp_img = Image.fromarray(np.uint8( add_endpoints_to_line(cur_est,est_coef_endpoints[k])*255))
        temp_img.save(os.path.join(save_path,'matched_coef_road_'+str(k) +'.jpg' ))
            
        # for m in range(len(true_assoc[k])):
        #     if true_assoc[k][m] > 0.5:
        #         first_one = add_endpoints_to_line(origs[k],coef_endpoints[k])
        #         second_one = add_endpoints_to_line(origs[m],coef_endpoints[m])
        #         tot = np.clip(first_one + second_one,0,1)
        #         temp_img = Image.fromarray(np.uint8( tot*255))
        #         temp_img.save(os.path.join(save_path,'gt_assoc_from_'+str(k)+'to_'+str(m)+'.jpg' ))
        
      
        
        # for m in range(len(est_assoc[k])):
        #     if est_assoc[k][m] > 0:
        #         first_one = add_endpoints_to_line(cur_est,est_coef_endpoints[k])
        #         temp_est = inter_points[m,...]
        #         temp_est = my_line_maker(temp_est)
        #         second_one = add_endpoints_to_line(temp_est,est_coef_endpoints[m])
        #         tot = np.clip(first_one + second_one,0,1)
        #         temp_img = Image.fromarray(np.uint8( tot*255))
        #         temp_img.save(os.path.join(save_path,'matched_assoc_from_'+str(k)+'to_'+str(m)+'.jpg' ))
        

        
def save_results_train(image,out,out_objects, targets, static_inter_dict, object_inter_dict, static_target_ids, object_target_ids, config):
    
    image = process_image(image)
    
    os.makedirs(os.path.join(config.save_logdir,'train_images'),exist_ok=True)
    fileList = glob.glob(os.path.join(config.save_logdir,'train_images','*'))
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)
    # logging.error('LEN OF POST PROCESS ' + str(len(out)))
    
    for fr in range(len(image)):
        cur_img = Image.fromarray(np.uint8(image[fr,...]))
        cur_img.save(os.path.join(config.save_logdir,'train_images','image_'+str(fr)+'.jpg'))       
    
    
    save_matched_results(static_inter_dict,targets[0],static_target_ids,config,os.path.join(config.save_logdir,'train_images'))
    
#    if targets[0]['obj_exists']:
#        try:
#            save_matched_objects(object_inter_dict,targets[0],object_target_ids,config,os.path.join(config.save_logdir,'train_images'))
#        except Exception as e:
#            logging.error("PROBLEM IN TRAIN RESULTS MATCHED SAVE: " + str(e))
    
    try:  
        visual_masks_gt(np.uint8(image),targets,os.path.join(config.save_logdir,'train_images'))
    except Exception as e:
        logging.error("PROBLEM IN VISUAL GT TRAIN SAVE: " + str(e))

    visual_est(np.uint8(image),out,os.path.join(config.save_logdir,'train_images'))
    
    try:    
        visual_object_est(np.uint8(image),out_objects,os.path.join(config.save_logdir,'train_images'))

    except Exception as e:
        logging.error("PROBLEM IN VISUAL OBJECT TRAIN SAVE: " + str(e))

      
def save_results_eval(image,out,out_objects, targets, static_inter_dict, object_inter_dict, static_target_ids, object_target_ids, config):
#    
    image = process_image(image)
    
    base_path = os.path.join(config.save_logdir,'val_images',targets[0]['scene_name'],targets[0]['sample_token'])
#    
    os.makedirs(base_path,exist_ok=True)
    fileList = glob.glob(os.path.join(base_path,'*'))
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)
    # logging.error('LEN OF POST PROCESS ' + str(len(out)))
    
    # for fr in range(len(image)):
    #     cur_img = Image.fromarray(np.uint8(image[fr,...]))
    #     cur_img.save(os.path.join(base_path,'image.jpg'))       
    
    # if targets[0]['obj_exists']:
    #     try:
    #         save_matched_objects(object_inter_dict,targets[0],object_target_ids,config,base_path)
    #     except Exception as e:
    #         logging.error("PROBLEM IN MATCHED OBJECT VAL SAVE: " + str(e))
    
    # save_matched_results(static_inter_dict,targets[0],static_target_ids,config,base_path)
    # out = get_selected_estimates(out, thresh = 0.5)
    try:
        visual_masks_gt(np.uint8(image),targets,base_path,name='_')
    except Exception as e:
        logging.error("PROBLEM IN VISUAL MASKS GT VAL SAVE: " + str(e))
    visual_est(np.uint8(image),out,base_path,name='_')
    
    
    try:
        all_masks = targets[0]['mask'].cpu().numpy()[0]
        
        
        visual_object_est(np.uint8(image),out_objects,base_path,vis_mask = all_masks,name='_')
    except Exception as e:
            logging.error("PROBLEM IN VISUAL OBJECT VAL SAVE: " + str(e))
def img_saver(img,path):
    img = Image.fromarray(np.uint8(img))
    img.save(path)
    
    
    
def pinet_save_results_eval(image,out,coefs_list,boundaries_list,targets, config):
#    
    image = process_image(image)
    
    base_path = os.path.join(config.save_logdir,'val_images',targets[0]['scene_name'],targets[0]['sample_token'])
#    
    save_path = base_path
    name = '_'
    os.makedirs(base_path,exist_ok=True)
    fileList = glob.glob(os.path.join(base_path,'*'))
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)
    # logging.error('LEN OF POST PROCESS ' + str(len(out)))
    
#    for fr in range(len(image)):
#        cur_img = Image.fromarray(np.uint8(image[fr,...]))
#        cur_img.save(os.path.join(base_path,'image.jpg'))       
    
  
    try:
        visual_masks_gt(np.uint8(image),targets,base_path,name='_')
    except Exception as e:
        logging.error("PROBLEM IN VISUAL MASKS GT VAL SAVE: " + str(e))

    
    res_img = out[-1]

    res = Image.fromarray(res_img[0][...,[2,1,0]])
    res.save(os.path.join(base_path,'est.jpg'))
    
    
    
    res_interpolated_list = []
    res_coef_list = []
    coef_all_roads = np.zeros((196,200))
    temp_dict = dict()
    for k in range(len(coefs_list)):
                

        control = coefs_list[k]
        
        coef_endpoints = np.concatenate([control[0:1,:],control[-1:,:]],axis=0)
        
        interpolated = bezier.interpolate_bezier(control,100)
        
        res_interpolated_list.append(np.copy(interpolated))
#        
#        line2 = my_color_line_maker(interpolated,coef_endpoints,size=(196,200))
        line2 = my_line_maker(interpolated,size=(196,200))
        res_coef_list.append(line2)
        coef_all_roads = coef_all_roads + np.float32(line2)
    
    b=0
    temp_dict['coef_lines'] = res_coef_list
    
    temp_dict['interpolated_points'] = res_interpolated_list
    
    all_lanes = np.zeros((196,200))
    for k in range(len(res_coef_list)):
                
        lane_poly = convert_line_to_lane(coefs_list[k], lane_width = 3.5)
        can = np.zeros((196,200))

        render_polygon(can, lane_poly, shape=(196,200), value=1)
        
        all_lanes = all_lanes + can
        
        res_lane = Image.fromarray(np.uint8(255*can))
        
        res_coef = Image.fromarray(res_coef_list[k])
        if name==None:
            
            res_coef.save(os.path.join(save_path,'batch_'+str(b) + '_est_coef_interp_road_'+str(k)+'.jpg'))
            res_lane.save(os.path.join(save_path,'batch_'+str(b) + '_est_lane_'+str(k)+'.jpg'))
        
        else:
            
            res_coef.save(os.path.join(save_path,name + '_est_coef_interp_road_'+str(k)+'.jpg'))
            res_lane.save(os.path.join(save_path,name + '_est_lane_'+str(k)+'.jpg'))
        



    all_lanes = np.uint8(np.clip(all_lanes,0,1)*255)
    if name==None:
        
        temp_img = Image.fromarray(all_lanes)
        temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_est_all_lanes.jpg' ))       
        

        coef_all_roads = np.uint8(np.clip(coef_all_roads,0,1)*255)
        temp_img = Image.fromarray(coef_all_roads)
        temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_est_coef_all_roads.jpg' ))       
    else:
        temp_img = Image.fromarray(all_lanes)
        temp_img.save(os.path.join(save_path,name + '_est_all_lanes.jpg' ))    
      
        coef_all_roads = np.uint8(np.clip(coef_all_roads,0,1)*255)
        temp_img = Image.fromarray(coef_all_roads)
        temp_img.save(os.path.join(save_path,name + '_est_coef_all_roads.jpg' ))    


# cur_x = np.array([0.4,0.5,0.6])

# cur_y = np.array([0.9,0.8,0.7])
# calib = np.array([[1260,0,800],[0,1260,600],[0,0,1]])



def get_spline_for_pinet(x,y, calib, targets):
    num_boundaries = len(x)
    calib[0] *= 1600/800
    calib[1] *= 900/448
    spline_list = []
    points_list = []
    for k in range(num_boundaries):
        
        yx = zip(list(np.array(y[k])/256), list(np.array(x[k])/512))
        
        yx = list(yx)
        
        yx = sorted(yx, key=lambda t: t[0])
        x_sorted = [x for y, x in yx]
        y_sorted = [y for y, x in yx]
        
        cur_x = np.array(x_sorted)
        cur_y = np.array(y_sorted)
        
#        cur_x = np.array(x[k])/512
#        cur_y = np.array(y[k])/256
        # logging.error('CUR X ')
        # logging.error(str(cur_x))
        
        # logging.error('CUR Y ')
        # logging.error(str(cur_y))
        
        cam_height = 1.7
        # z_dist = 2.5
        
        f = calib[0,0]
        y_center = calib[1,-1]
        z = cam_height*f/abs(cur_y*900 - y_center)
        real_x = (z*1600*cur_x - calib[0,-1]*z)/f
        
        real_x = (real_x + 25)/50
        
        z = 1 - (z - 2.5)/50
        
        # logging.error('MAPPED X ')
        # logging.error(str(real_x))
        # logging.error('MAPPED Z ')
        # logging.error(str(z))
                
        invalid = (z > 1) | (z < 0) | (real_x > 1) | (real_x < 0)
        
        valid = np.logical_not(invalid)
        
        if np.sum(valid) < 5:
            continue
        
        real_x = real_x[valid]
        z = z[valid]
        
        
        points = np.stack([real_x, z],axis=-1)
        
        
        
        points_list.append(points)
        res = bezier.fit_bezier(points, 3)[0]  
        
        spline_list.append(res)
        
    center_lines = []
    res_interpolated_list=[]
    out=dict()
    for boun in range(len(points_list)-1):
        
        cur_boun = np.copy(points_list[boun])
        cur_dists = []
        dist_mat_list = []
        for other in range(boun + 1,len(points_list)):
            
            to_comp = np.copy(points_list[other])
            
            dist_mat = cdist(cur_boun, to_comp,'euclidean')
            
            dist_mat_list.append(np.copy(dist_mat))
            
            dist = np.min(dist_mat,axis=-1)
            cur_dists.append(np.copy(dist))
            
        dist_ar = np.stack(cur_dists,axis=0)
        mean_dist = np.mean(dist_ar,axis=-1)
        
        selected_pair = np.argmin(mean_dist)
        real_id = np.arange(boun + 1,len(points_list))[selected_pair]
        
        pair = points_list[real_id]
        
        my_dist = dist_mat_list[selected_pair]
        
        pointwise_min = np.argmin(my_dist,axis=-1)
        
        other_points = pair[pointwise_min]
        
        centerline = (other_points + cur_boun)/2
        
        yx = np.array(sorted(list(centerline), key=lambda t: t[1]))
#        x_sorted = [x for x, y in yx]
#        y_sorted = [y for x, y in yx]
        
#        logging.error(str(yx))
        res = bezier.fit_bezier(yx, 3)[0]  
        interpolated = bezier.interpolate_bezier(res,100)
        res_interpolated_list.append(np.copy(interpolated))
        center_lines.append(res)
    
    if len(center_lines) > 0:
    
        coefs = np.stack(center_lines,axis=0)
        out['boxes'] = coefs
        
    
                
        out['interpolated_points'] = res_interpolated_list
        
        loss = cdist(np.reshape(coefs,(-1,6)), targets['control_points'].cpu().numpy())
        
        i,j=linear_sum_assignment(loss)        
            
        out['src_boxes'] = coefs[i]
        out['target_ids'] = (0,j)
        out['src_ids'] = (0,i)
    
    else:
        out['boxes'] = []
        
    
                
        out['interpolated_points'] = []
        
        
        out['src_boxes'] = []
        out['target_ids'] = 0
        out['src_ids'] = 0
    
        
    return center_lines,spline_list, out
    
    
#
#x=[0,1,2,3,4,5]
#y=[0,1,2,3,4,5]
#fig, ax = plt.subplots(1, figsize=(12,9))
#ax.plot(x,y,color=(1.0, 0.4980392156862745, 0.054901960784313725))
