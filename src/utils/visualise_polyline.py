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
from torchvision import ops as torch_ops
from scipy.spatial.distance import cdist, directed_hausdorff
from baseline.Utils.utils import class_to_xy
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter
image_mean=[0.485, 0.456, 0.406]
image_std=[0.229, 0.224, 0.225]

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


COLOR_LIST = [np.array(colors.to_rgb(k)) for k in TEMP_COLOR_LIST]

COLOR_LIST.append(np.array([1,1,1]))

DETECTION_NAMES = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle',
                   ]

# 'traffic_cone', 'barrier'

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
    
    # res[int(endpoints[0,1]*size[0]),int(endpoints[0,0]*size[1]),0] = 1
    # res[int(endpoints[0,1]*size[0]),int(endpoints[0,0]*size[1]),1] = 0
    # res[int(endpoints[0,1]*size[0]),int(endpoints[0,0]*size[1]),2] = 0
    
    # res[int(endpoints[1,1]*size[0]),int(endpoints[1,0]*size[1]),0] = 0
    # res[int(endpoints[1,1]*size[0]),int(endpoints[1,0]*size[1]),1] = 0
    # res[int(endpoints[1,1]*size[0]),int(endpoints[1,0]*size[1]),2] = 1
    
    return np.uint8(255*res)



def my_float_line_maker(points,size=(196,200)):
    
    res = np.zeros(size)
    for k in range(len(points)):
        res[np.min([int(points[k][1]*size[0]),int(size[0]-1)]),np.min([int(points[k][0]*size[1]),int(size[1]-1)])] = 1
    return res


    
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
    
def hausdorff_match(out, target):
    
    # res_coef_list = out['interpolated_points']
    est_coefs = out['boxes']
#    logging.error('HAUS EST ' + str(est_coefs.shape))
    orig_coefs = np.copy(target['control_points'].cpu().numpy())
    
    
    orig_coefs = np.reshape(orig_coefs, (-1, int(orig_coefs.shape[-1]/2),2))
    
#    logging.error('HAUS EST ORIG ' + str(orig_coefs.shape))
    
    interpolated_origs = []
    
    for k in range(len(orig_coefs)):
        inter = bezier.interpolate_bezier(orig_coefs[k],100)
        interpolated_origs.append(np.copy(inter))
    
    if len(est_coefs) == 0:
        out['src_boxes'] = est_coefs
        out['target_ids'] = (0,0)
        out['src_ids'] = (0,0)
        return None,None, interpolated_origs, out
    
    
    dist_mat = np.mean(np.sum(np.square(np.expand_dims(est_coefs,axis=1) - np.expand_dims(orig_coefs,axis=0)),axis=-1),axis=-1)
    
    ind = np.argmin(dist_mat, axis=-1)
    min_vals = np.min(dist_mat,axis=-1)
    
    loss = cdist(np.reshape(est_coefs,(len(est_coefs),-1)), target['control_points'].cpu().numpy())
        
    i,j=linear_sum_assignment(loss)        
        
    out['src_boxes'] = est_coefs[i]
    out['target_ids'] = (0,j)
    out['src_ids'] = (0,i)

#    out['src_boxes'] = est_coefs
#    out['target_ids'] = (0,np.arange(len(orig_coefs)))
#    out['src_ids'] = (0,np.arange(len(orig_coefs)))
        
        
    return min_vals, ind, interpolated_origs , out  
    

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


def get_selected_polylines(targets, thresh = 0.5, training=True):
    
    temp_dict = dict()
    
    
    if 'pred_polys' in targets:
        poly_locs = targets['pred_polys'].detach().cpu().numpy()
               
       
        selecteds = np.ones((len(poly_locs))) > 0
            
        if np.sum(selecteds) > 0:
            
            
            
            
            poly_xy = class_to_xy(poly_locs, 50)
    
            coeffs = poly_xy/(49)
           
             
            detected_coeffs = coeffs[selecteds,...]
            
            
            all_roads = np.zeros((196,200,3),np.float32)
            coef_all_roads = np.zeros((196,200,3),np.float32)
 
            res_list = []
            res_coef_list=[]
            
            res_interpolated_list=[]
            # res_assoc_list = []
            
            for k in range(len(detected_coeffs)):
                
        
                control = detected_coeffs[k]
                
                coef_endpoints = np.concatenate([control[0:1,:],control[-1:,:]],axis=0)
                
                interpolated = bezier.interpolate_bezier(control,100)
                
                res_interpolated_list.append(np.copy(interpolated))
              
                line2 = my_color_line_maker(interpolated,coef_endpoints,size=(196,200))

                coef_all_roads = coef_all_roads + np.float32(line2)
            
            temp_dict['boxes'] = detected_coeffs
            # temp_dict['scores'] = detected_scores
            temp_dict['lines'] = res_list
            temp_dict['coef_lines'] = res_coef_list
            
            temp_dict['interpolated_points'] = res_interpolated_list
            
    #            temp_dict['all_roads'] = all_roads
            temp_dict['coef_all_roads'] = coef_all_roads
         
            
            temp_dict['assoc'] = targets['pred_assoc'].sigmoid().squeeze(0).detach().cpu().numpy()

            to_merge = {'assoc': temp_dict['assoc'],  'boxes':detected_coeffs}
            merged = get_merged_coeffs(to_merge)
            temp_dict['merged_coeffs'] = merged
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
          
            temp_dict['boxes'] = []
            temp_dict['lines'] = []
            temp_dict['coef_lines'] = []
            temp_dict['all_roads'] = []
            temp_dict['coef_all_roads'] = []
            temp_dict['labels'] = []
            temp_dict['assoc'] = []
            temp_dict['interpolated_points'] = []
            temp_dict['merged_interpolated_points'] = []
            temp_dict['merged_coeffs'] = []
    
    else:
        
        logging.error('DETECTED NOTHING')
        temp_dict['scores'] = []
      
        temp_dict['boxes'] = []
        temp_dict['lines'] = []
        temp_dict['coef_lines'] = []
        temp_dict['all_roads'] = []
        temp_dict['coef_all_roads'] = []
        temp_dict['labels'] = []
        temp_dict['assoc'] = []
        temp_dict['interpolated_points'] = []
        temp_dict['merged_interpolated_points'] = []
        temp_dict['merged_coeffs'] = []
        
 
        
    return temp_dict


def get_selected_inits(targets, thresh = 0.5):
    
    temp_dict = dict()
    
    
    # probs = np.squeeze(targets['init_point_detection_softmaxed'].detach().cpu().numpy(), axis=0)
    init_point_heatmap = targets['pred_init_point_softmaxed'].detach().cpu().numpy()
#    poly_locs = targets['pred_polys'].detach().cpu().numpy()
           
#    if not training:
    sth_exist = init_point_heatmap > thresh
    selecteds = np.where(init_point_heatmap > thresh)
#        
#    else:
#    selecteds = np.ones((len(poly_locs))) > 0
        
    if np.sum(sth_exist) > 0:
        
        init_row, init_col = selecteds
        
        to_send = np.stack([init_col,init_row],axis=-1)
        
    else:
        
        to_send=None
    
        
 
        
    return to_send
            
def get_vertices(adj):

    ins = []
    outs = []
    
    for k in range(len(adj)):
    #for k in range(7):
        for m in range(len(adj)):
        
            if adj[k,m] > 0.6:
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


def get_merged_network(targets):
    
    
    poly_locs = targets['pred_polys'].detach().cpu().numpy()
    logging.error('POLY SHAPE ' + str(poly_locs.shape[1]))
    assoc = np.squeeze(targets['pred_assoc'].sigmoid().detach().cpu().numpy(),axis=0) 
   
    poly_xy = class_to_xy(poly_locs, 50)
    
    coeffs = poly_xy/(49)
#          
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
    b=0
    
#    probs = np.squeeze(targets['init_point_detection_softmaxed'].detach().cpu().numpy(), axis=0)
    init_point_heatmap = targets['pred_init_point_softmaxed'].detach().cpu().numpy()
    poly_locs = targets['pred_polys'].detach().cpu().numpy()
   
    est_assoc = np.squeeze(targets['pred_assoc'].sigmoid().detach().cpu().numpy(),axis=0) 
  
    merged = get_merged_network(targets)
        
    
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

    poly_xy = class_to_xy(poly_locs, 50)
    
    coeffs = poly_xy/49
  
    all_init_points = np.squeeze(init_point_heatmap)
    coef_all_roads = np.zeros((196,200,3))
    for k in range(len(coeffs)):
        
        
        control = coeffs[k]
            
        coef_endpoints = np.concatenate([control[0:1,:],control[-1:,:]],axis=0)
        
        interpolated = bezier.interpolate_bezier(control,100)
        
        line2 = my_color_line_maker(interpolated,coef_endpoints,size=(196,200))
        
        
        coef_all_roads = coef_all_roads + np.float32(line2)
        
#        cur_init = init_point_heatmap[k]
#        cur_init = gaussian_filter(cur_init, sigma=1)
                
#                logging.error('GAUSSIAN FILTERED')
#        cur_init = cur_init/np.max(cur_init)
                
#        cur_init_point = Image.fromarray(np.uint8(255*cur_init))
        
        
#        for m in range(len(est_assoc[k])):
#            if est_assoc[k][m] > 0.5:
#                control = coeffs[m]
#            
#                coef_endpoints = np.concatenate([control[0:1,:],control[-1:,:]],axis=0)
#                
#                interpolated = bezier.interpolate_bezier(control,100)
#                
#                line3 = my_color_line_maker(interpolated,coef_endpoints,size=(196,200))
#                
#                tot = np.clip(line2 + line3,0,1)
#                temp_img = Image.fromarray(np.uint8( tot*255))
#                temp_img.save(os.path.join(save_path,'matched_assoc_from_'+str(k)+'to_'+str(m)+'.jpg' ))
#        
#        
#        
#        res = Image.fromarray(line2)
##        prob_img = Image.fromarray(np.uint8(255*np.ones((20,20))*probs[k,-1]))
#        if name==None:
#            res.save(os.path.join(save_path,'batch_'+str(b) + '_est_interp_road_'+str(k)+'.jpg'))
#            
#            res_lane.save(os.path.join(save_path,'batch_'+str(b) + '_est_lane_'+str(k)+'.jpg'))
##            prob_img.save(os.path.join(save_path,'batch_'+str(b) + '_prob_'+str(k)+'.jpg'))
#            
##            cur_init_point.save(os.path.join(save_path,'batch_'+str(b) + '_init_'+str(k)+'.jpg'))
#        
#        else:
#            res.save(os.path.join(save_path,name + '_est_interp_road_'+str(k)+'.jpg'))
#  
#            res_lane.save(os.path.join(save_path,name + '_est_lane_'+str(k)+'.jpg'))
##            prob_img.save(os.path.join(save_path,name + '_prob_'+str(k)+'.jpg'))
##            cur_init_point.save(os.path.join(save_path,name + '_init_'+str(k)+'.jpg'))
#
#    
        
        
#                plt.figure()
#                fig, ax = plt.subplots(1, figsize=(196,200))
##                axes = plt.gca()
#                ax.set_xlim([0,1])
#                ax.set_ylim([0,1])
#                plt.plot(interpolated[:,0],interpolated[:,1])
#                
#                plt.savefig(os.path.join(save_path,'batch_'+str(b) + '_est_interp_road_'+str(k)+'.png'), bbox_inches='tight', pad_inches=0.0)   
#                plt.close()  
    
        # merged, merged_coeffs = get_merged_lines(coeffs,assoc,k)
 
    if name==None:
        
        
        temp_img = Image.fromarray(np.uint8(all_init_points*255))
        temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_est_all_init_points.png' ))       
        
        
        coef_all_roads = np.uint8(np.clip(coef_all_roads,0,1)*255)
        temp_img = Image.fromarray(coef_all_roads)
        temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_est_coef_all_roads.png' ))       
    else:
        
        temp_img = Image.fromarray(np.uint8(all_init_points*255))
        temp_img.save(os.path.join(save_path,name + '_est_all_init_points.png' ))    
        
        
        coef_all_roads = np.uint8(np.clip(coef_all_roads,0,1)*255)
        temp_img = Image.fromarray(coef_all_roads)
        temp_img.save(os.path.join(save_path,name + '_est_coef_all_roads.png' ))  
        
        
    

def visual_masks_gt(images,targets,save_path,name=None):
 
    
    for b in range(len(targets)):
        true_assoc = targets[b]['con_matrix']
        
        init_point_matrix = targets[b]['init_point_matrix'].cpu().numpy()
        
        sort_index = targets[b]['sort_index'].cpu().numpy()
        
        img_centers = targets[b]['center_img']
        
        img_centers = img_centers.cpu().numpy()
        
        orig_img_centers = targets[b]['orig_center_img']
        
        orig_img_centers = orig_img_centers.cpu().numpy()
        
        roads = targets[b]['roads'].cpu().numpy()[sort_index]
        
        all_endpoints = targets[b]['endpoints'].cpu().numpy()
        
        origs = targets[b]['origs'].cpu().numpy()
        
        all_roads = np.zeros((img_centers.shape[0],img_centers.shape[1],3))
        coef_all_roads = np.zeros((img_centers.shape[0],img_centers.shape[1],3))
        
        grid_all_roads = np.zeros((img_centers.shape[0],img_centers.shape[1],3))

        all_init_points = np.clip(np.sum(init_point_matrix,axis=0),0,1)
        
        orig_coefs = targets[b]['control_points'].cpu().numpy()[sort_index]
        grid_coefs = targets[b]['grid_sorted_control_points'].cpu().numpy()
        
        grid_coefs = grid_coefs/49
        grid_endpoints = get_grid_endpoints_from_coeffs(grid_coefs)
        coef_endpoints = get_endpoints_from_coeffs(orig_coefs)
    
        all_masks = targets[b]['mask'].cpu().numpy()

        vis_img = Image.fromarray(np.uint8(255*all_masks[0]))
        
        all_lanes = np.zeros((196,200))
        
        van_occ = targets[b]['bev_mask'].cpu().numpy()[-1]
        van_occ = Image.fromarray(np.uint8(255*van_occ))
#        if name==None:
#            
#            
#            van_occ.save(os.path.join(save_path,'batch_'+str(b) + '_van_occ.jpg'))
#            
#            vis_img.save(os.path.join(save_path,'batch_'+str(b) + '_vis.jpg'))
#        else:
#            
#            van_occ.save(os.path.join(save_path,name + '_van_occ.jpg'))
#            
#            vis_img.save(os.path.join(save_path,name + '_vis.jpg'))
            
        for k in range(len(roads)):
#            for m in range(len(true_assoc[k])):
#                if true_assoc[k][m] > 0.5:
#                    first_one = add_endpoints_to_line(origs[k],coef_endpoints[k])
#                    second_one = add_endpoints_to_line(origs[m],coef_endpoints[m])
#                    tot = np.clip(first_one + second_one,0,1)
#                    temp_img = Image.fromarray(np.uint8( tot*255))
#                    temp_img.save(os.path.join(save_path,'gt_assoc_from_'+str(k)+'to_'+str(m)+'.jpg' ))
#            
#          
            
            
            cur_full = add_endpoints_to_line(np.float32(img_centers == roads[k]),all_endpoints[k])
            cur_coef_full = add_endpoints_to_line(np.float32(img_centers == roads[k]),coef_endpoints[k])
            
            
            cur_grids = grid_coefs[k]
            interpolated = bezier.interpolate_bezier(cur_grids,100)
            
            line = my_color_line_maker(interpolated,grid_endpoints[k],size=(196,200))
            
            grid_all_roads = grid_all_roads + np.copy(line)
            
            grid_img = Image.fromarray(line)
            
            
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
            
            
            cur_init = Image.fromarray(np.uint8(255*init_point_matrix[k])).resize((196,200))
            
            
#            if name==None:
##                orig_img.save(os.path.join(save_path,'batch_'+str(b) + '_gt_orig_road_'+str(k)+'.jpg' ))
#                grid_img.save(os.path.join(save_path,'batch_'+str(b) + '_grid_road_'+str(k)+'.jpg' ))
#                temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_gt_visible_road_'+str(k)+'.jpg' ))
#                temp_coef_img.save(os.path.join(save_path,'batch_'+str(b) + '_gt_coef_visible_road_'+str(k)+'.jpg' ))
#                
#                lane_img.save(os.path.join(save_path,'batch_'+str(b) + '_gt_lane_'+str(k)+'.jpg' ))
#                
#                cur_init.save(os.path.join(save_path,'batch_'+str(b) + '_gt_gauss_init_'+str(k)+'.jpg' ))
#                
#                
#            
#            else:
##                orig_img.save(os.path.join(save_path,name + '_gt_orig_road_'+str(k)+'.jpg' ))
#                
#                # orig_temp_img.save(os.path.join(save_path,name + '_gt_comp_road_'+str(k)+'.jpg' ))
#                grid_img.save(os.path.join(save_path,name + '_grid_road_'+str(k)+'.jpg' ))
#                temp_img.save(os.path.join(save_path,name + '_gt_visible_road_'+str(k)+'.jpg' ))
#                temp_coef_img.save(os.path.join(save_path,name + '_gt_coef_visible_road_'+str(k)+'.jpg' ))
#                lane_img.save(os.path.join(save_path,name + '_gt_lane_'+str(k)+'.jpg' ))
#
#                cur_init.save(os.path.join(save_path,name + '_gt_gauss_init_'+str(k)+'.jpg' ))


                
        all_roads = np.clip(all_roads,0,1)
        coef_all_roads = np.clip(coef_all_roads,0,1)
        
        all_lanes = np.clip(all_lanes,0,1)
        
        grid_all_roads = np.clip(grid_all_roads,0,1)
        
        
        if name==None:
            temp_img = Image.fromarray(np.uint8(grid_all_roads*255))
            temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_grid_all_roads.png' ))
            
            temp_img = Image.fromarray(np.uint8(all_roads*255))
            temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_gt_visible_all_roads.png' ))
            
            temp_img = Image.fromarray(np.uint8(coef_all_roads*255))
            temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_gt_coef_visible_all_roads.png' ))
            
            temp_img = Image.fromarray(np.uint8(all_init_points*255))
            temp_img.save(os.path.join(save_path,'batch_'+str(b) + '_gt_all_inits.png' ))
            
        else:
            temp_img = Image.fromarray(np.uint8(grid_all_roads*255))
            temp_img.save(os.path.join(save_path,name + '_grid_all_roads.png' ))
            
            
            temp_img = Image.fromarray(np.uint8(all_roads*255))
            temp_img.save(os.path.join(save_path,name + '_gt_visible_all_roads.png' ))
            
            temp_img = Image.fromarray(np.uint8(coef_all_roads*255))
            temp_img.save(os.path.join(save_path,name + '_gt_coef_visible_all_roads.png' ))
            
            temp_img = Image.fromarray(np.uint8(all_init_points*255))
            temp_img.save(os.path.join(save_path,name + '_gt_all_inits.png' ))
            
     
     
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
    
def get_grid_endpoints_from_coeffs(coeffs):
    
    start = coeffs[:,0,:]
    end = coeffs[:,-1,:]
    
    return np.concatenate([start,end],axis=-1)

def save_results_train(image,out,targets,  config):
    
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
    
    
    
    
    
    try:  
        visual_masks_gt(np.uint8(image),targets,os.path.join(config.save_logdir,'train_images'))
    except Exception as e:
        logging.error("PROBLEM IN VISUAL GT TRAIN SAVE: " + str(e))
    try:  
        visual_est(np.uint8(image),out,os.path.join(config.save_logdir,'train_images'))
    except Exception as e:
        logging.error("PROBLEM IN VISUAL EST TRAIN SAVE: " + str(e))
    
def save_results_eval(image,out,targets,  config):
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
    
    for fr in range(len(image)):
        cur_img = Image.fromarray(np.uint8(image[fr,...]))
        cur_img.save(os.path.join(base_path,'image.jpg'))       
    
 
    try:
        visual_masks_gt(np.uint8(image),targets,base_path,name='_')
    except Exception as e:
        logging.error("PROBLEM IN VISUAL MASKS GT VAL SAVE: " + str(e))
        
    try:     
        visual_est(np.uint8(image),out,base_path,name='_')
    except Exception as e:
        logging.error("PROBLEM IN VISUAL EST VAL SAVE: " + str(e))
        
    
   
def img_saver(img,path):
    img = Image.fromarray(np.uint8(img))
    img.save(path)
