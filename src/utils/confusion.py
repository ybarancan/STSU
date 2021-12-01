import torch
import src.utils.visualise as vis_tools
import logging
import numpy as np
import cv2
import scipy.ndimage as ndimage
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
#from mean_average_precision import MeanAveragePrecision
def render_polygon(mask, polygon, shape, value=1):
    
    to_mult = np.expand_dims(np.array([shape[1],shape[0]]),axis=0)
    polygon = polygon*to_mult
    polygon = np.ascontiguousarray(polygon).round().astype(np.int32)
    cv2.fillConvexPoly(mask, polygon, value)


class BinaryConfusionMatrix(object):

    def __init__(self, num_class, num_object_class):
        self.num_class = num_class
        
        self.num_object_class = num_object_class
        
      
        self.static_steps = [0.01,  0.02, 0.03 , 0.04, 0.05, 0.06, 0.07, 0.08 ,0.09 ,0.1]
        
        self.refine_thresholds = [0.5, 0.5, 0.4, 0.4, 0.4, 0.5, 0.4, 0.4]
        
        self.object_dil_sizes = [0,1,2,3]
        self.struct_steps = [1,2,3]
        
        
        self.structs = []
        for k in self.struct_steps:
            self.structs.append(np.ones((int(2*k+1),int(2*k+1))) > 0) 
        
        
        self.matched_gt = 0
        self.unmatched_gt = 0
        
        self.merged_matched_gt = 0
        self.merged_unmatched_gt = 0
        
        self.obj_intersect_limits = [0.1, 0.25, 0.5]
        
        self.object_tp_dict = dict()
        self.object_fp_dict = dict()
        self.object_fn_dict = dict()
        for n in range(len(self.object_dil_sizes)):
            self.object_tp_dict[str(self.object_dil_sizes[n])] = []
            self.object_fp_dict[str(self.object_dil_sizes[n])] = []
            self.object_fn_dict[str(self.object_dil_sizes[n])] = []
            for k in range(len(self.obj_intersect_limits)):
                
            
                self.object_tp_dict[str(self.object_dil_sizes[n])].append(np.zeros(num_object_class))
                self.object_fp_dict[str(self.object_dil_sizes[n])].append(np.zeros(num_object_class))
                self.object_fn_dict[str(self.object_dil_sizes[n])].append(np.zeros(num_object_class))
                
         
        self.seg_object_tp = np.zeros((num_object_class, len(self.object_dil_sizes)))
        self.seg_object_fp = np.zeros((num_object_class, len(self.object_dil_sizes)))
        self.seg_object_fn = np.zeros((num_object_class, len(self.object_dil_sizes)))
        
        self.refine_object_tp = np.zeros((num_object_class))
        self.refine_object_fp = np.zeros((num_object_class))
        self.refine_object_fn = np.zeros((num_object_class))
        
        self.argmax_refine_object_tp = np.zeros((num_object_class))
        self.argmax_refine_object_fp = np.zeros((num_object_class))
        self.argmax_refine_object_fn = np.zeros((num_object_class))
        
        self.object_cm = np.zeros((num_object_class+1, num_object_class+1))
        
        self.object_mAP_list = np.zeros((num_object_class))
#       
        self.static_pr_total_est = 0
        self.static_pr_total_gt = 0
        self.static_pr_tp = []
        self.static_pr_fn = []
        self.static_pr_fp = []

        for k in range(len(self.static_steps)):
            self.static_pr_tp.append(0)
            self.static_pr_fn.append(0)
            self.static_pr_fp.append(0)
            
            
        self.merged_static_pr_total_est = 0
        self.merged_static_pr_total_gt = 0
        self.merged_static_pr_tp = []
        self.merged_static_pr_fn = []
        self.merged_static_pr_fp = []

        for k in range(len(self.static_steps)):
            self.merged_static_pr_tp.append(0)
            self.merged_static_pr_fn.append(0)
            self.merged_static_pr_fp.append(0)
        
        self.assoc_tp = 0
        self.assoc_fn = 0
        self.assoc_fp = 0
        
      
        
        self.static_mse_list=[]
        
        self.static_metrics = dict()
        self.object_metrics = dict()
        
        self.scene_name = ''
        
        self.ap_list=np.zeros((8))
        self.obj_exist_list=np.zeros((8)) 

    
            

    def update(self, out, haus_gt, haus_idx, targets, static=True, pinet=False, polyline=False, assoc_thresh=0.4):
        
        occ_mask = targets['bev_mask'][-1].cpu().numpy()
        
        if static:
            
#       
            '''
            PRECISION-RECALL
            '''
            res_interpolated_list = out['interpolated_points']
            
            
            num_estimates = len(res_interpolated_list)
            num_gt = len(haus_gt)
            
            if num_estimates == 0:
                for k in range(len(self.static_steps)):
                
                    self.static_pr_fn[k] = np.copy(self.static_pr_fn[k]) + len(haus_gt)*len(haus_gt[0]) 
                    
                self.unmatched_gt += len(haus_gt)
            
            else:
                
                '''
                PRE ASSOC
                '''
                
                m_g = len(np.unique(np.array(haus_idx)))
                self.matched_gt += m_g
                self.unmatched_gt += len(haus_gt) - m_g
                
                for est_id in range(num_estimates):
                    cur_gt = haus_gt[haus_idx[est_id]]
                    
                    dis = cdist(res_interpolated_list[est_id],cur_gt,'euclidean')
                    
                    res_dis = np.min(dis,axis=-1)
                    gt_dis = np.min(dis,axis=0)
                    
                    self.static_pr_total_gt += len(cur_gt)
                    self.static_pr_total_est += len(res_interpolated_list[est_id])
                    
                    for k in range(len(self.static_steps)):
                    
                        self.static_pr_tp[k] = np.copy(self.static_pr_tp[k]) + np.sum(res_dis < self.static_steps[k]) 
                        self.static_pr_fn[k] = np.copy(self.static_pr_fn[k]) + np.sum(gt_dis > self.static_steps[k]) 
                        self.static_pr_fp[k] = np.copy(self.static_pr_fp[k]) + np.sum(res_dis > self.static_steps[k]) 
                
            '''
            MSE
            '''
            orig_coeffs = targets['control_points'].cpu().numpy()
            
            
#            
#          
            
            if not pinet:
                '''
                ASSOC LOSS
                '''
#                if polyline:
#                    assoc_est = out['pred_assoc']
#                else:
                assoc_est = out['assoc']
                gt_con_matrix = targets['con_matrix'].cpu().numpy()
                for est_id in range(num_estimates):
                    matched_gt = haus_idx[est_id]
                    cur_gt_assoc = gt_con_matrix[matched_gt]
                    cur_est_assoc = assoc_est[est_id]
                    
                    
                    
                    for m in range(len(cur_est_assoc)):
                        if cur_est_assoc[m] > assoc_thresh:
                            temp_id = haus_idx[m]
                            if temp_id == matched_gt:
                                self.assoc_tp += 1
                            elif cur_gt_assoc[temp_id] > 0.5: 
                                self.assoc_tp += 1
                            else:
                                self.assoc_fp += 1
                    
                for gt_id in range(len(gt_con_matrix)):
                    cur_gt_assoc = gt_con_matrix[gt_id]
                    
                    temp_mat = np.copy(cur_gt_assoc)
                    temp_mat = -temp_mat
                    
                    if not np.any(haus_idx == None):
                        
                    
                        if gt_id in haus_idx:
                            matched_ests = np.where(np.array(haus_idx)==gt_id)[0]
                            
                            
                            for m in range(len(cur_gt_assoc)):
                            
                                if cur_gt_assoc[m] > 0.5:
                                    
                                    if temp_mat[m] == -1:
                                        
                                        
                                        if m in haus_idx:
                                            other_ests = np.where(np.array(haus_idx)==m)[0]
                                             
                                            cur_est_assoc = assoc_est[matched_ests]
                                            
    #                                        temp_found = False
                                            for my_est in range(len(cur_est_assoc)):
                                                if np.any(cur_est_assoc[my_est][other_ests] > assoc_thresh):
    #                                                temp_found=True
                                                    temp_mat[m] = 1
                                                    break
                                                
                                            
                            self.assoc_fn += np.sum(temp_mat == -1)
                                            
                        else:
                            self.assoc_fn += np.sum(cur_gt_assoc)
                    else:
                        self.assoc_fn += np.sum(cur_gt_assoc)
                                
            
        else:
            
              
            
            corners = out['corners']
            probs = out['probs']
            
            est_class = np.argmax(probs[:,:-1],axis=-1)

            true_segs = targets['bev_mask'].cpu().numpy()
            
            true_objs = targets['obj_corners'].cpu().numpy()
            
            nms_ind = out['nms_keep_ind']
            
            if 'refine_out' in out:
                refined = out['refine_out'].detach().cpu().numpy()
                refined = np.squeeze(refined)
                refined_exists = True
            else:
                refined_exists = False
                
            
            if len(true_objs) > 0:
#              
                gt_converteds = []
                
                
                gt_classes = []
                for k in range(len(true_objs)):
                    
                    temp_gt_obj_seg_out = np.float32(np.zeros((196,200)))
                    
                    gt_classes.append(np.copy(true_objs[k][-1]))
                    
                    my_obj = true_objs[k][:-1]
                    my_obj = np.reshape(my_obj,(4,2))
                    
                    render_polygon(temp_gt_obj_seg_out, my_obj, [196,200])
                    gt_converteds.append(np.copy(temp_gt_obj_seg_out))
                
            
                gt_converteds = np.stack(gt_converteds,axis=0)
                
                gt_classes = np.array(gt_classes)    
                        
                gt_found = np.zeros((len(gt_classes),len(self.obj_intersect_limits)))
                
            # self.object_mAP_list
            else:
                gt_converteds = None
                
                gt_classes = None    
              
            
            '''
            WITH NO NMS
            '''
            for cl in range(self.num_object_class):
                
                gt_obj_seg_out = true_segs[cl]
                
                selected_est_id = est_class == cl
                
                est_obj_seg_out = np.float32(np.zeros_like(gt_obj_seg_out))
                
                
                if refined_exists:
                
                    cur_refined = refined[cl] > self.refine_thresholds[cl]
                    self.refine_object_tp[cl] += np.sum(np.float32(gt_obj_seg_out*cur_refined * occ_mask))
                    self.refine_object_fp[cl] += np.sum(np.float32((1-gt_obj_seg_out)*cur_refined  * occ_mask))
                    self.refine_object_fn[cl] += np.sum(np.float32(gt_obj_seg_out*(1-cur_refined)  * occ_mask))
                  
                    cur_refined = np.argmax(refined,axis=0) == cl
                    
                    self.argmax_refine_object_tp[cl] += np.sum(np.float32(gt_obj_seg_out*cur_refined * occ_mask))
                    self.argmax_refine_object_fp[cl] += np.sum(np.float32((1-gt_obj_seg_out)*cur_refined  * occ_mask))
                    self.argmax_refine_object_fn[cl] += np.sum(np.float32(gt_obj_seg_out*(1-cur_refined)  * occ_mask))
              
                    
                
                if np.sum(selected_est_id) > 0:
                
                    selected_est = corners[selected_est_id,:,:]
                    
                        
                    
                    est_obj_list=[]
                    
                 
                    
                    for k in range(len(selected_est)):
                        temp_est_obj_seg_out = np.float32(np.zeros_like(gt_obj_seg_out))
                        render_polygon(temp_est_obj_seg_out, selected_est[k], [196,200])
                        
                        est_obj_list.append(np.copy(temp_est_obj_seg_out))
                        est_obj_seg_out = np.copy(temp_est_obj_seg_out) + est_obj_seg_out
                        
                        
                            
                
                
                    if len(true_objs) > 0:
                        if np.any(gt_classes == cl):
                            selected_converteds = gt_converteds[gt_classes == cl]
                            
                            for n in range(len(self.object_dil_sizes)):
                                
                            
                                dilated_est = self.get_dilated_estimates(np.stack(est_obj_list, axis=0),self.object_dil_sizes[n] )
                                
                                expanded_est = np.expand_dims(dilated_est,axis=1)
                                expanded_gt = np.expand_dims(selected_converteds,axis=0)
                                
                                temp_miou = np.sum(expanded_est*expanded_gt,axis=(2,3))/np.sum(np.clip(expanded_est + expanded_gt,0,1),axis=(2,3))
                                
                                
                                row_ind, col_ind = linear_sum_assignment(1-temp_miou)
                                
                                # tot_tp = []
                                
                                for thre in range(len(self.obj_intersect_limits)):
                                    
                                    thre_tp=0
                                    for row in range(len(row_ind)):
                                        
                                        if temp_miou[row_ind[row],col_ind[row]] >= self.obj_intersect_limits[thre]:
                                            
                                            
                                            self.object_tp_dict[str(self.object_dil_sizes[n])][thre][cl] = self.object_tp_dict[str(self.object_dil_sizes[n])][thre][cl] + 1
                                            thre_tp = thre_tp + 1
                                        
                                        
                                    self.object_fp_dict[str(self.object_dil_sizes[n])][thre][cl] = self.object_fp_dict[str(self.object_dil_sizes[n])][thre][cl] + len(row_ind) - thre_tp
                                    self.object_fn_dict[str(self.object_dil_sizes[n])][thre][cl] = self.object_fn_dict[str(self.object_dil_sizes[n])][thre][cl] + len(row_ind) - thre_tp
                                    
                                if np.sum(gt_classes == cl) > len(selected_est):
                                    for thre in range(len(self.obj_intersect_limits)):
                                    
                                        self.object_fn_dict[str(self.object_dil_sizes[n])][thre][cl] = self.object_fn_dict[str(self.object_dil_sizes[n])][thre][cl] + np.sum(gt_classes == cl) - len(selected_est)
                                    
                                else:
                                    self.object_fp_dict[str(self.object_dil_sizes[n])][thre][cl] = self.object_fp_dict[str(self.object_dil_sizes[n])][thre][cl] + len(selected_est) - np.sum(gt_classes == cl) 
        
                            
                        
                        else:
                            for n in range(len(self.object_dil_sizes)):
                                for thre in range(len(self.obj_intersect_limits)):
                                    self.object_fp_dict[str(self.object_dil_sizes[n])][thre][cl] = self.object_fp_dict[str(self.object_dil_sizes[n])][thre][cl] + len(selected_est) 
                        
                    else:
                        for n in range(len(self.object_dil_sizes)):
                            for thre in range(len(self.obj_intersect_limits)):
                                self.object_fp_dict[str(self.object_dil_sizes[n])][thre][cl] = self.object_fp_dict[str(self.object_dil_sizes[n])][thre][cl]+ len(selected_est) 
                            
                      
                else:
                    if len(true_objs) > 0:
                        if np.any(gt_classes == cl):
                            for n in range(len(self.object_dil_sizes)):
                                for thre in range(len(self.obj_intersect_limits)):
                                    self.object_fn_dict[str(self.object_dil_sizes[n])][thre][cl] = self.object_fn_dict[str(self.object_dil_sizes[n])][thre][cl] + np.sum(gt_classes == cl)
                
                   
                    # if refined_exists:
                    #     self.refine_object_fn[cl] += np.sum(np.float32(gt_obj_seg_out*occ_mask))
                
                est_obj_seg_out = np.clip(est_obj_seg_out,0,1)    
                gt_obj_seg_out = np.clip(gt_obj_seg_out,0,1)   
                for n in range(len(self.object_dil_sizes)):
                    dilated = self.get_dilated_estimates(np.expand_dims(est_obj_seg_out,axis=0), self.object_dil_sizes[n])
                    
                    self.seg_object_tp[cl,n] += np.sum(np.float32(gt_obj_seg_out*dilated * occ_mask))
                    self.seg_object_fp[cl,n] += np.sum(np.float32((1-gt_obj_seg_out)*dilated  * occ_mask))
                    self.seg_object_fn[cl,n] += np.sum(np.float32(gt_obj_seg_out*(1-dilated)  * occ_mask))
              
                        
                
               
            
    def get_dilated_estimates(self, ar, dil_size):
        
        if dil_size == 0:
            return ar
        
        else:
            res = []
            for k in range(len(ar)):           
                dilated = ndimage.binary_dilation(ar[k], structure=self.structs[dil_size-1])  
                res.append(np.copy(dilated))
            return np.stack(res, axis=0)
            
    @property
    def get_res_dict(self):

        
        rec_list = []
        pre_list = []
        for k in range(len(self.static_steps)):
            
            self.static_metrics['precision_'+str(self.static_steps[k])] = self.static_pr_tp[k]/(self.static_pr_fp[k] + self.static_pr_tp[k] + 0.0001)
            self.static_metrics['recall_'+str(self.static_steps[k])] = self.static_pr_tp[k]/(self.static_pr_fn[k] + self.static_pr_tp[k] + 0.0001)
            pre_list.append(self.static_pr_tp[k]/(self.static_pr_fp[k] + self.static_pr_tp[k] + 0.0001))
            rec_list.append(self.static_pr_tp[k]/(self.static_pr_fn[k] + self.static_pr_tp[k] + 0.0001))
            
        self.static_metrics['mean_recall'] = np.mean(rec_list)
        self.static_metrics['mean_pre'] = np.mean(pre_list)
        
        
        self.static_metrics['mean_f_score'] = np.mean(pre_list)*np.mean(rec_list)*2/(np.mean(pre_list)+np.mean(rec_list)+ 0.001)

        
        self.static_metrics['mse'] = np.mean(self.static_mse_list)
  
        
        self.static_metrics['assoc_iou'] = self.assoc_tp/(self.assoc_tp + self.assoc_fn + self.assoc_fp + 0.0001)
        
        self.static_metrics['assoc_precision'] = self.assoc_tp/(self.assoc_tp +  self.assoc_fp + 0.0001)
        self.static_metrics['assoc_recall'] = self.assoc_tp/(self.assoc_tp + self.assoc_fn +  0.0001)
        
        
        self.static_metrics['assoc_f'] = self.static_metrics['assoc_precision']*self.static_metrics['assoc_recall']*2/(self.static_metrics['assoc_precision']+self.static_metrics['assoc_recall']+ 0.001)
        
        
        self.static_metrics['matched_gt'] = self.matched_gt
        self.static_metrics['unmatched_gt'] = self.unmatched_gt
        self.static_metrics['detection_ratio'] = self.matched_gt/(self.matched_gt+self.unmatched_gt+ 0.001)
        
        
        '''
        OBJECT 
        '''
        self.object_metrics['refined_miou'] = self.refine_object_tp/(self.refine_object_tp +
                                                        self.refine_object_fp +self.refine_object_fn+0.0001 )
        
        self.object_metrics['refined_precision'] = self.refine_object_tp/(self.refine_object_tp +
                                                        self.refine_object_fp +0.0001 )
        self.object_metrics['refined_recall'] = self.refine_object_tp/(self.refine_object_tp +
                                                        self.refine_object_fn+0.0001 )
        
        
        self.object_metrics['argmax_refined_miou'] = self.argmax_refine_object_tp/(self.argmax_refine_object_tp +
                                                        self.argmax_refine_object_fp +self.argmax_refine_object_fn+0.0001 )
        
        self.object_metrics['argmax_refined_precision'] = self.argmax_refine_object_tp/(self.argmax_refine_object_tp +
                                                        self.argmax_refine_object_fp +0.0001 )
        self.object_metrics['argmax_refined_recall'] = self.argmax_refine_object_tp/(self.argmax_refine_object_tp +
                                                        self.argmax_refine_object_fn+0.0001 )
        
        
        
        for n in range(len(self.object_dil_sizes)):
            for k in range(len(self.obj_intersect_limits)):
            
                self.object_metrics['object_instance_precision_'+str(self.object_dil_sizes[n])+'_'+str(self.obj_intersect_limits[k])] =\
                    self.object_tp_dict[str(self.object_dil_sizes[n])][k]/(self.object_tp_dict[str(self.object_dil_sizes[n])][k] + 
                                                                           self.object_fp_dict[str(self.object_dil_sizes[n])][k] + 0.001)
                self.object_metrics['object_instance_recall_'+str(self.object_dil_sizes[n])+'_'+str(self.obj_intersect_limits[k])] =\
                    self.object_tp_dict[str(self.object_dil_sizes[n])][k]/(self.object_tp_dict[str(self.object_dil_sizes[n])][k] + 
                                                                           self.object_fn_dict[str(self.object_dil_sizes[n])][k] + 0.001)
        
        self.object_metrics['object_seg_miou'] = self.seg_object_tp/(self.seg_object_tp +
                                                        self.seg_object_fp +self.seg_object_fn+0.0001 )
        
        self.object_metrics['object_seg_precision'] = self.seg_object_tp/(self.seg_object_tp +
                                                        self.seg_object_fp +0.0001 )
        self.object_metrics['object_seg_recall'] = self.seg_object_tp/(self.seg_object_tp +
                                                        self.seg_object_fn+0.0001 )
        
        self.object_metrics['mAP'] = self.ap_list/(self.obj_exist_list + 0.001)
     
        
    
        
        return self.static_metrics, self.object_metrics
        
                
    @property
    def static_mse(self):
#        return self.tp.float() / (self.tp + self.fn + self.fp).float()
        return np.mean(self.static_mse_list)
           
    
    @property
    def object_seg_iou(self):
#        return self.tp.float() / (self.tp + self.fn + self.fp).float()
        return self.object_tp / (self.object_tp + self.object_fn + self.object_fp + 0.0001)
    
    
    
    @property
    def object_class_iou(self):
#        return self.tp.float() / (self.tp + self.fn + self.fp).float()
        ious=[]
        for k in range(self.num_object_class):
            
            tp = self.object_cm[k,k]
            fp = np.sum(self.object_cm[:,k]) - tp
            fn = np.sum(self.object_cm[k,:]) - tp
            ious.append(tp/(tp + fp + fn))
            
        return np.array(ious)
    
    
    def reset(self):
        
        num_object_class = self.num_object_class
        
        self.matched_gt = 0
        self.unmatched_gt = 0
        
        self.merged_matched_gt = 0
        self.merged_unmatched_gt = 0
        
        self.obj_intersect_limits = [0.1, 0.25, 0.5]
        
        self.object_tp_dict = dict()
        self.object_fp_dict = dict()
        self.object_fn_dict = dict()
        for n in range(len(self.object_dil_sizes)):
            self.object_tp_dict[str(self.object_dil_sizes[n])] = []
            self.object_fp_dict[str(self.object_dil_sizes[n])] = []
            self.object_fn_dict[str(self.object_dil_sizes[n])] = []
            for k in range(len(self.obj_intersect_limits)):
                
            
                self.object_tp_dict[str(self.object_dil_sizes[n])].append(np.zeros(num_object_class))
                self.object_fp_dict[str(self.object_dil_sizes[n])].append(np.zeros(num_object_class))
                self.object_fn_dict[str(self.object_dil_sizes[n])].append(np.zeros(num_object_class))
                
         
        self.seg_object_tp = np.zeros((num_object_class, len(self.object_dil_sizes)))
        self.seg_object_fp = np.zeros((num_object_class, len(self.object_dil_sizes)))
        self.seg_object_fn = np.zeros((num_object_class, len(self.object_dil_sizes)))
        
        self.refine_object_tp = np.zeros((num_object_class))
        self.refine_object_fp = np.zeros((num_object_class))
        self.refine_object_fn = np.zeros((num_object_class))
        
        self.argmax_refine_object_tp = np.zeros((num_object_class))
        self.argmax_refine_object_fp = np.zeros((num_object_class))
        self.argmax_refine_object_fn = np.zeros((num_object_class))
        
        self.object_cm = np.zeros((num_object_class+1, num_object_class+1))
        
        self.object_mAP_list = np.zeros((num_object_class))
#       
        self.static_pr_total_est = 0
        self.static_pr_total_gt = 0
        self.static_pr_tp = []
        self.static_pr_fn = []
        self.static_pr_fp = []

        for k in range(len(self.static_steps)):
            self.static_pr_tp.append(0)
            self.static_pr_fn.append(0)
            self.static_pr_fp.append(0)
            
            
        self.merged_static_pr_total_est = 0
        self.merged_static_pr_total_gt = 0
        self.merged_static_pr_tp = []
        self.merged_static_pr_fn = []
        self.merged_static_pr_fp = []

        for k in range(len(self.static_steps)):
            self.merged_static_pr_tp.append(0)
            self.merged_static_pr_fn.append(0)
            self.merged_static_pr_fp.append(0)
        
        self.assoc_tp = 0
        self.assoc_fn = 0
        self.assoc_fp = 0
        
      
        
        self.static_mse_list=[]
        
        self.static_metrics = dict()
        self.object_metrics = dict()
        
        self.scene_name = ''
        
        self.ap_list=np.zeros((8))
        self.obj_exist_list=np.zeros((8)) 

    
    @property
    def mean_iou(self):
        # Only compute mean over classes with at least one ground truth
        valid = (self.tp + self.fn) > 0
        if not valid.any():
            return 0
        return float(self.iou[valid].mean())

    @property
    def dice(self):
        return 2 * self.tp.float() / (2 * self.tp + self.fp + self.fn).float()
    
    @property
    def macro_dice(self):
        valid = (self.tp + self.fn) > 0
        if not valid.any():
            return 0
        return float(self.dice[valid].mean())
    
    @property
    def precision(self):
        return self.tp.float() / (self.tp + self.fp).float()
    
    @property
    def recall(self):
        return self.tp.float() / (self.tp + self.fn).float()