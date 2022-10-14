import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from nuscenes import NuScenes
from torchvision.transforms.functional import to_tensor
import cv2
from .utils import CAMERA_NAMES, NUSCENES_CLASS_NAMES, iterate_samples
from ..utils import decode_binary_labels
from src.utils import bezier
import random
from src.data.nuscenes import utils as nusc_utils
from src.data import utils as vis_utils

import numpy as np

import logging

from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndimage
import h5py

LOCATIONS = ['boston-seaport', 'singapore-onenorth', 'singapore-queenstown',
             'singapore-hollandvillage']

class NuScenesMapDataset(Dataset):

    def __init__(self, nuscenes, config,map_apis, apply_zoom_augment,
                 scene_names=None, pinet=False, work_objects=True):
        
        
        self.pinet=pinet
        
        self.work_objects = work_objects
        
        self.config = config
        self.nuscenes = nuscenes
        self.map_root = os.path.expandvars(config.nusc_root)
        
        
        self.resolution = self.config.map_resolution
        self.extents = self.config.map_extents
        
        
        self.line_label_root = self.config.line_label_root
        self.seg_label_root = self.config.seg_label_root
        
        self.n_control = config.n_control_points
        
        if self.pinet:
            self.image_size = [512,256]
        self.image_size = config.patch_size

        # Preload the list of tokens in the dataset
        self.get_tokens(scene_names)

        self.map_apis = map_apis
        
        self.lines_dict = {}
        
        self.loc_dict = np.load(config.loc_dict_path,allow_pickle=True)
        
        self.obj_dict = np.load(config.obj_dict_path,allow_pickle=True)
        
       
        
        self.zoom_sampling_dict = np.load(config.zoom_sampling_dict_path, allow_pickle=True)   
       
        self.apply_zoom_augment = apply_zoom_augment
 
        
        
        for location in LOCATIONS:
            
            scene_map_api = self.map_apis[location]
            all_lines = scene_map_api.lane + scene_map_api.lane_connector
            all_lines_tokens = []
            for lin in all_lines:
                all_lines_tokens.append(lin['token'])
            
            self.lines_dict[location] = all_lines_tokens
            
        for location in LOCATIONS:
            logging.error('N LINES IN '+ str(location) + ' '+str(len(self.lines_dict[location])))
            
        self.all_discretized_centers = {location : self.map_apis[location].discretize_centerlines(self.config.map_resolution)
                 for location in nusc_utils.LOCATIONS}
        
        # Allow PIL to load partially corrupted images
        # (otherwise training crashes at the most inconvenient possible times!)
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        self.struct = ndimage.generate_binary_structure(2, 1)
        
        self.augment_steps=[0.5,1,1.5,2]
        
    
    def minmax_normalize(self,img, norm_range=(0, 1), orig_range=(0, 255)):
        # range(0, 1)
        norm_img = (img - orig_range[0]) / (orig_range[1] - orig_range[0])
        # range(min_value, max_value)
        norm_img = norm_img * (norm_range[1] - norm_range[0]) + norm_range[0]
        return norm_img

    def get_tokens(self, scene_names=None):
        
        self.tokens = list()
        
        # all_files = glob.glob(os.path.join(self.line_label_root,'*png'))
        # self.tokens = [file.split('/')[-1][:-4] for file in all_files]
        # Iterate over scenes
        for scene in self.nuscenes.scene:
            
            # Ignore scenes which don't belong to the current split
            if scene_names is not None and scene['name'] not in scene_names:
                continue
             
            # Iterate over samples
            for sample in iterate_samples(self.nuscenes, 
                                          scene['first_sample_token']):
                
                # Iterate over cameras
                for camera in CAMERA_NAMES:
                    self.tokens.append(sample['data'][camera])
        
        return self.tokens


    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        
        try:
            token = self.tokens[index]
            
            
            if self.apply_zoom_augment:
                augment = np.random.rand() < self.config.zoom_augment_prob
                
                if augment:
                    sample_data = self.nuscenes.get('sample_data', token)
                    sample_token = sample_data['sample_token']

                    sample = self.nuscenes.get('sample', sample_token)
                    scene_token = sample['scene_token']

                    scene = self.nuscenes.get('scene', scene_token)
                    
                    log = self.nuscenes.get('log', scene['log_token'])
                    
                    selected_zoom_ind = random.randint(0,len(self.augment_steps)-1)
                    
                    beta = self.augment_steps[selected_zoom_ind]
                    
                    temp_ar = self.zoom_sampling_dict.item().get(log['location'])
                    
                    temp_ar = temp_ar[selected_zoom_ind]
                    
                    image = self.load_image(token, True, temp_ar)
                    
                else:
                    image = self.load_image(token, False, None)
                    beta=0
                
            else:
                image = self.load_image(token, False, None)
                beta=0
                augment=False
            
            calib = self.load_calib(token)
            obj_to_return, center_width_orient,con_matrix,endpoints,mask, bev_mask, orig_img_centers,\
            origs, scene_token,sample_token,to_return_centers, labels,roads,coeffs,\
            outgoings, incomings, singapore,problem, obj_exists = self.load_line_labels(token, augment, beta)
            
            if problem:
                return (None, dict(), True)
            
            
            if self.work_objects:
                #1.5 is camera height
                if len(center_width_orient) > 0:
                    my_calib = calib.cpu().numpy()
                    obj_center = center_width_orient[:,:2]
                    
                    obj_x = obj_center[:,0]*(self.config.map_extents[2]-self.config.map_extents[0]) + self.config.map_extents[0]
                    obj_y = obj_center[:,1]*(self.config.map_extents[3]-self.config.map_extents[1]) + self.config.map_extents[1]
                    
                    img_x = (obj_x*my_calib[0,0] + obj_y*my_calib[0,-1])/(obj_y + 0.0001)
                    img_y = (1.5*my_calib[1,1] + obj_y*my_calib[1,-1])/(obj_y + 0.0001)
                    
                    img_x = img_x / self.image_size[0]
                    img_y = img_y / self.image_size[1]
                    
                    to_keep = np.logical_not((img_x > 1) | (img_x < 0) | (img_y > 1) | (img_y < 0))
                    
                    img_centers = np.stack([img_x,img_y],axis=-1)
                    
                    if np.sum(to_keep) == 0:
                        img_centers = []
                        center_width_orient = []
                        obj_to_return = []
                        obj_exists = False
                    else:
                        img_centers = img_centers[to_keep]
                        center_width_orient = center_width_orient[to_keep]
                        obj_to_return = obj_to_return[to_keep]
                else:
                    img_centers = []
                    
            
            scene_name = self.nuscenes.get('scene', scene_token)['name']
            
            init_points = np.reshape(endpoints,(-1,2,2))[:,0]
            
            sorted_init_points, sort_index = self.get_sorted_init_points(init_points)
            
            temp_ar = np.zeros((len(sorted_init_points),2*self.config.polyrnn_feat_side,2*self.config.polyrnn_feat_side))
            for k in range(len(sorted_init_points)):
                temp_ar[k,int(np.clip(sorted_init_points[k,1]*2*self.config.polyrnn_feat_side,0,2*self.config.polyrnn_feat_side-1)),int(np.clip(sorted_init_points[k,0]*2*self.config.polyrnn_feat_side,0,2*self.config.polyrnn_feat_side-1))]=1
                
                temp_ar[k] = gaussian_filter(temp_ar[k], sigma=0.1)
                
                temp_ar[k] = temp_ar[k]/np.max(temp_ar[k])
                
            
            # sorted_points = np.copy(np.ascontiguousarray(coeffs[sort_index,:]))
            sorted_points = np.copy(coeffs)
            grid_sorted_points = np.reshape(sorted_points,(-1,self.n_control ,2))
            grid_sorted_points[...,0]= np.int32(grid_sorted_points[...,0]*(self.config.polyrnn_feat_side - 1))
            grid_sorted_points[...,1]= np.int32(grid_sorted_points[...,1]*(self.config.polyrnn_feat_side - 1))
            
            my_grid_points = np.copy(np.ascontiguousarray(grid_sorted_points))
            
           
            target = dict()
            

            target['calib'] = calib
            target['center_img'] = to_return_centers
            target['orig_center_img'] = orig_img_centers
            target['labels'] = labels.long()
            target['roads'] = torch.tensor(np.int64(roads)).long()
            target['control_points'] = torch.tensor(coeffs)
            target['con_matrix'] = torch.tensor(con_matrix)
            
            
            target['init_point_matrix'] = torch.tensor(np.copy(np.ascontiguousarray(temp_ar)))
            
            target['sorted_control_points'] = torch.tensor(sorted_points)
            
            target['grid_sorted_control_points'] = torch.tensor(my_grid_points)
            
            target['sort_index'] = torch.tensor(np.copy(np.ascontiguousarray(sort_index)))
            
            if self.work_objects:
                target['obj_corners'] = torch.tensor(obj_to_return).float()
                target['obj_converted'] = torch.tensor(center_width_orient).float()
                target['obj_exists'] = torch.tensor(obj_exists)
                
                
            else:
                target['obj_corners'] = torch.tensor(np.zeros((2,8))).float()
                target['obj_converted'] = torch.tensor(np.zeros((2,5))).float()
                target['obj_exists'] = torch.tensor(False)
                

            target['endpoints'] = torch.tensor(endpoints)

            target['origs'] = torch.tensor(origs)
            target['mask'] = mask
            target['bev_mask'] = bev_mask
#            target['static_mask'] = static_mask
            
            
            target['scene_token'] = scene_token
            target['sample_token'] = sample_token
            target['scene_name'] = scene_name
            target['outgoings'] = outgoings
            target['incomings'] = incomings
            target['left_traffic'] = torch.tensor(singapore)
            return (image, target, False)
        
        except Exception as e:
            logging.error('NUSC DATALOADER ' + str(e))
            
            return (None, dict(), True)
    

        
    def get_sorted_init_points(self, points):
        
        '''
        FROM BOTTOM UP AND RIGHT TO LEFT
        '''
        x = points[:,0]*self.config.rnn_size[1]
        y = points[:,1]*self.config.rnn_size[0]
        
        place = self.config.rnn_size[1]*x + y
        
        sort_ind = np.argsort(place)
        
        sort_ind = np.flip(sort_ind)
        
        return points[sort_ind,:], sort_ind
        
    
    def load_image(self, token, augment, temp_ar):

        # Load image as a PIL image
        image = Image.open(self.nuscenes.get_sample_data_path(token))
        image = np.array(image,np.float32)
        
        if augment:
            
            write_row = np.reshape(temp_ar[...,0],(image.shape[0],image.shape[1]))
            
            write_col = np.reshape(temp_ar[...,1],(image.shape[0],image.shape[1]))
            
            total_mask = np.reshape(temp_ar[...,2],(image.shape[0],image.shape[1]))
            
            
            
            sampled = image[write_row.flatten(),write_col.flatten(),:]
    
            sampled = np.reshape(sampled,[image.shape[0],image.shape[1],3])
            
            sampled = sampled*np.stack([total_mask,total_mask,total_mask],axis=-1) + image*(1-np.stack([total_mask,total_mask,total_mask],axis=-1))
            
            image=sampled
            
        if self.pinet:
            image = cv2.resize(image, (512,256), cv2.INTER_LINEAR)[...,[2,1,0]]
        else:
            image = cv2.resize(image, (self.config.patch_size[0], self.config.patch_size[1]), cv2.INTER_LINEAR)
            image = np.float32(image)
            image = self.minmax_normalize(image, norm_range=(-1, 1))
        
        # Convert to a torch tensor
        return to_tensor(image).float()
    

    def load_calib(self, token):

        # Load camera intrinsics matrix
        sample_data = self.nuscenes.get('sample_data', token)
        sensor = self.nuscenes.get(
            'calibrated_sensor', sample_data['calibrated_sensor_token'])
        intrinsics = torch.tensor(sensor['camera_intrinsic'])

        # Scale calibration matrix to account for image downsampling
        intrinsics[0] *= self.image_size[0] / sample_data['width']
        intrinsics[1] *= self.image_size[1] / sample_data['height']
        return intrinsics
    
    
    def  get_line_orientation(self,sample_token, road,img_centers,loc,vis_mask, custom_endpoints=None, augment=False):
        
        try:
            scene_map_api = self.map_apis[loc]
            all_lines_tokens = self.lines_dict[loc]
            
            other_lines_tokens = []
            for l in self.lines_dict.keys():
                other_lines_tokens = other_lines_tokens + self.lines_dict[l]
            
            if augment:
                all_ends = custom_endpoints
            else:
                all_ends = self.loc_dict.item().get(sample_token)
            
            my_row_id = img_centers.shape[0] - all_ends[road-1,:,0] - 1
            my_col_id = all_ends[road-1,:,1]
            
            my_rows = np.float32(my_row_id)/img_centers.shape[0]
            my_cols = np.float32(my_col_id)/img_centers.shape[1]
            
            
            
            token = all_lines_tokens[road-1]
     #            logging.error('TOKEN ' + token)
            outgoing_token = scene_map_api.get_outgoing_lane_ids(token)
            outgoing_id = []
            for tok in outgoing_token:
     #                logging.error('OUTGOING ' + tok)
     
                 if tok in all_lines_tokens:
     
                     outgoing_id.append(all_lines_tokens.index(tok))
     
                 else:
#                     logging.error('LINE ' + tok + ' not in lines')
                     if tok in other_lines_tokens:
                         logging.error('LINE ' + tok + ' is in other map')
                     else:
                         logging.error('LINE ' + tok + ' doesnt exist')
        
            incoming_token = scene_map_api.get_incoming_lane_ids(token)
            incoming_id = []
            for tok in incoming_token:
     #                logging.error('INCOMING ' + tok)
                 if tok in all_lines_tokens:
     
                     incoming_id.append(all_lines_tokens.index(tok))
     
                 else:
#                     logging.error('LINE ' + tok + ' not in lines')
                     if tok in other_lines_tokens:
                         logging.error('LINE ' + tok + ' is in other map')
                     else:
                         logging.error('LINE ' + tok + ' doesnt exist')
                
            return incoming_id, outgoing_id, np.stack([my_cols,my_rows],axis=-1),np.stack([my_col_id,my_row_id],axis=-1)
    
        except Exception as e:
            logging.error('ORIENT ' + str(e))
            
            return [],[],[],[]
        
    
    def get_connectivity(self,roads,outgoings, incomings):
        try:
            con_matrix = np.zeros((len(roads),len(roads)))

            for k in range(len(roads)):
                
                con_matrix[k,k] = 0
                outs = outgoings[k]

                for ou in outs:
                    
                    
                    sel = ou + 1
                    if sel in roads:
                        
                        ind = roads.index(sel)
                  
                    
                        con_matrix[k,ind] = 1
                    
            return con_matrix
        
        except Exception as e:
            logging.error('CONNECT ' + str(e))
            return None
    
    
    
    def get_object_params(self, token, vis_mask,beta,augment):
        resolution=self.resolution
        
        objs = self.obj_dict.item().get(token)
            
   
        to_return=[]
        center_width_orient=[]
        
        obj_exists = False
        for obj in objs:
            
        
            if obj[-1] > 7:
                continue


            reshaped = np.reshape(np.copy(obj)[:8],(4,2))
            reshaped[:,0] = (reshaped[:,0] - self.config.map_extents[0])/(self.config.map_extents[2]-self.config.map_extents[0])
            
            if augment:
                reshaped[:,1] = reshaped[:,1] - beta
            
            reshaped[:,1] =  (reshaped[:,1] - self.config.map_extents[1])/(self.config.map_extents[3]-self.config.map_extents[1])
            
            
            
            reshaped[:,1] = 1 - reshaped[:,1]
            
            coords = (np.clip(np.int64(reshaped[:,1]*(self.config.map_extents[3]-self.config.map_extents[1])/resolution),0,195),
                      np.clip(np.int64(reshaped[:,0]*(self.config.map_extents[2]-self.config.map_extents[0])/resolution),0,199))
         
            inside = False
            for k in range(4):
                inside = inside | ((vis_mask[coords[0][k], coords[1][k]] > 0.5) & 
                                   ((reshaped[k,1] >= 0) & (reshaped[k,1] <= 1)) & 
                                   ((reshaped[k,0] >= 0) & (reshaped[k,0] <= 1)))
                    
            if inside:
                
                # logging.error('INSIDE')
                
                res_ar = np.zeros(5)
                
                temp=np.squeeze(np.zeros((9,1),np.float32))
                temp[:8] = reshaped.flatten()
                temp[-1] = obj[-1]
                to_return.append(np.copy(temp))
                    
                reshaped[:,1] = 1 - reshaped[:,1]
                all_edges = np.zeros((4,2))
                for k in range(4):
                    first_corner = reshaped[k%4]
                    second_corner = reshaped[(k+1)%4]
                
                    all_edges[k,:]=np.copy(second_corner - first_corner)
                    
                all_lengths = np.sqrt(np.square(all_edges[:,0]) + np.square(all_edges[:,1]))
                long_side = np.argmax(all_lengths)
                
#                egim = np.sign(all_edges[long_side][1]/(all_edges[long_side][0] + 0.00001))*\
#                    np.abs(all_edges[long_side][1])/(all_lengths[long_side]  + 0.00001)
                my_abs_cos = np.abs(all_edges[long_side][0])/(all_lengths[long_side]  + 0.00001)
                my_sign = np.sign(all_edges[long_side][1]/(all_edges[long_side][0] + 0.00001))
                    
               
                angle = np.arccos(my_abs_cos*my_sign)
                
                center = np.mean(reshaped,axis=0)
                
                long_len = np.max(all_lengths)
                short_len = np.min(all_lengths)
                
                res_ar[:2] = center
#                res_ar[4] = my_abs_cos
#                res_ar[5] = my_sign
                res_ar[4] = angle
                res_ar[2] = long_len
                res_ar[3] = short_len
                
                center_width_orient.append(np.copy(res_ar))
                
                obj_exists = True
                
        return np.array(to_return), np.array(center_width_orient), obj_exists
                
                
    def load_seg_labels(self, token, scene_token):
        
        # Load label image as a torch tensor
#        label_path = os.path.join(self.vis_label_root, token + '.png')
#        
#        ar = Image.open(label_path)
##        
##        ar = self.vis_label_root[token]
#        
#        ar = np.flipud(ar)
#        
#        encoded_labels = torch.tensor(ar.copy()).long()
#
#        # Decode to binary labels
#        # num_class = len(NUSCENES_CLASS_NAMES)
#        vis_labels = decode_binary_labels(encoded_labels, 2)
#        # labels, mask = labels[:-1], ~labels[-1]
        
        
        bev_label = np.array(Image.open( os.path.join(self.seg_label_root,  
                                        token + '.png')),np.int32)
        
#        bev_label = np.array(self.seg_label_root[token],np.int32)
        
        
        bev_label = torch.tensor(np.flipud(bev_label).copy()).long()
        bev_label = decode_binary_labels(bev_label,self.config.nusc_num_object_class+1)
        
        # bev_label = torch.tensor(np.zeros((7,192,200))).long()
        
#        static_label = np.array(Image.open( os.path.join(self.static_label_root,  
#                                       token + '.png')),np.int32)
##        
#        
##        static_label = np.array(self.static_label_root[token] ,np.int32)
#        
#        static_label = torch.tensor(np.flipud(static_label).copy()).long()
#        static_label = decode_binary_labels(static_label,8)
#        
        
        # bev_label = torch.stack([static_label[0],static_label[0],static_label[0],static_label[0],static_label[0],static_label[0]],dim=0)
        
        return bev_label
    
    
    def load_line_labels(self, token, augment, beta):
        
        sample_data = self.nuscenes.get('sample_data', token)
        sample_token = sample_data['sample_token']
        sensor = self.nuscenes.get(
        'calibrated_sensor', sample_data['calibrated_sensor_token'])
        intrinsics = np.array(sensor['camera_intrinsic'])
    
    
        sample = self.nuscenes.get('sample', sample_token)
        scene_token = sample['scene_token']
        
        scene = self.nuscenes.get('scene', scene_token)
        
        log = self.nuscenes.get('log', scene['log_token'])
        
        # Load label image as a torch tensor
        label_path = os.path.join(self.line_label_root, token + '.png')

        orig_img_centers = cv2.imread(label_path, cv2.IMREAD_UNCHANGED )
        
        bev_labels = self.load_seg_labels(token, scene_token)
        
#        np_mask = vis_labels.numpy()
#        occ_mask = np_mask[1]
#        vis_mask = np_mask[0]
#        
        vis_mask = vis_utils.get_visible_mask(intrinsics, sample_data['width'], 
                                  self.config.map_extents, self.config.map_resolution)
        vis_mask = np.flipud(vis_mask)
        vis_labels = np.stack([vis_mask, vis_mask],axis=0)
        vis_labels = torch.tensor(vis_labels)
        
        if augment:
            
            orig_img_centers, trans_endpoints = nusc_utils.get_moved_centerlines(self.nuscenes, self.all_discretized_centers[log['location']], sample_data, self.extents, self.resolution, vis_mask,beta,orig_img_centers)
            orig_img_centers = np.flipud(orig_img_centers)
            
            # logging.error('AUGMENT PASSED ')
        if self.work_objects:
            obj_to_return, center_width_orient, obj_exists = self.get_object_params(token,vis_mask,beta,augment)
        else:
            obj_to_return, center_width_orient, obj_exists = None, None, None
        
        img_centers = orig_img_centers*np.uint16(vis_mask)
        
        roads = np.unique(img_centers)[1:]
    
        outgoings = []
        incomings = []
        coeffs_list = []
        to_remove=[]
        
        origs = []
#        starts = []
        endpoints = []
        
        singapore = 'singapore' in log['location']
        
        for k in range(len(roads)):
            
            sel = img_centers == roads[k]
            
            locs = np.where(sel)
            
            sorted_x = locs[1]/img_centers.shape[1]
            sorted_y = locs[0]/img_centers.shape[0]
           
            
            if len(sorted_x) < 10:
                to_remove.append(roads[k])   
                continue
            
            if augment:
                inc, out, endpoint, endpoint_id = self.get_line_orientation(token,roads[k],img_centers,log['location'],vis_mask,custom_endpoints=trans_endpoints, augment=True)
            else:
                inc, out, endpoint, endpoint_id = self.get_line_orientation(token,roads[k],img_centers,log['location'],vis_mask,custom_endpoints=None, augment=False)
            
            if len(endpoint) == 0:
                continue
            
            
            reshaped_endpoint = np.reshape(endpoint,(-1))
            endpoints.append(reshaped_endpoint)
            incomings.append(inc)
            outgoings.append(out)
            
            
            points = np.array(list(zip(sorted_x,sorted_y)))
            res = bezier.fit_bezier(points, self.n_control)[0]
            
            start_res = res[0]
            end_res = res[-1]
            
            first_diff = (np.sum(np.square(start_res - endpoint[0])) ) + (np.sum(np.square(end_res - endpoint[1])))
            second_diff = (np.sum(np.square(start_res - endpoint[1])) ) + (np.sum(np.square(end_res - endpoint[0])))
            if first_diff <= second_diff:
                fin_res = res
            else:
                fin_res = np.zeros_like(res)
                for m in range(len(res)):
                    fin_res[len(res) - m - 1] = res[m]
                    
            fin_res = np.clip(fin_res,0,1)
          
            coeffs_list.append(np.reshape(np.float32(fin_res),(-1)))
        
            sel = np.float32(sel)
            
            origs.append(sel)
#       
        if len(to_remove) > 0:
#           
            roads = list(roads)
            
            for k in to_remove:
                img_centers[img_centers == k] = 0
                
                roads.remove(k)
        
        else:
            roads = list(roads)
        
        if len(coeffs_list) == 0:
            return None,None,None,None,None,None,\
        None,None,\
        None,None,\
        None,None,None,\
        None,None,None,True,True, False

        
        con_matrix = self.get_connectivity(roads,outgoings, incomings)
#           
        to_return_centers = torch.tensor(np.int64(img_centers)).long()
        orig_img_centers = torch.tensor(np.int64(orig_img_centers)).long()
        labels = torch.ones(len(roads))
        
#        
        return obj_to_return, center_width_orient,con_matrix,np.array(endpoints),vis_labels,bev_labels,\
    orig_img_centers,np.stack(origs),\
    scene_token,sample_token,\
    to_return_centers,labels, roads,\
    np.array(coeffs_list), outgoings, incomings,singapore, False, obj_exists
