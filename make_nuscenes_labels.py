import os
import logging
import sys
import numpy as np
from PIL import Image

from collections import OrderedDict
import cv2
from shapely.strtree import STRtree
from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap


#sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))

from src.utils.configs import get_default_configuration
from src.data.utils import get_visible_mask, get_occlusion_mask, transform, \
    encode_binary_labels
from src.data.nuscenes import utils as nusc_utils


def process_scene(nuscenes, map_data,map_api, all_centers,scene, config, loc_dict, obj_dict, output_seg_root, output_line_root):
    logging.error('WORKING ON SCENE ' + str(scene['name']))
    # Get the map corresponding to the current sample data
    log = nuscenes.get('log', scene['log_token'])
    
   
    
    scene_map_data = map_data[log['location']]

#    scene_map_api = my_map_apis[log['location']]

    centers = all_centers[log['location']]
    # Iterate over samples
    first_sample_token = scene['first_sample_token']
    for sample in nusc_utils.iterate_samples(nuscenes, first_sample_token):
        process_sample(nuscenes, scene_map_data, sample, config, centers, loc_dict, obj_dict, output_seg_root, output_line_root)


def process_sample(nuscenes, map_data, sample, config,centers,loc_dict, obj_dict, output_seg_root, output_line_root ):

    # Load the lidar point cloud associated with this sample
    lidar_data = nuscenes.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_pcl = nusc_utils.load_point_cloud(nuscenes, lidar_data)

    # # Transform points into world coordinate system
    lidar_transform = nusc_utils.get_sensor_transform(nuscenes, lidar_data)
    lidar_pcl = transform(lidar_transform, lidar_pcl)
    # lidar_pcl = None
    # Iterate over sample data
    for camera in nusc_utils.CAMERA_NAMES:
        sample_data = nuscenes.get('sample_data', sample['data'][camera])
        obj_entries, loc_array = process_sample_data(nuscenes, map_data, sample_data, lidar_pcl, config,centers,  output_seg_root, output_line_root)
        loc_dict.append_sample( loc_array, sample_data['token'])     
        obj_dict.append_sample( obj_entries, sample_data['token'])     

def process_sample_data(nuscenes, map_data, sample_data, lidar, config,centers,  output_seg_root, output_line_root):
#    
#    map_masks = nusc_utils.get_map_masks(nuscenes, 
#                                         map_data, 
#                                         sample_data, 
#                                         config.map_extents, 
#                                         config.map_resolution)
#    
    # Render dynamic object masks
    obj_entries, obj_masks = nusc_utils.get_object_masks(nuscenes, 
                                            sample_data, 
                                            config.map_extents, 
                                            config.map_resolution)
#    masks = np.concatenate([map_masks, obj_masks], axis=0)
    masks = obj_masks
    # Ignore regions of the BEV which are outside the image
    sensor = nuscenes.get('calibrated_sensor', 
                          sample_data['calibrated_sensor_token'])
    intrinsics = np.array(sensor['camera_intrinsic'])
    
    vis_mask = get_visible_mask(intrinsics, sample_data['width'], 
                               config.map_extents, config.map_resolution)
    
    masks[-1] = vis_mask
    
    
    # Transform lidar points into camera coordinates
    cam_transform = nusc_utils.get_sensor_transform(nuscenes, sample_data)
    cam_points = transform(np.linalg.inv(cam_transform), lidar)
    
    occ_mask = np.uint8(~get_occlusion_mask(cam_points, config.map_extents,
                                    config.map_resolution))
    
    
    masks[-1] = occ_mask*vis_mask
    
    # Encode masks as integer bitmask
    labels = encode_binary_labels(masks)

    # Save outputs to disk
    output_path = os.path.join(output_seg_root,  
                               sample_data['token'] + '.png')
    Image.fromarray(labels.astype(np.int32), mode='I').save(output_path)
    
    img_centers, loc_array = nusc_utils.get_centerlines(nuscenes, centers, sample_data, config.map_extents, config.map_resolution, vis_mask)
    img_centers = np.flipud(img_centers)
  
    
    cv2.imwrite(os.path.join(output_line_root,sample_data['token'] + '.png'),img_centers)
    
    
    return obj_entries, loc_array


def load_map_data(dataroot, location):

    # Load the NuScenes map object
    nusc_map = NuScenesMap(dataroot, location)
    

    map_data = OrderedDict()
    for layer in nusc_utils.STATIC_CLASSES:
        
        # Retrieve all data associated with the current layer
        records = getattr(nusc_map, layer)
        polygons = list()

        # Drivable area records can contain multiple polygons
        if layer == 'drivable_area':
            for record in records:

                # Convert each entry in the record into a shapely object
                for token in record['polygon_tokens']:
                    poly = nusc_map.extract_polygon(token)
                    if poly.is_valid:
                        polygons.append(poly)
        else:
            for record in records:

                # Convert each entry in the record into a shapely object
                poly = nusc_map.extract_polygon(record['polygon_token'])
                if poly.is_valid:
                    polygons.append(poly)
        
        # Store as an R-Tree for fast intersection queries
        map_data[layer] = STRtree(polygons)
    
    return map_data



class loc_dict_class(object):
    def __init__(self,locations,all_centers,scenes,nuscenes):
        
        self.locations = locations
        self.all_centers = all_centers
        
        self.loc_dict = dict()
        
    def append_sample(self, ar, sample_token):
        
        
        self.loc_dict[sample_token] = ar
#        
    def get_res(self):
        
        return self.loc_dict
    
    
class obj_dict_class(object):
    def __init__(self,nuscenes):
        self.nuscenes = nuscenes
        self.loc_dict = dict()
        
    def append_sample(self, ar, sample_token):
        
        
        self.loc_dict[sample_token] = ar

            
    def get_res(self):
        
        return self.loc_dict
    
        

if __name__ == '__main__':

    # Load the default configuration
    config = get_default_configuration()

    # Load NuScenes dataset
    # dataroot = os.path.expandvars(config.dataroot)
    
        
    dataroot = config.nusc_root
    
#    dataroot=  '/srv/beegfs02/scratch/tracezuerich/data/datasets/nuScenes'
    nuscenes = NuScenes('v1.0-trainval', dataroot)

    # Preload NuScenes map data
    map_data = { location : load_map_data(dataroot, location) 
                 for location in nusc_utils.LOCATIONS }
    
    my_map_apis = { location : NuScenesMap(dataroot, location) 
             for location in nusc_utils.LOCATIONS }
    
    all_centers = {location : my_map_apis[location].discretize_centerlines(0.25)
             for location in nusc_utils.LOCATIONS}
    
    new_dict = dict()
    
    for my_key in all_centers.keys():
        logging.error('KEY : ' + str(my_key))
        centers = all_centers[my_key]
        my_max=0
        my_min_x=0
        my_min_y=0
        my_min=0
        max_length = 0
        for k in range(len(centers)):
            if np.max(centers[k]) > my_max:
                my_max = np.max(centers[k])
                
            if np.min(centers[k]) < my_min:
                my_min = np.min(centers[k])
                
            # if np.min(centers[k,:,1]) < my_min_y:
            #     my_min_y = np.min(centers[k,:,1])
                
            if len(centers[k])>max_length:
                max_length = len(centers[k])
        
        
        new_ar = np.ones((len(centers),max_length,3))
        new_ar[:,:,0] = my_min-10000
        new_ar[:,:,1] = my_min-10000
        
        for k in range(len(centers)):
            new_ar[k][:len(centers[k])] = np.array(centers[k])
            
        new_dict[str(my_key)] = np.copy(new_ar)
            
    
    loc_dict = loc_dict_class(nusc_utils.LOCATIONS,all_centers,nuscenes.scene,nuscenes)
    obj_dict = obj_dict_class(nuscenes)
   
    logging.error('NEW METHOD')
    # Create a directory for the generated labels
    # output_root = os.path.expandvars(config.nuscenes.label_root)
#    output_root = '/srv/beegfs02/scratch/tracezuerich/data/cany/lanelines/'
    
    
    output_seg_root = config.seg_label_root
    output_line_root = config.line_label_root
    
    logging.error('OUTPUT FOLDER SEG IS '+ output_seg_root)
    logging.error('OUTPUT FOLDER LINE IS '+ output_line_root)
    
  
    os.makedirs(output_seg_root, exist_ok=True)
    os.makedirs(output_line_root, exist_ok=True)
    # Iterate over NuScene scenes
    print("\nGenerating labels...")
    for scene in nuscenes.scene:
        process_scene(nuscenes, map_data,my_map_apis, new_dict, scene, config, loc_dict, obj_dict, output_seg_root, output_line_root)
        
    np.save(config.loc_dict_path, loc_dict.get_res())
    np.save(config.obj_dict_path, obj_dict.get_res())


#
#scene = nuscenes.scene[0]
#
#    
#log = nuscenes.get('log', scene['log_token'])
#    
#   
#
#scene_map_data = map_data[log['location']]
#
##    scene_map_api = my_map_apis[log['location']]
#
#centers = all_centers[log['location']]
## Iterate over samples
#first_sample_token = scene['first_sample_token']
#   
#sample = nuscenes.get('sample', first_sample_token)
#
## Load the lidar point cloud associated with this sample
#lidar_data = nuscenes.get('sample_data', sample['data']['LIDAR_TOP'])
#lidar_pcl = nusc_utils.load_point_cloud(nuscenes, lidar_data)
#
## # Transform points into world coordinate system
#lidar_transform = nusc_utils.get_sensor_transform(nuscenes, lidar_data)
#lidar_pcl = transform(lidar_transform, lidar_pcl)
#
#camera = 'CAM_FRONT'
#
#sample_data = nuscenes.get('sample_data', sample['data'][camera])
#
#my_max=0
#my_min=0
#max_length = 0
#for k in range(len(centers)):
#    if np.max(centers[k]) > my_max:
#        my_max = np.max(centers[k])
#        
#    if np.min(centers[k]) < my_min:
#        my_min = np.min(centers[k])
#        
#    if len(centers[k])>max_length:
#        max_length = len(centers[k])
#
#
#new_ar = np.ones((len(centers),max_length,3))
#new_ar[:,:,:3] = -100
#
#for k in range(len(centers)):
#    new_ar[k][:len(centers[k])] = np.array(centers[k])
#
#
#truth = cv2.imread('/srv/beegfs02/scratch/tracezuerich/data/cany/lanelines/lines/'+sample_data['token'] + '.png', cv2.IMREAD_UNCHANGED )
#      
#
#
#
#
#sensor = nuscenes.get('calibrated_sensor', 
#                          sample_data['calibrated_sensor_token'])
#intrinsics = np.array(sensor['camera_intrinsic'])
#
#mask = np.zeros((196,200),np.uint8)
#vis_mask = get_visible_mask(intrinsics, sample_data['width'], 
#                               config.map_extents, config.map_resolution)
#
#mask |= ~get_visible_mask(intrinsics, sample_data['width'], 
#                               config.map_extents, config.map_resolution)
#
## Transform lidar points into camera coordinates
#cam_transform = nusc_utils.get_sensor_transform(nuscenes, sample_data)
#cam_points = transform(np.linalg.inv(cam_transform), lidar_pcl)
#
#occ_mask = np.uint8(~get_occlusion_mask(cam_points, config.map_extents,
#                                config.map_resolution))
#
#
#mask |= ~occ_mask
#
#
#
#extents = config.map_extents
#resolution = config.map_resolution
#
#tfm = get_sensor_transform(nuscenes, sample_data)[[0, 1, 3]][:, [0, 2, 3]]
#
#my_thresh = 100
#
#my_x = tfm[0,-1]
#my_y = tfm[1,-1] 
#
#road_ind_ar = np.arange(len(centers))
#
#selecteds = np.abs(new_ar[:,:,0] - my_x) + np.abs(new_ar[:,:,1] - my_y) < my_thresh
#
#selected_lines = np.any(selecteds, axis=-1)
#
#my_road_ar = road_ind_ar[selected_lines]
#
#my_lines = new_ar[selected_lines]
#my_sel_points = selecteds[selected_lines]
#
#inv_tfm = np.linalg.inv(tfm)
#
## Create a patch representing the birds-eye-view region in map coordinates
#map_patch = geometry.box(*extents)
#map_patch = transform_polygon(map_patch, tfm)
#
## Initialise the map mask
#x1, z1, x2, z2 = extents
#
#
#mask = np.zeros((int((z2 - z1) / resolution), int((x2 - x1) / resolution)),
#                dtype=np.uint16)
#
## Find all polygons which intersect with the area of interest
#
#loc_array = np.zeros((len(centers),2,2),np.uint8)
#
#for road_id in range(len(my_lines)):
#    
#    cons_points = my_lines[road_id][my_sel_points[road_id]]
#    
#    cur_min = False
#    cur_last = (None,None)
#    
#    for p in range(len(cons_points)):
#        cur = cons_points[p][:2]
#        cur_point = Point(cur)
#        cont = map_patch.contains(cur_point)
#        
#        if cont:
#
##            # Transform into map coordinates
#            polygon = transform_polygon(cur_point, inv_tfm)
#            if len(polygon.coords) > 0:
#                polygon = (polygon.coords[0]- np.array(extents[:2])) / resolution
#                polygon = np.ascontiguousarray(polygon).round().astype(np.int32)
#                if ((polygon[0] >= 0) & (polygon[1] >= 0)):
#                    if ((polygon[0] < mask.shape[1]) & (polygon[1] < mask.shape[0])):
#                        mask[polygon[1],polygon[0]] = my_road_ar[road_id] + 1
##                       
#                        if vis_mask[polygon[1],polygon[0]] > 0.5:
#                            
#                            if not cur_min:
##                                
##                                
#                                loc_array[my_road_ar[road_id],0,0] = np.uint8(polygon[1])
#                                loc_array[my_road_ar[road_id],0,1] = np.uint8(polygon[0]) 
#                                cur_min = True
##                            
#                            cur_last = (polygon[1],polygon[0])
##    
#    if cur_last[0] != None:
##         
#        loc_array[my_road_ar[road_id],1,0] = np.uint8(cur_last[0])
#        loc_array[my_road_ar[road_id],1,1] = np.uint8(cur_last[1]) 
#    else:
#        loc_array[my_road_ar[road_id],1,0] = 255
#        loc_array[my_road_ar[road_id],1,1] = 255 
#        
#    if not cur_min:
#        loc_array[my_road_ar[road_id],0,0] = 255
#        loc_array[my_road_ar[road_id],0,1] = 255
#        
#        
#
#
#
#
#
#
## Get the 2D affine transform from bev coords to map coords
#tfm = get_sensor_transform(nuscenes, sample_data)[[0, 1, 3]][:, [0, 2, 3]]
#
#
#
#inv_tfm = np.linalg.inv(tfm)
#
## Create a patch representing the birds-eye-view region in map coordinates
#map_patch = geometry.box(*extents)
#map_patch = transform_polygon(map_patch, tfm)
#
## Initialise the map mask
#x1, z1, x2, z2 = extents
#
#
#mask = np.zeros((int((z2 - z1) / resolution), int((x2 - x1) / resolution)),
#                dtype=np.uint16)
#
## Find all polygons which intersect with the area of interest
#
#loc_array = np.zeros((len(centers),2,2),np.uint8)
#
#for road_id in range(len(centers)):
#
#    cur_min = False
#    cur_last = (None,None)
#    
#    for p in range(len(centers[road_id])):
#        cur = centers[road_id][p][:2]
#        cur_point = Point(cur)
#        cont = map_patch.contains(cur_point)
#        
#        if cont:
#
##            # Transform into map coordinates
#            polygon = transform_polygon(cur_point, inv_tfm)
#            if len(polygon.coords) > 0:
#                polygon = (polygon.coords[0]- np.array(extents[:2])) / resolution
#                polygon = np.ascontiguousarray(polygon).round().astype(np.int32)
#                if ((polygon[0] >= 0) & (polygon[1] >= 0)):
#                    if ((polygon[0] < mask.shape[1]) & (polygon[1] < mask.shape[0])):
#                        mask[polygon[1],polygon[0]] = road_id + 1
##                       
#                        if vis_mask[polygon[1],polygon[0]] > 0.5:
#                            
#                            if not cur_min:
##                                
##                                
#                                loc_array[road_id,0,0] = np.uint8(polygon[1])
#                                loc_array[road_id,0,1] = np.uint8(polygon[0]) 
#                                cur_min = True
##                            
#                            cur_last = (polygon[1],polygon[0])
##    
#    if cur_last[0] != None:
##         
#        loc_array[road_id,1,0] = np.uint8(cur_last[0])
#        loc_array[road_id,1,1] = np.uint8(cur_last[1]) 
#    else:
#        loc_array[road_id,1,0] = 255
#        loc_array[road_id,1,1] = 255 
#        
#    if not cur_min:
#        loc_array[road_id,0,0] = 255
#        loc_array[road_id,0,1] = 255
#            
#    return mask, loc_array
#
#
#
#
#

