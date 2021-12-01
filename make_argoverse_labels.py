import os
import sys
import numpy as np
from PIL import Image

import logging
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_tracking_loader \
    import ArgoverseTrackingLoader
# from argoverse.utils.camera_stats import RING_CAMERA_LIST

from src.utils.configs import get_default_configuration
from src.data.utils import get_visible_mask, get_occlusion_mask, \
    encode_binary_labels
from src.data.argoverse.utils import get_object_masks, get_map_mask
from src.data.argoverse.splits import TRAIN_LOGS, VAL_LOGS
ALL_LOGS = TRAIN_LOGS + VAL_LOGS

RING_CAMERA_LIST = [
    "ring_front_center"

]

def process_split( map_data, config,loc_dict):

    # Create an Argoverse loader instance
    path = config.argo_log_root
    print("Loading Argoverse tracking data at " + path)
    loader = ArgoverseTrackingLoader(path)
    
    for scene in loader:

        process_scene( scene, map_data, config,loc_dict)


def process_scene(scene, map_data, config,loc_dict):

    logging.error("\n\n==> Processing scene: " + scene.current_log)


    # Iterate over each camera and each frame in the sequence
    for camera in RING_CAMERA_LIST:
        for frame in range(scene.num_lidar_frame):
            # progress.update(i)
           process_frame( scene, camera, frame, map_data, config, loc_dict)
         
            
def process_frame(scene, camera, frame, map_data, config, loc_dict):

    # Compute object masks
    obj_entries, obj_masks = get_object_masks(scene, camera, frame, config.map_extents,
                             config.map_resolution)
    
  
    masks = obj_masks
    # Ignore regions of the BEV which are outside the image
    calib = scene.get_calibration(camera)


    
    vis_mask = np.copy(np.uint8(np.flipud(get_visible_mask(np.copy(calib.K), np.copy(calib.camera_config.img_width),
                                       config.map_extents, config.map_resolution))))
    masks[-1] = vis_mask
    
    # Ignore regions of the BEV which are occluded (based on LiDAR data)
    lidar = scene.get_lidar(frame)
    cam_lidar = calib.project_ego_to_cam(lidar)
    occ_mask = np.uint8(~get_occlusion_mask(cam_lidar, config.map_extents, 
                                    config.map_resolution))
    
    masks[-1] = occ_mask*vis_mask
    # Encode masks as an integer bitmask
    labels = encode_binary_labels(masks)

    # Create a filename and directory
    timestamp = str(scene.image_timestamp_list_sync[camera][frame])
    output_path = os.path.join(config.argo_seg_label_root, 
                               scene.current_log, camera, 
                               f'{camera}_{timestamp}.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save encoded label file to disk
    Image.fromarray(labels.astype(np.int32), mode='I').save(output_path)
    

    loc_dict.append_sample( obj_entries, scene.current_log +'_' + str(timestamp))   
            

    

class obj_dict_class(object):
    def __init__(self):
        
        self.loc_dict = dict()
        
    def append_sample(self, ar, sample_token):
        
        
        self.loc_dict[sample_token] = ar

            
    def get_res(self):
        
        return self.loc_dict
    
    
if __name__ == '__main__':

    config = get_default_configuration()
    
    # Create an Argoverse map instance
    map_data = ArgoverseMap()
    obj_dict = obj_dict_class()
    process_split(map_data, config, obj_dict)

    np.save(config.argo_obj_dict_path, obj_dict.get_res())

