import numpy as np
from scipy.ndimage import affine_transform
from ..utils import render_polygon


# Define Argoverse-specific constants
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200

ARGOVERSE_CLASS_NAMES = [
    'drivable_area', 'vehicle', 'pedestrian', 'large_vehicle', 'bicycle', 'bus',
    'trailer', 'motorcycle',
]

ARGOVERSE_CLASS_MAPPING = {
    'VEHICLE' : 'vehicle',
    'PEDESTRIAN' : 'pedestrian',
    # 'ON_ROAD_OBSTACLE' : 'ignore',
    'LARGE_VEHICLE' : 'large_vehicle',
    'BICYCLE' : 'bicycle',
    'BICYCLIST' : 'bicycle',
    'BUS' : 'bus',
    # 'OTHER_MOVER' : 'ignore',
    'TRAILER' : 'trailer',
    'MOTORCYCLIST' : 'motorcycle',
    'MOPED' : 'motorcycle',
    'MOTORCYCLE' : 'motorcycle',
    # 'STROLLER' : 'ignore',
    'EMERGENCY_VEHICLE' : 'vehicle',
    # 'ANIMAL' : 'ignore',
}

def argoverse_name_to_class_id(name):
    if name in ARGOVERSE_CLASS_MAPPING:
        return ARGOVERSE_CLASS_NAMES.index(ARGOVERSE_CLASS_MAPPING[name])
    else:
        return -1



def get_object_masks(scene, camera, frame, extents, resolution):

    # Get the dimensions of the birds-eye-view mask
    x1, z1, x2, z2 = extents
    mask_width = int((x2 - x1) / resolution)
    mask_height = int((z2 - z1) / resolution)

    # Initialise masks
    num_class = len(ARGOVERSE_CLASS_NAMES)
    masks = np.zeros((num_class + 1, mask_height, mask_width), dtype=np.uint8)

    # Get calibration information
    calib = scene.get_calibration(camera)
    obj_list=[]
    # Iterate over objects in the scene
    for obj in scene.get_label_object(frame):

        # Get the bounding box and convert into camera coordinates
        bbox = obj.as_2d_bbox()[[0, 1, 3, 2]]
        cam_bbox = calib.project_ego_to_cam(bbox)[:, [0, 2]]
        class_id = argoverse_name_to_class_id(obj.label_class)
        temp_ar = np.squeeze(np.zeros((9,1),np.float32))
        temp_ar[:8] = np.float32(cam_bbox).flatten()
        temp_ar[-1] = class_id
        
        obj_list.append(np.copy(temp_ar))
        # Render the bounding box to the appropriate mask layer
        
        render_polygon(masks[class_id], cam_bbox, extents, resolution)
    
    return np.array(obj_list), masks.astype(np.bool)


def get_map_mask(scene, camera, frame, map_data, extents, resolution):

    # Get the dimensions of the birds-eye-view mask
    x1, z1, x2, z2 = extents
    mask_width = int((x2 - x1) / resolution)
    mask_height = int((z2 - z1) / resolution)

    # Get rasterised map
    city_mask, map_tfm = map_data.get_rasterized_driveable_area(scene.city_name)

    # Get 3D transform from camera to world coordinates
    extrinsic = scene.get_calibration(camera).extrinsic
    pose = scene.get_pose(frame).transform_matrix
    cam_to_world_tfm = np.matmul(pose, np.linalg.inv(extrinsic))

    # Get 2D affine transform from camera to map coordinates
    cam_to_map_tfm = np.matmul(map_tfm, cam_to_world_tfm[[0, 1, 3]])
    
    # Get 2D affine transform from BEV coords to map coords
    bev_to_cam_tfm = np.array([[resolution, 0, x1], 
                               [0, resolution, z1], 
                               [0, 0, 1]])
    bev_to_map_tfm = np.matmul(cam_to_map_tfm[:, [0, 2, 3]], bev_to_cam_tfm)

    # Warp map image to bev coordinate system
    mask = affine_transform(city_mask, bev_to_map_tfm[[1, 0]], 
                            output_shape=(mask_width, mask_height)).T
    return mask[None]



    

    
def get_centerlines(map_data,calib,pose, scene,extents, resolution, vis_mask):

    
    
    # Initialise the map mask
    x1, z1, x2, z2 = extents
    
    
    mask = np.zeros((int((z2 - z1) / resolution), int((x2 - x1) / resolution)),
                    dtype=np.uint16)


    camera_height = calib.T[1]
    city_mask, map_tfm = map_data.get_rasterized_driveable_area(scene.city_name)

    

    cor_ego_points = np.array([[25.       ,  0.       , 0]])

    temp_points = np.ones((1,4),np.float32)
    temp_points[:,:3] = cor_ego_points
    cor_map_points = np.matmul(pose,temp_points.T)
    
    lane_ids = map_data.get_lane_ids_in_xy_bbox(cor_map_points[0],cor_map_points[1],scene.city_name, 50)
    
    local_lane_centerlines = [map_data.get_lane_segment_centerline(lane_id, scene.city_name) for lane_id in lane_ids]
    
    for li in range(len(lane_ids)):
        cur_id = lane_ids[li]
        cur_line = local_lane_centerlines[li]
        
        cur_line = np.concatenate([cur_line,np.ones((10,1))],axis=-1)
    
        ego_line = np.matmul(np.linalg.inv(pose),cur_line.T)
        
        cam_line = calib.project_ego_to_cam(ego_line.T[:,:3])
    
    
    
        for k in range(len(cam_line)):
            cur_point = cam_line[k][:2]
            
            if ((cur_point[0] < z2) & (cur_point[0] > z1)):
                if ((cur_point[1] < x2) & (cur_point[1] > x1)):
            
                    cur_loc = [int(((z2-z1) - (cur_point[0] - z1))/resolution), int((cur_point[1] - x1)/resolution)]
      
            
   






