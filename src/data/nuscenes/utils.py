import os
import numpy as np
from shapely import geometry, affinity
from pyquaternion import Quaternion
from shapely.geometry import Point
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.detection.constants import DETECTION_NAMES
from nuscenes.utils.data_classes import LidarPointCloud
import logging
from src.data.utils import transform_polygon, render_polygon, transform
import cv2
import time
CAMERA_NAMES = ['CAM_FRONT']
# CAMERA_NAMES = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
#                 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']

NUSCENES_CLASS_NAMES = [
    'drivable_area', 'ped_crossing', 'walkway', 'carpark', 'car', 'truck', 
    'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 
    'bicycle', 'traffic_cone', 'barrier'
]

# NUSCENES_CLASS_NAMES = [
#     'drivable_area', 'ped_crossing', 'walkway', 'carpark']

STATIC_CLASSES = ['drivable_area', 'ped_crossing', 'walkway', 'carpark_area']

LOCATIONS = ['boston-seaport', 'singapore-onenorth', 'singapore-queenstown',
             'singapore-hollandvillage']


def iterate_samples(nuscenes, start_token):
    sample_token = start_token
    while sample_token != '':
        sample = nuscenes.get('sample', sample_token)
        yield sample
        sample_token = sample['next']
    

def get_map_masks(nuscenes, map_data, sample_data, extents, resolution):

    # Render each layer sequentially
    layers = [get_layer_mask(nuscenes, polys, sample_data, extents, 
              resolution) for layer, polys in map_data.items()]

    return np.stack(layers, axis=0)


def get_layer_mask(nuscenes, polygons, sample_data, extents, resolution):

    # Get the 2D affine transform from bev coords to map coords
    tfm = get_sensor_transform(nuscenes, sample_data)[[0, 1, 3]][:, [0, 2, 3]]
    inv_tfm = np.linalg.inv(tfm)

    # Create a patch representing the birds-eye-view region in map coordinates
    map_patch = geometry.box(*extents)
    map_patch = transform_polygon(map_patch, tfm)

    # Initialise the map mask
    x1, z1, x2, z2 = extents
    mask = np.zeros((int((z2 - z1) / resolution), int((x2 - x1) / resolution)),
                    dtype=np.uint8)

    # Find all polygons which intersect with the area of interest
    for polygon in polygons.query(map_patch):

        polygon = polygon.intersection(map_patch)
        
        # Transform into map coordinates
        polygon = transform_polygon(polygon, inv_tfm)

        # Render the polygon to the mask
        render_shapely_polygon(mask, polygon, extents, resolution)
    
    return mask.astype(np.bool)



def get_object_masks(nuscenes, sample_data, extents, resolution):

    # Initialize object masks
    nclass = len(DETECTION_NAMES) + 1
    grid_width = int((extents[2] - extents[0]) / resolution)
    grid_height = int((extents[3] - extents[1]) / resolution)
    masks = np.zeros((nclass, grid_height, grid_width), dtype=np.uint8)

    # Get the 2D affine transform from bev coords to map coords
    tfm = get_sensor_transform(nuscenes, sample_data)[[0, 1, 3]][:, [0, 2, 3]]
    inv_tfm = np.linalg.inv(tfm)
    
    obj_list=[]
    
    for box in nuscenes.get_boxes(sample_data['token']):

        # Get the index of the class
        det_name = category_to_detection_name(box.name)
        if det_name not in DETECTION_NAMES:
            class_id = -1
        else:
            class_id = DETECTION_NAMES.index(det_name)
        
        # Get bounding box coordinates in the grid coordinate frame
            bbox = box.bottom_corners()[:2]
            local_bbox = np.dot(inv_tfm[:2, :2], bbox).T + inv_tfm[:2, 2]
            temp_ar = np.squeeze(np.zeros((9,1),np.float32))
            temp_ar[:8] = np.float32(local_bbox).flatten()
            temp_ar[-1] = class_id
            
            obj_list.append(np.copy(temp_ar))
            
            # Render the rotated bounding box to the mask
            render_polygon(masks[class_id], local_bbox, extents, resolution)
        
    return np.array(obj_list), masks
#
#def get_object_masks(nuscenes, sample_data, extents, resolution):
#
#    # Initialize object masks
#    nclass = len(DETECTION_NAMES) + 2
#    grid_width = int((extents[2] - extents[0]) / resolution)
#    grid_height = int((extents[3] - extents[1]) / resolution)
#    masks = np.zeros((nclass, grid_height, grid_width), dtype=np.uint8)
#
#    # Get the 2D affine transform from bev coords to map coords
#    tfm = get_sensor_transform(nuscenes, sample_data)[[0, 1, 3]][:, [0, 2, 3]]
#    inv_tfm = np.linalg.inv(tfm)
#
#    for box in nuscenes.get_boxes(sample_data['token']):
#
#        # Get the index of the class
#        det_name = category_to_detection_name(box.name)
#        if det_name not in DETECTION_NAMES:
#            class_id = -1
#        else:
#            class_id = DETECTION_NAMES.index(det_name)
#        
#        # Get bounding box coordinates in the grid coordinate frame
#        bbox = box.bottom_corners()[:2]
#        local_bbox = np.dot(inv_tfm[:2, :2], bbox).T + inv_tfm[:2, 2]
#
#        # Render the rotated bounding box to the mask
#        render_polygon(masks[class_id], local_bbox, extents, resolution)
#    
#    return masks.astype(np.bool)


def get_sensor_transform(nuscenes, sample_data):

    # Load sensor transform data
    sensor = nuscenes.get(
        'calibrated_sensor', sample_data['calibrated_sensor_token'])
    sensor_tfm = make_transform_matrix(sensor)

    # Load ego pose data
    pose = nuscenes.get('ego_pose', sample_data['ego_pose_token'])
    pose_tfm = make_transform_matrix(pose)

    return np.dot(pose_tfm, sensor_tfm)


def load_point_cloud(nuscenes, sample_data):

    # Load point cloud
    lidar_path = os.path.join(nuscenes.dataroot, sample_data['filename'])
    pcl = LidarPointCloud.from_file(lidar_path)
    return pcl.points[:3, :].T


def make_transform_matrix(record):
    """
    Create a 4x4 transform matrix from a calibrated_sensor or ego_pose record
    """
    my_transform = np.eye(4)
    my_transform[:3, :3] = Quaternion(record['rotation']).rotation_matrix
    my_transform[:3, 3] = np.array(record['translation'])
    return my_transform


def render_shapely_polygon(mask, polygon, extents, resolution):

    if polygon.geom_type == 'Polygon':

        # Render exteriors
        
#        logging.error('POLYGON ' + str(polygon.exterior.coords))
#        time.sleep(1)
        render_polygon(mask, polygon.exterior.coords, extents, resolution, 1)

        # Render interiors
        for hole in polygon.interiors:
            render_polygon(mask, hole.coords, extents, resolution, 0)
    
    # Handle the case of compound shapes
    else:
        for poly in polygon:
            render_shapely_polygon(mask, poly, extents, resolution)
            
            
def render_point(mask, polygon, extents, resolution,value):

        # Render exteriors
#    logging.error('POLYGON ' + str(polygon.coords))
#    logging.error('EXTENTS ' + str(np.array(extents[:2])))
    polygon = (polygon - np.array(extents[:2])) / resolution
    polygon = np.ascontiguousarray(polygon).round().astype(np.int32)
    cv2.fillConvexPoly(mask, polygon, value)
    render_polygon(mask, polygon.coords, extents, resolution, value)



#def render_centerlines(map_api,resolution_meters=0.5,                       
#                       figsize: Union[None, float, Tuple[float, float]] = None,
#                       bitmap: Optional[BitMap] = None) -> Tuple[Figure, Axes]:
#    """
#    Render the centerlines of all lanes and lane connectors.
#    :param resolution_meters: How finely to discretize the lane. Smaller values ensure curved
#        lanes are properly represented.
#    :param figsize: Size of the figure.
#    :param bitmap: Optional BitMap object to render below the other map layers.
#    """
#    # Discretize all lanes and lane connectors.
#    pose_lists = map_api.discretize_centerlines(resolution_meters)
#
#
#
#    # Render connectivity lines.
#    fig = plt.figure(figsize=self._get_figsize(figsize))
#    ax = fig.add_axes([0, 0, 1, 1 / self.canvas_aspect_ratio])
#
#    if bitmap is not None:
#        bitmap.render(self.map_api.canvas_edge, ax)
#
#    for pose_list in pose_lists:
#        if len(pose_list) > 0:
#            plt.plot(pose_list[:, 0], pose_list[:, 1])
#
#    return fig, ax


def view_points(points, view, normalize=True):
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.
    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False
    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]


    norm_const = points[2:3, :]
    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points,norm_const


# def check_visible(polygon, vis_mask):
    
def get_centerlines(nuscenes, new_ar, sample_data, extents, resolution, vis_mask, already_found=None):
    
    
    tfm = get_sensor_transform(nuscenes, sample_data)[[0, 1, 3]][:, [0, 2, 3]]
    
    my_thresh = 100
    
    my_x = tfm[0,-1]
    my_y = tfm[1,-1] 
    
    road_ind_ar = np.arange(len(new_ar))
    
    selecteds = np.abs(new_ar[:,:,0] - my_x) + np.abs(new_ar[:,:,1] - my_y) < my_thresh
    
    selected_lines = np.any(selecteds, axis=-1)
    
    logging.error('FOUND ' + str(np.sum(selected_lines)) + ' LINES')
    
    my_road_ar = road_ind_ar[selected_lines]
    
    my_lines = new_ar[selected_lines]
    my_sel_points = selecteds[selected_lines]
    
    inv_tfm = np.linalg.inv(tfm)
    
    # Create a patch representing the birds-eye-view region in map coordinates
    map_patch = geometry.box(*extents)
    map_patch = transform_polygon(map_patch, tfm)
    
    # Initialise the map mask
    x1, z1, x2, z2 = extents
    
    
    mask = np.zeros((int((z2 - z1) / resolution), int((x2 - x1) / resolution)),
                    dtype=np.uint16)
    
    # Find all polygons which intersect with the area of interest
    
    loc_array = np.zeros((len(new_ar),2,2),np.uint8)
    
    for road_id in range(len(my_lines)):
        
        cons_points = my_lines[road_id][my_sel_points[road_id]]
        
        cur_min = False
        cur_last = (None,None)
        
        for p in range(len(cons_points)):
            cur = cons_points[p][:2]
            cur_point = Point(cur)
            cont = map_patch.contains(cur_point)
            
            if cont:
    
    #            # Transform into map coordinates
                polygon = transform_polygon(cur_point, inv_tfm)
                if len(polygon.coords) > 0:
                    polygon = (polygon.coords[0]- np.array(extents[:2])) / resolution
                    polygon = np.ascontiguousarray(polygon).round().astype(np.int32)
                    if ((polygon[0] >= 0) & (polygon[1] >= 0)):
                        if ((polygon[0] < mask.shape[1]) & (polygon[1] < mask.shape[0])):
                            mask[polygon[1],polygon[0]] = my_road_ar[road_id] + 1
    #                       
                            if vis_mask[polygon[1],polygon[0]] > 0.5:
                                
                                if not cur_min:
    #                                
    #                                
                                    loc_array[my_road_ar[road_id],0,0] = np.int32(polygon[1])
                                    loc_array[my_road_ar[road_id],0,1] = np.int32(polygon[0]) 
                                    cur_min = True
    #                            
                                cur_last = (np.int32(polygon[1]),np.int32(polygon[0]))
    #    
        if cur_last[0] != None:
    #         
            loc_array[my_road_ar[road_id],1,0] = np.int32(cur_last[0])
            loc_array[my_road_ar[road_id],1,1] = np.int32(cur_last[1]) 
        else:
            loc_array[my_road_ar[road_id],1,0] = 255
            loc_array[my_road_ar[road_id],1,1] = 255 
            
        if not cur_min:
            loc_array[my_road_ar[road_id],0,0] = 255
            loc_array[my_road_ar[road_id],0,1] = 255
            
            
    return mask, loc_array
#
#def get_centerlines(nuscenes, centers, sample_data, extents, resolution, vis_mask, already_found=None):
#
#    # Get the 2D affine transform from bev coords to map coords
#    tfm = get_sensor_transform(nuscenes, sample_data)[[0, 1, 3]][:, [0, 2, 3]]
#    
#    tfm[1,-1] = tfm[1,-1]
#    
#    
#    inv_tfm = np.linalg.inv(tfm)
#
#    # Create a patch representing the birds-eye-view region in map coordinates
#    map_patch = geometry.box(*extents)
#    map_patch = transform_polygon(map_patch, tfm)
#
#    # Initialise the map mask
#    x1, z1, x2, z2 = extents
#    
#    
#    mask = np.zeros((int((z2 - z1) / resolution), int((x2 - x1) / resolution)),
#                    dtype=np.uint16)
#
#    # Find all polygons which intersect with the area of interest
#    
#    loc_array = np.zeros((len(centers),2,2),np.uint8)
#    
#    for road_id in range(len(centers)):
#
#        cur_min = False
#        cur_last = (None,None)
#        
#        for p in range(len(centers[road_id])):
#            cur = centers[road_id][p][:2]
#            cur_point = Point(cur)
#            cont = map_patch.contains(cur_point)
#            
#            if cont:
#
##            # Transform into map coordinates
#                polygon = transform_polygon(cur_point, inv_tfm)
#                if len(polygon.coords) > 0:
#                    polygon = (polygon.coords[0]- np.array(extents[:2])) / resolution
#                    polygon = np.ascontiguousarray(polygon).round().astype(np.int32)
#                    if ((polygon[0] >= 0) & (polygon[1] >= 0)):
#                        if ((polygon[0] < mask.shape[1]) & (polygon[1] < mask.shape[0])):
#                            mask[polygon[1],polygon[0]] = road_id + 1
#    #                       
#                            if vis_mask[polygon[1],polygon[0]] > 0.5:
#                                
#                                if not cur_min:
#    #                                
#    #                                
#                                    loc_array[road_id,0,0] = np.uint8(polygon[1])
#                                    loc_array[road_id,0,1] = np.uint8(polygon[0]) 
#                                    cur_min = True
#    #                            
#                                cur_last = (polygon[1],polygon[0])
##    
#        if cur_last[0] != None:
##         
#            loc_array[road_id,1,0] = np.uint8(cur_last[0])
#            loc_array[road_id,1,1] = np.uint8(cur_last[1]) 
#        else:
#            loc_array[road_id,1,0] = 255
#            loc_array[road_id,1,1] = 255 
#            
#        if not cur_min:
#            loc_array[road_id,0,0] = 255
#            loc_array[road_id,0,1] = 255
#            
#    return mask, loc_array


def get_moved_centerlines(nuscenes, centers, sample_data, extents, resolution, vis_mask, beta, already_found):

 
        
    start_point_base = 5000
    end_point_base = 10000
    
    # Get the 2D affine transform from bev coords to map coords
    tfm = get_sensor_transform(nuscenes, sample_data)[[0, 1, 3]][:, [0, 2, 3]]
    
    tfm[1,-1] = tfm[1,-1] - beta
    
    
    inv_tfm = np.linalg.inv(tfm)

    # Create a patch representing the birds-eye-view region in map coordinates
    map_patch = geometry.box(*extents)
    map_patch = transform_polygon(map_patch, tfm)

    # Initialise the map mask
    x1, z1, x2, z2 = extents
    
    
    mask = np.zeros((int((z2 - z1) / resolution), int((x2 - x1) / resolution)),
                    dtype=np.uint16)

    # Find all polygons which intersect with the area of interest
    
#    selected_roads=[]
#    found=False
    
    loc_array = np.zeros((len(centers),2,2),np.uint8)
    
    road_ids = list(np.int64(np.unique(already_found)[1:] - 1))
    
    for road_id in road_ids:
#        temp_mask = np.zeros((int((z2 - z1) / resolution), int((x2 - x1) / resolution)),
#                    dtype=np.uint16)
#        per_road_check = False
        
        cur_min = False
        cur_last = (None,None)
        
        for p in range(len(centers[road_id])):
            cur = centers[road_id][p][:2]
            cur_point = Point(cur)
            cont = map_patch.contains(cur_point)
            
            if cont:
#                logging.error('road_id ' + str(road_id))
#                logging.error('point ' + str(p))
#                found=True
#                break
#            # Transform into map coordinates
                polygon = transform_polygon(cur_point, inv_tfm)
                if len(polygon.coords) > 0:
                    polygon = (polygon.coords[0]- np.array(extents[:2])) / resolution
                    polygon = np.ascontiguousarray(polygon).round().astype(np.int32)
                    if ((polygon[0] >= 0) & (polygon[1] >= 0)):
                        if ((polygon[0] < mask.shape[1]) & (polygon[1] < mask.shape[0])):
                            mask[polygon[1],polygon[0]] = road_id + 1
    #                       
                            
                            if vis_mask[polygon[1],polygon[0]] > 0.5:
                                
    
    
                                if not cur_min:
    #                                
    #                                if mask[polygon[1],polygon[0]] > start_point_base:
    #                                    
    #                                    if mask[polygon[1],polygon[0]] > end_point_base:
    #                                        
    #                                        rel_id = mask[polygon[1],polygon[0]] - end_point_base - 1
    #                                        logging.error('START OF ROAD '+ str(road_id) + ' and END OF '+ str(rel_id))
    #                                
    #                                    else:
    #                                        rel_id = mask[polygon[1],polygon[0]] - start_point_base - 1
    #                                    
    #                                        logging.error('START OF ROAD '+ str(road_id) + ' and START OF '+ str(rel_id))
    #                                
                                    loc_array[road_id,0,0] = np.uint8(polygon[1])
                                    loc_array[road_id,0,1] = np.uint8(polygon[0]) 
                                    cur_min = True
    #                            
                                cur_last = (polygon[1],polygon[0])
#    
    
    
    
    #            # Render the polygon to the mask
#                logging.error('POLYGON ' + str(polygon.coords[1]))
#                logging.error('EXTENTS ' + str(np.array(extents[:2])))
#                polygon = (polygon - np.array(extents[:2])) / resolution
#                polygon = np.ascontiguousarray(polygon).round().astype(np.int32)
#                cv2.fillConvexPoly(mask, polygon, road_id)
#                render_point(mask, polygon, extents, resolution,road_id)
#        if found:
#            break
    
        if cur_last[0] != None:
#            if mask[cur_last[0],cur_last[1]] > 25000:
#                logging.error('ENDPOITNS COLLIDED IN ROAD '+ str(road_id) + ' and '+ str(np.float32(mask[cur_last[0],cur_last[1]])//10))
#            mask[cur_last[0],cur_last[1]] = (road_id + 1)*10 + 1
            loc_array[road_id,1,0] = np.uint8(cur_last[0])
            loc_array[road_id,1,1] = np.uint8(cur_last[1]) 
        else:
            loc_array[road_id,1,0] = 255
            loc_array[road_id,1,1] = 255 
            
        if not cur_min:
            loc_array[road_id,0,0] = 255
            loc_array[road_id,0,1] = 255
            
    return mask, loc_array

def zoom_augment_grids(image_shape, intrinsics, cs, beta):
    
    image = np.zeros(image_shape)
    
    col_ar2 = np.arange(image.shape[1])
    row_ar2 = np.arange(image.shape[0])
    
    
    mesh2_col, mesh2_row = np.meshgrid(col_ar2, row_ar2)
    
    write_col = np.copy(mesh2_col)
    write_row = np.copy(mesh2_row)
    
    col_ar1 = np.arange(image.shape[1])
    row_ar1 = np.arange(image.shape[0])
    
    
    mesh1_col, mesh1_row = np.meshgrid(col_ar1, row_ar1)
    
    x_center = intrinsics[0,-1]
    y_center = intrinsics[1,-1]
    f = intrinsics[0,0]
    Y = -cs[-1]
    
    for m in range(mesh1_row.shape[0]):
        for n in range(mesh1_row.shape[1]):
            
            write_col[m,n] = int((mesh2_col[m,n] - x_center)*f*Y/(f*Y - beta*mesh2_row[m,n] + beta*y_center) + x_center)
            
            write_row[m,n] = int(f*Y*(mesh2_row[m,n] - y_center)/(f*Y - beta*mesh2_row[m,n] + beta*y_center) + y_center)
    
    total_mask = np.ones_like(write_col)
    total_mask[write_col < 0] = 0
    total_mask[write_col > (image.shape[1]-1)] = 0
    
    total_mask[write_row < 0] = 0
    total_mask[write_row > (image.shape[0]-1)] = 0
    
    write_col[write_col < 0] = 0
    write_col[write_col > (image.shape[1]-1)] = 0
    
    write_row[write_row < 0] = 0
    write_row[write_row > (image.shape[0]-1)] = 0

    return write_row, write_col, total_mask