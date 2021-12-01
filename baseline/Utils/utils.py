from skimage.io import imread
import skimage.color as color
import cv2
import torch
import os
import numpy as np
from scipy.ndimage.morphology import distance_transform_cdt
from baseline.Utils.poly_point_isect import isect_polygon__naive_check


def check_self_intersection(poly):
    # The polygon MUST be in float
    return isect_polygon__naive_check(poly)

def count_self_intersection(polys, grid_size):
    """
    :param polys: Nx1 poly 
    :return: number of polys that have self-intersection 
    """
    new_polys = []
    isects = []
    for poly in polys:
        poly = get_masked_poly(poly, grid_size)
        poly = class_to_xy(poly, grid_size).astype(np.float32)
        isects.append(check_self_intersection(poly.tolist()))

    return np.array(isects, dtype=np.float32)

#def create_folder(path):
#    if os.path.exists(path):
#        resp = raw_input('Path %s exists. Continue? [y/n]'%(path))
#        if resp == 'n' or resp == 'N':
#            raise RuntimeError()
#    
#    else:
#        os.system('mkdir -p %s'%(path))
#        print('Experiment folder created at: %s'%(path))

def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return nodes[np.argmin(dist_2)]


def closest_node_index(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)


def rgb_img_read(img_path):
    """
    Read image and always return it as a RGB image (3D vector with 3 channels).
    """
    img = imread(img_path)
    if len(img.shape) == 2:
        img = color.gray2rgb(img)

    # Deal with RGBA
    img = img[..., :3]

    if img.dtype == 'uint8':
        # [0,1] image
        img = img.astype(np.float32)/255

    return img

def get_full_mask_from_instance(min_area, instance):
    img_h, img_w = instance['img_height'], instance['img_width']

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for component in instance['components']:
        p = np.array(component['poly'], np.int)
        if component['area'] < min_area:
            continue
        else:
            draw_poly(mask, p)

    return mask

def get_full_mask_from_xy(poly, grid_size, patch_w, starting_point,
    out_h, out_w):
    poly = get_masked_poly(poly, grid_size)
    poly = class_to_xy(poly, grid_size)
    poly = poly0g_to_poly01(poly, grid_size)
    poly = poly * patch_w
    poly[:,0] += starting_point[0]
    poly[:,1] += starting_point[1]
    mask = np.zeros((out_h, out_w), dtype=np.uint8)
    poly = poly.astype(np.int)
    draw_poly(mask, poly)

    return mask, poly

def poly01_to_poly0g(poly, grid_size):
    """
    [0, 1] coordinates to [0, grid_size] coordinates

    Note: simplification is done at a reduced scale
    """
    poly = np.floor(poly * grid_size).astype(np.int32)
    poly = cv2.approxPolyDP(poly, 0, False)[:, 0, :]

    return poly

def poly0g_to_poly01(polygon, grid_side):
    """
    [0, grid_side] coordinates to [0, 1].
    Note: we add 0.5 to the vertices so that the points
    lie in the middle of the cell.
    """
    result = (polygon.astype(np.float32) + 0.5)/grid_side

    return result

def get_vertices_mask(poly, mask):
    """
    Generate a vertex mask
    """
    mask[poly[:, 1], poly[:, 0]] = 1.

    return mask


def get_edge_mask(poly, mask):
    """
    Generate edge mask
    """
    cv2.polylines(mask, [poly], True, [1])

    return mask

def class_to_grid(poly, out_tensor, grid_size):
    """
    NOTE: Torch function
    accepts out_tensor to do it inplace

    poly: [batch, ]
    out_tensor: [batch, 1, grid_size, grid_size]
    """
    out_tensor.zero_()
    # Remove old state of out_tensor

    b = 0
    for i in poly:
        if i < grid_size * grid_size:
            x = (i%grid_size).long()
            y = (i/grid_size).long()
            out_tensor[b,0,y,x] = 1
        b += 1

    return out_tensor

def xy_to_class(poly, grid_size):
    """
    NOTE: Torch function
    poly: [bs, time_steps, 2]
    
    Returns: [bs, time_steps] with class label
    for x,y location or EOS token
    """
    batch_size = poly.size(0)
    time_steps = poly.size(1)

    poly[:,:,1] *= grid_size
    poly = torch.sum(poly, dim=-1)

    poly[poly < 0] = grid_size**2
    # EOS token

    return poly

def prepare_ggnn_component(pred_polys, gt_polys, poly_ce_grid_size, poly_ggnn_grid_size, max_poly_len):

    batch_arr_fwd_poly = []
    batch_arr_mask = []
    batch_arr_local_prediction = []
    batch_array_feature_indexs = []
    batch_adj_matrix = []
    for i in range(pred_polys.shape[0]):


        curr_p = pred_polys[i]
        curr_p = get_masked_poly(curr_p, poly_ce_grid_size)
        curr_p = class_to_xy(curr_p, poly_ce_grid_size)

        corrected_poly = poly0g_to_poly01(curr_p, poly_ce_grid_size)
        # print curr_p
        # import ipdb; ipdb.set_trace()
        curr_g = gt_polys[i]
        gt_poly_112 = np.floor(curr_g * poly_ggnn_grid_size).astype(np.int32)

        enhanced_poly = []
        for i in range(len(corrected_poly)):
            if i < len(corrected_poly) - 1:
                enhanced_poly.append(corrected_poly[i])

                enhanced_poly.append(
                    np.array(
                        [(corrected_poly[i][0] + corrected_poly[i + 1][0])/2,
                        (corrected_poly[i][1] + corrected_poly[i + 1][1])/2])
                )
            else:
                enhanced_poly.append(corrected_poly[i])
                enhanced_poly.append(
                    np.array(
                    [(corrected_poly[i][0] + corrected_poly[0][0])/2,
                                               (corrected_poly[i][1] + corrected_poly[0][1])/2])
                )

        fwd_poly = np.floor(np.array(enhanced_poly) * poly_ggnn_grid_size).astype(np.int32)
        feature_indexs = poly0g_to_index(fwd_poly, poly_ggnn_grid_size)


        delta_x =[]
        delta_y = []
        for idx in range(len(enhanced_poly)):
            if idx % 2 == 0:
                curr_poly = fwd_poly[idx]

                corresponding_node = closest_node(curr_poly, gt_poly_112)

                delta_x.append(corresponding_node[0] - curr_poly[0])
                delta_y.append(corresponding_node[1] - curr_poly[1])
            else:
                if idx < len(enhanced_poly) -1 :

                    curr_poly = fwd_poly[idx]
                    curr_point0 = fwd_poly[idx - 1]
                    corresponding_node0_index = closest_node_index(curr_point0, gt_poly_112)
                    curr_point1 = fwd_poly[idx + 1]
                    corresponding_node1_index = closest_node_index(curr_point1, gt_poly_112)
                    if corresponding_node1_index - corresponding_node0_index > 1:
                        corresponding_node =  closest_node(curr_poly, gt_poly_112[corresponding_node0_index+1: corresponding_node1_index])
                        delta_x.append(corresponding_node[0] - curr_poly[0])
                        delta_y.append(corresponding_node[1] - curr_poly[1])
                    else:
                        delta_x.append((int(gt_poly_112[corresponding_node1_index][0] +  gt_poly_112[corresponding_node0_index][0])/2) - curr_poly[0])
                        delta_y.append((int(gt_poly_112[corresponding_node1_index][1] + gt_poly_112[corresponding_node0_index][1]) / 2) - curr_poly[1])
                else:

                    curr_poly = fwd_poly[idx]
                    curr_point0 = fwd_poly[idx - 1]

                    corresponding_node0_index = closest_node_index(curr_point0, gt_poly_112)

                    curr_point1 = fwd_poly[0]

                    corresponding_node1_index = closest_node_index(curr_point1, gt_poly_112)

                    if corresponding_node1_index - corresponding_node0_index > 1:
                        corresponding_node = closest_node(curr_poly, gt_poly_112[corresponding_node0_index+1: corresponding_node1_index])
                        delta_x.append(corresponding_node[0] - curr_poly[0])
                        delta_y.append(corresponding_node[1] - curr_poly[1])
                    else:
                        delta_x.append((int(gt_poly_112[corresponding_node1_index][0] +  gt_poly_112[corresponding_node0_index][0])/2) - curr_poly[0])
                        delta_y.append((int(gt_poly_112[corresponding_node1_index][1] + gt_poly_112[corresponding_node0_index][1]) / 2) - curr_poly[1])

        local_prediction = []
        for x,y in zip(delta_x, delta_y):

            local_x = 7+x
            if local_x>=0:
                local_x = min(14,local_x)
            if local_x<0:
                local_x = 0
            local_y = 7+y
            if local_y >= 0:
                local_y = min(14, local_y)
            if local_y < 0:
                local_y = 0
            local_prediction.append(local_x+15*local_y)
        local_prediction = np.array(local_prediction)


        arr_fwd_poly = np.zeros((max_poly_len * 2, 2), np.float32)
        arr_mask = np.zeros(max_poly_len * 2, np.int32)

        arr_local_prediction = np.zeros((max_poly_len * 2), np.float32)

        array_feature_indexs = np.zeros((max_poly_len * 2), np.float32)


        len_to_keep = min(len(fwd_poly), max_poly_len * 2)
        try:
            arr_fwd_poly[:len_to_keep] = fwd_poly[:len_to_keep]
        except ValueError:
#            print fwd_poly
            import ipdb;
            ipdb.set_trace()

        arr_mask[:len_to_keep] = 1
        arr_local_prediction[:len_to_keep] = local_prediction[:len_to_keep]
        array_feature_indexs[:len_to_keep] = feature_indexs[:len_to_keep]

        adj_matrix = create_adjacency_matrix_cat(arr_mask, max_poly_len)

        batch_arr_fwd_poly.append(arr_fwd_poly)
        batch_arr_mask.append(arr_mask)
        batch_arr_local_prediction.append(arr_local_prediction)
        batch_array_feature_indexs.append(array_feature_indexs)
        batch_adj_matrix.append(adj_matrix)

    return {'ggnn_fwd_poly': torch.Tensor(np.stack(batch_arr_fwd_poly, axis=0)),
            'ggnn_mask': torch.Tensor(np.stack(batch_arr_mask, axis=0)),
            'ggnn_local_prediction': torch.Tensor(np.stack(batch_arr_local_prediction, axis=0)),
            'ggnn_feature_indexs':torch.Tensor(np.stack(batch_array_feature_indexs, axis=0)),
            'ggnn_adj_matrix':torch.Tensor(np.stack(batch_adj_matrix, axis=0))
            }



def create_adjacency_matrix_cat(mask, max_poly_len):
    n_nodes = max_poly_len * 2
    n_edge_types = 3

    a = np.zeros([n_nodes, n_nodes * n_edge_types * 2])

    index, = np.where(mask == 0)
    if len(index) > 0:
        index = index[0]
        if index > 2:
            for i in range(index):
                if i % 2 == 0:
                    if i < index - 2:

                        a[i][(0) * n_nodes + i + 2] = 1
                        a[i + 2][(0 + n_edge_types) * n_nodes + i] = 1

                        a[i + 2][(0) * n_nodes + i] = 1
                        a[i][(0 + n_edge_types) * n_nodes + i + 2] = 1

                        a[i][(1) * n_nodes + i + 1] = 1
                        a[i + 1][(1 + n_edge_types) * n_nodes + i] = 1

                        a[i + 1][(2) * n_nodes + i] = 1
                        a[i][(2 + n_edge_types) * n_nodes + i + 1] = 1

                    else:
                        a[i][(0) * n_nodes + 0] = 1
                        a[0][(0 + n_edge_types) * n_nodes + i] = 1

                        a[0][(0) * n_nodes + i] = 1
                        a[i][(0 + n_edge_types) * n_nodes + 0] = 1

                        a[i][(1) * n_nodes + i + 1] = 1
                        a[i + 1][(1 + n_edge_types) * n_nodes + i] = 1

                        a[i + 1][(2) * n_nodes + i] = 1
                        a[i][(2 + n_edge_types) * n_nodes + i + 1] = 1

                else:
                    if i < index - 1:
                        a[i][(2) * n_nodes + i + 1] = 1
                        a[i + 1][(2 + n_edge_types) * n_nodes + i] = 1

                        a[i + 1][(1) * n_nodes + i] = 1
                        a[i][(1 + n_edge_types) * n_nodes + i + 1] = 1


                    else:
                        a[i][(2) * n_nodes + 0] = 1
                        a[0][(2 + n_edge_types) * n_nodes + i] = 1

                        a[0][(1) * n_nodes + i] = 1
                        a[i][(1 + n_edge_types) * n_nodes + 0] = 1

    return a.astype(np.float32)


def class_to_xy(poly, grid_size):
    """
    NOTE: Numpy function
    poly: [bs, time_steps] or [time_steps]

    Returns: [bs, time_steps, 2] or [time_steps, 2]
    """
    x = (poly % grid_size).astype(np.int32)
    y = (poly / grid_size).astype(np.int32)

    out_poly = np.stack([x,y], axis=-1)

    return out_poly

def draw_poly(mask, poly):
    """
    NOTE: Numpy function

    Draw a polygon on the mask.
    Args:
    mask: np array of type np.uint8
    poly: np array of shape N x 2
    """
    if not isinstance(poly, np.ndarray):
        poly = np.array(poly)

    cv2.fillPoly(mask, [poly], 255)

    return mask

def get_masked_poly(poly, grid_size):
    """
    NOTE: Numpy function

    Given a polygon of shape (N,), finds the first EOS token
    and masks the predicted polygon till that point
    """
    if np.max(poly) == grid_size**2:
        # If there is an EOS in the prediction
        length = np.argmax(poly)
        poly = poly[:length]
        # This automatically removes the EOS

    return poly

def poly0g_to_index(polygon, grid_side=112):
    result = []
    for item in polygon:
        result.append(item[0] + item[1] * grid_side)
    return result

def mask_and_flatten_poly(polygons, masks, grid_side):
    result = []
    for i in range(len(polygons)):

        if masks[i]:
            result.append(polygons[i][0] + polygons[i][1] * grid_side)
        else:
            result.append(grid_side ** 2)

    return np.array(result)


def local_prediction_2xy(output_dim, t_vertices):
    """
    Convert a list of vertices index into a list of xy vertices
    """
    side = output_dim / 2

    x = t_vertices % output_dim - side

    y = ((t_vertices) / output_dim) - side

    return x, y

def dt_targets_from_class(poly, grid_size, dt_threshold):
    """
    NOTE: numpy function!
    poly: [bs, time_steps], each value in [0, grid*size**2+1)
    grid_size: size of the grid the polygon is in
    dt_threshold: threshold for smoothing in dt targets

    returns: 
    full_targets: [bs, time_steps, grid_size**2+1] array containing 
    dt smoothed targets to be used for the polygon loss function
    """
    full_targets = []
    for b in range(poly.shape[0]):
        targets = []
        for p in poly[b]:
            t = np.zeros(grid_size**2+1, dtype=np.int32)
            t[p] += 1

            if p != grid_size**2:#EOS
                spatial_part = t[:-1]
                spatial_part = np.reshape(spatial_part, [grid_size, grid_size, 1])

                # Invert image
                spatial_part = -1 * (spatial_part - 1)
                # Compute distance transform
                spatial_part = distance_transform_cdt(spatial_part, metric='taxicab').astype(np.float32)
                # Threshold
                spatial_part = np.clip(spatial_part, 0, dt_threshold)
                # Normalize
                spatial_part /= dt_threshold
                # Invert back
                spatial_part = -1. * (spatial_part - 1.)

                spatial_part /= np.sum(spatial_part)
                spatial_part = spatial_part.flatten()

                t = np.concatenate([spatial_part, [0.]], axis=-1)

            targets.append(t.astype(np.float32))
        full_targets.append(targets)

    return np.array(full_targets, dtype=np.float32)

if __name__ == '__main__':
    poly = np.array([[5, 5], [8, 8], [8, 5]])
    img = np.zeros((2, 10, 10), np.uint8)
    img = draw_poly(img[0], poly)
#    print img