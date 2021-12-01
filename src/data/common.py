from collections import defaultdict
# from src.structures import ImageList
# from src.utils.vis import overlay_mask_on_image, create_color_map
from torch.nn import functional as F
import skimage
import cv2
import math
import numpy as np
import torch
import logging

def scale_and_normalize_images(images, means, scales, invert_channels, normalize_to_unit_scale):
    """
    Scales and normalizes images
    :param images: tensor(T, C, H, W)
    :param means: list(float)
    :param scales: list(float)
    :param invert_channels: bool
    :return: tensor(T, C, H, W)
    """
    means = torch.tensor(means, dtype=torch.float32)[None, :, None, None]  # [1, 3, 1, 1]
    scales = torch.tensor(scales, dtype=torch.float32)[None, :, None, None]  # [1. 3. 1. 1]
    if normalize_to_unit_scale:
        images = images / 255.

    images = (images - means) / scales
    if invert_channels:
        return images.flip(dims=[1])
    else:
        return images


def compute_padding(width, height):
    pad_right = (int(math.ceil(width / 32)) * 32) - width
    pad_bottom = (int(math.ceil(height / 32)) * 32) - height
    return pad_right, pad_bottom



def expand_mask_with_bg(masks,n_instances,stack_dim=0):
    
    res=[]
    res.append(masks == 0)
    for obj in range(1,n_instances+1):
        res.append(masks == obj)
    
    return np.stack(res,axis=stack_dim)
        
def cv2_aspect_resize_and_pad_images(images, target_shape,interpolation='bilinear'):
    """
    Resizes and pads images for input to network
    :param images: tensor(T, C, H, W)
    :param min_dim: int
    :param max_dim: int
    :return: tensor(T, C, H, W)
    """
    height, width = images.shape[:2]
    
    if len(images.shape)==3:
        last_dim = images.shape[-1]
    # resize_width, resize_height, _ = compute_resize_params_2((width, height), min_dim, max_dim)

    # make width and height a multiple of 32
    
    target_width, target_height = target_shape
    ratio = np.min([target_width/width,target_height/height])
    resize_width = int(ratio*width)
    resize_height = int(ratio*height)
    
    pad_left = int(np.floor((target_width - resize_width)/2))
    pad_right = int(np.ceil((target_width - resize_width)/2))
    
    pad_up = int(np.floor((target_height - resize_height)/2))
    pad_down = int(np.ceil((target_height - resize_height)/2))
    
    
    images = skimage.transform.resize(np.float32(images),(resize_height,resize_width))
    
    # if interpolation == 'bilinear':
    #     images = cv2.resize(images, (resize_width, resize_height), interpolation = cv2.INTER_LINEAR)
    # elif interpolation == 'nearest':
    #     images = cv2.resize(images, (resize_width, resize_height), interpolation = cv2.INTER_NEAREST)
    # else:
    #     images = cv2.resize(images, (resize_width, resize_height), interpolation = cv2.INTER_LINEAR)
    
    
    
    if len(images.shape) == 2:
        return np.expand_dims(np.pad(images, ((pad_up, pad_down),(pad_left, pad_right) )),axis=-1)
    else:
        return np.pad(images, ((pad_up, pad_down),(pad_left, pad_right),(0,0) ))
    
        
def torch_aspect_resize_and_pad_images(images, target_shape):
    """
    Resizes and pads images for input to network
    :param images: tensor(T, C, H, W)
    :param min_dim: int
    :param max_dim: int
    :return: tensor(T, C, H, W)
    """
    height, width = images.shape[-2:]
    # resize_width, resize_height, _ = compute_resize_params_2((width, height), min_dim, max_dim)

    # make width and height a multiple of 32
    
    target_width, target_height = target_shape
    ratio = np.min([target_width/width,target_height/height])
    resize_width = int(ratio*width)
    resize_height = int(ratio*height)
    
    pad_left = int(np.floor((target_width - resize_width)/2))
    pad_right = int(np.ceil((target_width - resize_width)/2))
    
    pad_up = int(np.floor((target_height - resize_height)/2))
    pad_down = int(np.ceil((target_height - resize_height)/2))
    
    images = F.interpolate(images, (resize_height,resize_width), mode="bilinear", align_corners=False)
    return F.pad(images, (pad_left, pad_right, pad_up, pad_down))


def pad_masks_to_image(image_seqs, targets):
    padded_h, padded_w = image_seqs.max_size

    for targets_per_seq in targets:
        instance_masks = targets_per_seq['masks']  # [N, T, H, W]
        ignore_masks = targets_per_seq['ignore_masks']  # [T, H, W]

        mask_h, mask_w = instance_masks.shape[-2:]
        pad_bottom, pad_right = padded_h - mask_h, padded_w - mask_w

        instance_masks = F.pad(instance_masks, (0, pad_right, 0, pad_bottom))
        ignore_masks = F.pad(ignore_masks.unsqueeze(0), (0, pad_right, 0, pad_bottom)).squeeze(0)

        targets_per_seq['masks'] = instance_masks
        targets_per_seq['ignore_masks'] = ignore_masks

    return targets


# def collate_fn(samples):
#     image_seqs, targets, original_dims, meta_info = zip(*samples)
#     image_seqs = ImageList.from_image_sequence_list(image_seqs, original_dims)
#     targets = pad_masks_to_image(image_seqs, targets)
#     return image_seqs, targets, meta_info


def targets_to(targets, *args, **kwargs):
    to_targets = []
    for targets_per_image in targets:
        to_targets_per_image = {}
        for k, v in targets_per_image.items():
            if isinstance(v, torch.Tensor):
                to_targets_per_image[k] = v.to(*args, **kwargs)
            else:
                to_targets_per_image[k] = v
        to_targets.append(to_targets_per_image)
    return to_targets


def tensor_struct_to(struct, *args, **kwargs):
    if isinstance(struct, (list, tuple)):
        to_struct = []
        for elem in struct:
            if torch.is_tensor(elem) or hasattr(elem, "to"):
                to_struct.append(elem.to(*args, **kwargs))
            else:
                to_struct.append(tensor_struct_to(elem, *args, **kwargs))
    elif isinstance(struct, dict):
        to_struct = {}
        for k, v in struct.items():
            if torch.is_tensor(v) or hasattr(v, "to"):
                to_struct[k] = v.to(*args, **kwargs)
            else:
                to_struct[k] = tensor_struct_to(v, *args, **kwargs)
    else:
        raise TypeError("Variable of unknown type {} found".format(type(struct)))

    return to_struct


def tensor_struct_to_cuda(struct):
    return tensor_struct_to(struct, device="cuda:0")


def targets_to_cuda(targets):
    return targets_to(targets, "cuda:0")


def nested_dict_to(d, *args, **kwargs):
    to_dict = dict()
    for k, v in d.items():
        if torch.is_tensor(v):
            to_dict[k] = v.to(*args, **kwargs)
        elif isinstance(v, (dict, defaultdict)):
            to_dict[k] = nested_dict_to(v, *args, **kwargs)
        else:
            to_dict[k] = v
    return to_dict


def nested_dict_to_cuda(d):
    return nested_dict_to(d, "cuda:0")


def compute_resize_params_2(image_dims, min_resize_dim, max_resize_dim):
    """
    :param image_dims: as tuple of (width, height)
    :param min_resize_dim:
    :param max_resize_dim:
    :return:
    """
    lower_size = float(min(image_dims))
    higher_size = float(max(image_dims))

    scale_factor = min_resize_dim / lower_size
    if (higher_size * scale_factor) > max_resize_dim:
        scale_factor = max_resize_dim / higher_size

    width, height = image_dims
    new_height, new_width = round(scale_factor * height), round(scale_factor * width)

    return new_width, new_height, scale_factor


def compute_resize_params(image, min_dim, max_dim):
    lower_size = float(min(image.shape[:2]))
    higher_size = float(max(image.shape[:2]))

    scale_factor = min_dim / lower_size
    if (higher_size * scale_factor) > max_dim:
        scale_factor = max_dim / higher_size

    height, width = image.shape[:2]
    new_height, new_width = round(scale_factor * height), round(scale_factor * width)

    return new_width, new_height, scale_factor


def compute_mask_gradients(masks, dilation_kernel_size=5):
    """
    :param masks: tensor(N, T, H, W)
    :return:
    """
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size))
    mask_gradients = masks.to(torch.float32).numpy()
    mask_gradients = torch.stack([
        torch.stack([
            torch.from_numpy(cv2.dilate(cv2.Laplacian(mask_gradients[n, t], cv2.CV_32F), kernel))
            for t in range(mask_gradients.shape[1])
        ])
        for n in range(mask_gradients.shape[0])
    ]) > 0
    mask_gradients = mask_gradients.to(torch.uint8)
    return torch.any(mask_gradients, dim=0)


@torch.no_grad()
def instance_masks_to_semseg_mask(instance_masks, category_labels):
    """
    Converts a tensor containing instance masks to a semantic segmentation mask.
    :param instance_masks: tensor(N, T, H, W)  (N = number of instances)
    :param category_labels: tensor(N) containing semantic category label for each instance.
    :return: semantic mask as tensor(T, H, W] with pixel values containing class labels
    """
    assert len(category_labels) == instance_masks.shape[0], \
        "Number of instances do not match: {}, {}".format(len(category_labels), len(instance_masks))
    semseg_masks = instance_masks.long()

    for i, label in enumerate(category_labels):
        semseg_masks[i] = torch.where(instance_masks[i], label, semseg_masks[i])

    # for pixels with differing labels, assign to the category with higher ID number (arbitrary criterion)
    return semseg_masks.max(dim=0)[0]  # [T, H, W]


# def visualize_semseg_masks(image, semseg_mask):
#     category_labels = set(np.unique(semseg_mask).tolist()) - {0}
#     if not category_labels:
#         return image
#     assert max(category_labels) < 256

#     image = np.copy(image)
#     cmap = create_color_map()

#     for label in category_labels:
#         image = overlay_mask_on_image(image, semseg_mask == label, mask_color=cmap[label])

#     return image
