import torch
import torch.nn.functional as F
import numpy as np
from Utils import utils

def accuracy(poly, mask, pred_polys, grid_size):
    """
    Computes prediction accuracy

    poly: [batch_size, time_steps]
    pred_polys: [batch_size, time_steps,]
    Each element stores y*grid_size + x, or grid_size**2 for EOS

    mask: [batch_size, time_steps,]
    The mask of valid time steps in the GT poly. It is manipulated
    inside this function!

    grid_size: size of the grid in which the polygons are in    
    """
    idxs = np.argmax(pred_polys, axis=-1)
    for i,idx in enumerate(idxs):
        if pred_polys[i,idx] == grid_size**2:
            # If EOS
            if idx > np.sum(mask[i,:]):
                # If there are more predictions than
                # ground truth points, then extend mask
                mask[i, :idx] = 1.

        else:
            # If no EOS was predicted
            mask[i, :] = 1.
    
    corrects = pred_polys == poly

    corrects = corrects * mask
    percentage = np.sum(corrects, axis=-1)*1.0/np.sum(mask, axis=-1)

    return np.mean(percentage)

def train_accuracy(poly, mask, pred_polys, grid_size):
    """
    Computes prediction accuracy with GT masks

    poly: [batch_size, time_steps]
    pred_polys: [batch_size, time_steps,]
    Each element stores y*grid_size + x, or grid_size**2 for EOS

    mask: [batch_size, time_steps,]

    grid_size: size of the grid in which the polygons are in    
    accepts grid_size to be compatible with accuracy()
    """
    corrects = (pred_polys == poly).astype(np.float32)

    corrects = corrects * mask

    percentage = np.sum(corrects, axis=-1)*1.0/np.sum(mask, axis=-1)

    return np.mean(percentage)

def iou_from_mask(pred, gt):
    """
    Compute intersection over the union.
    Args:
        pred: Predicted mask
        gt: Ground truth mask
    """
    pred = pred.astype(np.bool)
    gt = gt.astype(np.bool)

    # true_negatives = np.count_nonzero(np.logical_and(np.logical_not(gt), np.logical_not(pred)))
    false_negatives = np.count_nonzero(np.logical_and(gt, np.logical_not(pred)))
    false_positives = np.count_nonzero(np.logical_and(np.logical_not(gt), pred))
    true_positives = np.count_nonzero(np.logical_and(gt, pred))

    union = float(true_positives + false_positives + false_negatives)
    intersection = float(true_positives)

    iou = intersection / union if union > 0. else 0.

    return iou

def iou_from_poly(pred, gt, width, height):
    """
    Compute IoU from poly. The polygons should
    already be in the final output size

    pred: list of np arrays of predicted polygons
    gt: list of np arrays of gt polygons
    grid_size: grid_size that the polygons are in

    """
    masks = np.zeros((2, height, width), dtype=np.uint8)

    if not isinstance(pred, list):
        pred = [pred]
    if not isinstance(gt, list):
        gt = [gt]

    for p in pred: 
        masks[0] = utils.draw_poly(masks[0], p)

    for g in gt:
        masks[1] = utils.draw_poly(masks[1], g)

    return iou_from_mask(masks[0], masks[1]), masks
