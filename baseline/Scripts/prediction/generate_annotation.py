import torch
import json
import os
import argparse
import numpy as np
import torch.nn as nn
import warnings
import skimage.io as sio
from torch.utils.data import DataLoader
from tqdm import tqdm
from Utils import utils
from DataProvider import cityscapes
from Models.Poly import polyrnnpp



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--reload', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--fp_beam_size', type=int, default=1)
    parser.add_argument('--lstm_beam_size', type=int, default=1)

    args = parser.parse_args()

    return args

def get_data_loaders(opts, DataProvider):
    print 'Building dataloaders'

    dataset_val = DataProvider(split='val', opts=opts['train_val'], mode='test')

    val_loader = DataLoader(dataset_val, batch_size=opts['train_val']['batch_size'],
        shuffle = False, num_workers=opts['train_val']['num_workers'], collate_fn=cityscapes.collate_fn)
    
    return val_loader

def override_options(opts):
    opts['mode'] = 'test'
    opts['temperature'] = 0.0
    opts['dataset']['train_val']['skip_multicomponent'] = False
    opts.pop('encoder_reload', None)
    # No need to reload resnet first

    return opts

class Tester(object):
    def __init__(self, args):
        self.opts = json.load(open(args.exp, 'r'))
        self.output_dir = args.output_dir
        self.fp_beam_size = args.fp_beam_size
        self.lstm_beam_size = args.lstm_beam_size
        if self.output_dir is None:
            self.output_dir = os.path.join(self.opts['exp_dir'], 'preds')

        utils.create_folder(self.output_dir)
        self.opts = override_options(self.opts)
        self.val_loader = get_data_loaders(self.opts['dataset'], cityscapes.DataProvider)
        self.model = polyrnnpp.PolyRNNpp(self.opts).to(device)
        self.model.reload(args.reload, strict=False)

        if self.opts['use_ggnn'] == True:
            self.grid_size = self.model.ggnn.ggnn_grid_size
        else:
            self.grid_size = self.model.encoder.feat_size

    def process_outputs(self, data, output, save=True):
        """
        Process outputs to get final outputs for the whole image
        Optionally saves the outputs to a folder for evaluation
        """
        instances = data['instance']
        polys = []
        for i, instance in enumerate(instances):
            # Postprocess polygon
            poly = output['pred_polys'][i]

            _, poly = utils.get_full_mask_from_xy(poly,
                    self.grid_size,
                    data['patch_w'][i],
                    data['starting_point'][i],
                    instance['img_height'],
                    instance['img_width'])

            polys.append(poly)

            if save:
                img_h, img_w = instance['img_height'], instance['img_width']
                predicted_poly = []

                # Paint pred mask
                pred_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                poly = poly.astype(np.int)
                utils.draw_poly(pred_mask, poly)
                predicted_poly.append(poly.tolist())

                # Paint GT mask
                gt_mask = utils.get_full_mask_from_instance(
                    self.opts['dataset']['train_val']['min_area'], 
                    instance)

                instance['my_predicted_poly'] = predicted_poly            
                instance_id = instance['instance_id']
                image_id = instance['image_id']
           
                pred_mask_fname = os.path.join(self.output_dir, '{}_pred.png'.format(instance_id))
                instance['pred_mask_fname'] = os.path.relpath(pred_mask_fname, self.output_dir)

                gt_mask_fname = os.path.join(self.output_dir, '{}_gt.png'.format(instance_id))
                instance['gt_mask_fname'] = os.path.relpath(gt_mask_fname, self.output_dir)
            
                instance['n_corrections'] = 0

                info_fname = os.path.join(self.output_dir, '{}_info.json'.format(instance_id))

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sio.imsave(pred_mask_fname, pred_mask)
                    sio.imsave(gt_mask_fname, gt_mask)

                with open(info_fname, 'w') as f:
                    json.dump(instance, f, indent=2)

        return polys

    def test(self):
        print 'Starting testing'
        self.model.encoder.eval()
        self.model.first_v.eval()
        if self.model.evaluator is not None:
            self.model.evaluator.eval()
        if self.model.ggnn is not None:
            self.model.ggnn.encoder.eval()
        # Leave LSTM in train mode

        grid_size = self.model.encoder.feat_size
        with torch.no_grad():
            for step, data in enumerate(tqdm(self.val_loader)):
                # Forward pass
                output = self.model(data['img'].to(device), 
                    fp_beam_size=self.fp_beam_size, 
                    lstm_beam_size=self.lstm_beam_size)

                # Bring everything to cpu/numpy
                for k in output.keys():
                    output[k] = output[k].cpu().numpy()

                self.process_outputs(data, output, save=True)
                del(output)

if __name__ == '__main__':
    args = get_args()
    tester = Tester(args)
    tester.test()
