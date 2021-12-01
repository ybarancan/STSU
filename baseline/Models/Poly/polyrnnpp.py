import torch
import torch.nn as nn
import torch.nn.functional as F

import baseline.Utils.utils as utils
#from Models.Encoder.resnet_skip import SkipResnet50
from baseline.Models.Poly.conv_lstm import AttConvLSTM
#from Models.Poly.first_v import FirstVertex
#from Models.Evaluator.evaluator import Evaluator
#from Models.GGNN.poly_ggnn import PolyGGNN
import numpy as np
import logging
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolyRNNpp(nn.Module):
    def __init__(self, opts, final_dim, feat_size):
        super(PolyRNNpp, self).__init__()

        self.opts = opts
#        print 'Building polyrnnpp with opts:\n',opts
        self.mode = self.opts['mode']
        self.temperature = self.opts['temperature']
        self.final_dim = final_dim
        self.feat_size = feat_size
        
#        if 'use_correction' not in self.opts.keys():
#            self.opts['use_correction'] = False
#
#
##        print 'Building encoder'
#        self.encoder = SkipResnet50()
#
#        if 'train_encoder' in self.opts.keys() and not self.opts['train_encoder']:
#            for p in self.encoder.parameters():
#                p.requires_grad = False
#
##        print 'Building first vertex network'
#        self.first_v = FirstVertex(opts, feats_channels = self.encoder.final_dim,
#            feats_dim = self.encoder.feat_size)

#        print 'Building convlstm'
        self.conv_lstm = AttConvLSTM(
            opts,
            feats_channels = final_dim,
            feats_dim = feat_size,
            time_steps = opts['max_poly_len'],
            use_bn = self.opts['use_bn_lstm']
        )

#        if 'train_attention' in self.opts.keys() and not self.opts['train_attention']:
#            for n,p in self.conv_lstm.named_parameters():
#                if 'att' in n:
#                    p.requires_grad = False
#
#        if 'use_evaluator' in self.opts and self.opts['use_evaluator']:
##            print 'Building Evaluator'
#
#            self.evaluator = Evaluator(
#                feats_dim = self.encoder.feat_size,
#                feats_channels = self.encoder.final_dim,
#                hidden_channels = self.conv_lstm.hidden_dim
#            )
#
#        else:
#            self.evaluator = None
#
#        if 'use_ggnn' in self.opts and self.opts['use_ggnn']:
##            print 'Building GGNN'
#
#            for p in self.encoder.parameters():
#                p.requires_grad = False
#            for p in self.first_v.parameters():
#                p.requires_grad = False
#            for p in self.conv_lstm.parameters():
#                p.requires_grad = False
#            for p in self.evaluator.parameters():
#                p.requires_grad = False
#
#            if 'train_ggnn_encoder' not in self.opts.keys():
#                self.opts['train_ggnn_encoder'] = False
#
#            self.ggnn = PolyGGNN(
#                image_feature_dim=self.encoder.image_feature_dim,
#                ggnn_n_steps=self.opts['ggnn_n_steps'],
#                state_dim=self.opts['ggnn_state_dim'],
#                output_dim=self.opts['ggnn_output_dim'],
#                max_poly_len=self.opts['max_poly_len'],
#                use_separate_encoder=self.opts['use_separate_encoder'],
#                poly_ce_grid_size=self.encoder.feat_size,
#                ggnn_grid_size=self.opts['ggnn_grid_size']
#            )
#
#        else:
#            self.opts['use_ggnn'] = False
#            self.ggnn = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self,
        x,
        poly=None,
        fp_beam_size=1,
        lstm_beam_size=1,
        orig_poly=None,
        run_ggnn=False,
        training=True):
        """
        x: [bs, 3, 224, 224]
#        poly: [bs, self.max_time]
        poly: [bs, 3, 2]
        
        """
        batch_size = x.size(0)
    
        if lstm_beam_size != 1 or fp_beam_size != 1:
            assert 'train' not in self.mode, 'Run beam search only in test mode!'

#        concat_feats, feats = self.encoder(x)
#
#        edge_logits, vertex_logits, first_logprob, first_v = self.first_v(feats,
#            temperature = self.temperature,
#            beam_size = fp_beam_size)
#
#        poly_class = None
#        if poly is not None:
#            poly_class = utils.xy_to_class(poly, grid_size=self.encoder.feat_size)

#        if self.mode == 'train_ce':
            # When training with cross_entropy
#            first_v = poly_class[:,0]
#            first_logprob = None

        if training:
            # logging.error('BEFORE CONVERTING')
            # logging.error(str(poly))
            poly_class = utils.xy_to_class(poly, grid_size=self.feat_size)
            # logging.error('CONVERTED TO CLASS')
            # logging.error(str(poly_class))
            
            # temp_gir = utils.class_to_xy(poly_class.data.detach().cpu().numpy(),grid_size=self.feat_size)
            # logging.error('CONVERTED BACK TO GRID')
            # logging.error(str(temp_gir))
            
            first_logprob = None
            first_v = poly_class[:,0]
            out_dict = self.conv_lstm(
                x,
                first_v,
                poly_class,
                temperature = self.temperature,
                mode = self.mode,
                # fp_beam_size = fp_beam_size,
                # beam_size = lstm_beam_size,
                first_log_prob = first_logprob,
               
            )
            
#            logging.error('ESTIMATES')
#            logging.error(str(out_dict['pred_polys']))
            
        else:
            poly_class = utils.xy_to_class(poly, grid_size=self.feat_size)
            first_v = poly_class[:,0]
            first_logprob = None
            out_dict = self.conv_lstm(
                x,
                first_v,
                None,
                temperature = self.temperature,
                mode = 'test',
                # fp_beam_size = fp_beam_size,
                # beam_size = lstm_beam_size,
                first_log_prob = first_logprob,
                # return_attention = self.opts['return_attention'],
                # use_correction=self.opts['use_correction']
            )
#        elif self.mode == 'train_ggnn' and self.opts['use_correction']:
#            # GGNN trains on corrected polys
#            assert poly is not None, 'Need to pass poly for GGNN training!'
#            assert 'correction_threshold' in self.opts.keys(),\
#            'Need to pass correction threshold for GGNN training!'
#
#            poly_class = self.first_v.first_point_correction(
#                first_v, poly_class, self.opts['correction_threshold'])
#
#            first_v = poly_class[:, 0]
#
#        elif 'tool' in self.mode and poly is not None:
#            # tool in fixing mode
#            first_v = poly_class[:, 0]
#            first_logprob = None
#            lstm_beam_size = 1

        

        if self.mode == 'train_ce' or self.mode == 'train_rl':
            # out_dict['edge_logits'] = edge_logits
            # out_dict['vertex_logits'] = vertex_logits
        
            if poly_class is not None:
                out_dict['poly_class'] = poly_class.type(torch.long)

        # if self.evaluator is not None:
        #     ious = self.evaluator(
        #         out_dict['feats'],
        #         out_dict['rnn_state'],
        #         out_dict['pred_polys']
        #     )
        #     comparison_metric = ious
        #     out_dict['ious'] = ious

        # else:
        comparison_metric = out_dict['logprob_sums']

        # if fp_beam_size != 1 or lstm_beam_size != 1:
        if not training:
            # Automatically means that this is in test mode
            # because of the assertion in the beginning

            # comparison metric is of shape [batch_size * fp_beam_size * lstm_beam_size]
#            if 'tool' in self.mode:
#                # Check for intersections and remove
#                isect = utils.count_self_intersection(
#                    out_dict['pred_polys'].cpu().numpy(),
#                    self.encoder.feat_size
#                )
#                isect[isect != 0] -= float('inf')
#                # 0 means no intersection, -inf for intersection
#                isect = torch.from_numpy(isect).to(torch.float32).to(device)
#                comparison_metric = comparison_metric + isect
#                print comparison_metric

            comparison_metric = comparison_metric.view(batch_size, fp_beam_size, lstm_beam_size)
            out_dict['pred_polys'] = out_dict['pred_polys'].view(batch_size, fp_beam_size, lstm_beam_size, -1)

            # Max across beams
            comparison_metric, beam_idx = torch.max(comparison_metric, dim=-1)

            # Max across first points
            comparison_metric, fp_beam_idx = torch.max(comparison_metric, dim=-1)

            pred_polys = torch.zeros(batch_size, self.opts['max_poly_len'], device=device,
                dtype=out_dict['pred_polys'].dtype)

            for b in torch.arange(batch_size, dtype=torch.int32):
                # Get best beam from all first points and all beams
                pred_polys[b, :] = out_dict['pred_polys'][b, fp_beam_idx[b], beam_idx[b, fp_beam_idx[b]], :]

            out_dict['pred_polys'] = pred_polys

        # No need to send rnn_state and feats back
        # out_dict.pop('rnn_state')
        # out_dict.pop('feats')
        
#        if self.opts['use_ggnn'] and run_ggnn:
#            pred_polys = out_dict['pred_polys'].detach().cpu().numpy()
#
#            del (out_dict)
#            resnet_feature = concat_feats
#            out_dict = self.ggnn(x, pred_polys, mode=self.mode, gt_polys=orig_poly, resnet_feature=resnet_feature)

        return out_dict

    def reload(self, path, strict=False):
#        print "Reloading full model from: ", path
        self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage)['state_dict'],
            strict=strict)
        # In case we want to reload parts of the model, strict is False

if __name__ == '__main__':
    opts = {}
    opts['batch_size'] = 8
    opts['time_steps'] = 71
    opts['temperature'] = 0.0
    opts['mode'] = 'train'
    opts['return_attention'] = False

    model = PolyRNNpp(opts)

    x = torch.rand(opts['batch_size'], 3, 224, 224)
    poly = torch.randint(low=0, high=27, size=[opts['batch_size'], opts['time_steps'], 2])

    output = model(x, poly)

#    for k in output.keys():
#        print k, output[k].size()
