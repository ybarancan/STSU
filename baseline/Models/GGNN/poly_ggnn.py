import torch
import torch.nn as nn
import Utils.utils as utils
from Models.Encoder.resnet_skip import SkipResnet50
from Models.Encoder.ggnn_feature_encoder import GgnnFeatureEncoder
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolyGGNN(nn.Module):
    def __init__(self,
                 image_feature_dim=256,
                 ggnn_n_steps=3,
                 state_dim=256,
                 output_dim=15,
                 max_poly_len=71,
                 use_separate_encoder=True,
                 poly_ce_grid_size=28,
                 ggnn_grid_size=112
                 ):
      
        super(PolyGGNN, self).__init__()

        self.image_feature_dim = image_feature_dim
        self.n_steps = ggnn_n_steps
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.max_poly_len = max_poly_len
        self.num_nodes =  max_poly_len * 2
        self.n_edge_types = 3
        self.poly_ce_grid_size = poly_ce_grid_size
        self.ggnn_grid_size = ggnn_grid_size
        self.use_separate_encoder = use_separate_encoder


        if not self.use_separate_encoder:
            print 'Building GGNN Feature Encoder'
            self.encoder = GgnnFeatureEncoder(input_dim=image_feature_dim, final_dim=image_feature_dim)
        else:
            print 'Building GGNN Encoder'
            self.encoder = SkipResnet50()
            self.extract_local_feature = nn.Conv2d(
                in_channels = self.image_feature_dim,
                out_channels =  self.image_feature_dim,
                kernel_size = 15,
                padding = 7,
                bias = True
            )

        # FC layers for edge type 0
        in_state_0_0 = nn.Linear(
            in_features = self.state_dim,
            out_features = self.state_dim,
        )

        in_state_0_1 = nn.Linear(
            in_features = self.state_dim,
            out_features = self.state_dim,
        )

        out_state_0_0 = nn.Linear(
            in_features = self.state_dim,
            out_features = self.state_dim,
        )

        out_state_0_1 = nn.Linear(
            in_features= self.state_dim,
            out_features= self.state_dim,
        )

        self.in_state_0 = nn.Sequential(
            in_state_0_0,
            nn.Tanh(),
            in_state_0_1
        )

        self.out_state_0 = nn.Sequential(
            out_state_0_0,
            nn.Tanh(),
            out_state_0_1
        )

        # FC layers for edge type 1
        in_state_1_0 = nn.Linear(
            in_features = self.state_dim,
            out_features = self.state_dim,
        )

        in_state_1_1 = nn.Linear(
            in_features = self.state_dim,
            out_features = self.state_dim,
        )

        out_state_1_0 = nn.Linear(
            in_features = self.state_dim,
            out_features = self.state_dim,
        )

        out_state_1_1 = nn.Linear(
            in_features= self.state_dim,
            out_features= self.state_dim,
        )

        self.in_state_1 = nn.Sequential(
            in_state_1_0,
            nn.Tanh(),
            in_state_1_1
        )

        self.out_state_1 = nn.Sequential(
            out_state_1_0,
            nn.Tanh(),
            out_state_1_1
        )

        # FC layers for edge type 2
        in_state_2_0 = nn.Linear(
            in_features=self.state_dim,
            out_features=self.state_dim,
        )

        in_state_2_1 = nn.Linear(
            in_features=self.state_dim,
            out_features=self.state_dim,
        )

        out_state_2_0 = nn.Linear(
            in_features=self.state_dim,
            out_features=self.state_dim,
        )

        out_state_2_1 = nn.Linear(
            in_features=self.state_dim,
            out_features=self.state_dim,
        )

        self.in_state_2 = nn.Sequential(
            in_state_2_0,
            nn.Tanh(),
            in_state_2_1
        )

        self.out_state_2 = nn.Sequential(
            out_state_2_0,
            nn.Tanh(),
            out_state_2_1
        )

        self.gates_other_h_fc = nn.Linear(
            in_features=2 * self.state_dim,
            out_features=2 * self.state_dim,
        )

        self.gates_curr_h_fc = nn.Linear(
            in_features=2 * self.state_dim,
            out_features=2 * self.state_dim,
        )

        self.transformed_output_other_h_fc = nn.Linear(
            in_features=2 * self.state_dim,
            out_features=self.state_dim,
        )

        self.transformed_output_curr_h_fc = nn.Linear(
            in_features=  self.state_dim,
            out_features=self.state_dim,
        )

        self.ggnn_output_layer = nn.Sequential(
            nn.Linear(
                in_features=self.state_dim,
                out_features=self.state_dim,
            ),
            nn.Tanh(),
            nn.Linear(
                in_features=self.state_dim,
                out_features=self.output_dim * self.output_dim,
            )
        )

    def propagate(self, prop_inputs):
        in_edge_states = torch.cat((self.in_state_0(prop_inputs), self.in_state_1(prop_inputs), self.in_state_2(prop_inputs)), dim=1)
        out_edge_states = torch.cat((self.out_state_0(prop_inputs), self.out_state_1(prop_inputs), self.out_state_2(prop_inputs)), dim=1)

        return in_edge_states, out_edge_states

    def GRU_update(self, state_in, state_out, state_cur, forward_adj, backward_adj):

        forward_input = torch.bmm(forward_adj, state_in)
        backward_input =  torch.bmm(backward_adj, state_out)
        curr_input = torch.cat((forward_input, backward_input), dim=2)
        gates_other_h = self.gates_other_h_fc(curr_input)
        gates_curr_h = self.gates_curr_h_fc(curr_input)
        gates = torch.sigmoid(gates_curr_h + gates_other_h)
        update_gate = gates[:,:,:self.state_dim]
        reset_gate = gates[:,:,self.state_dim:]
        transformed_output_other_h = self.transformed_output_other_h_fc(curr_input)
        transformed_output_curr_h = self.transformed_output_curr_h_fc(reset_gate * state_cur)
        transformed_output = torch.tanh(transformed_output_other_h + transformed_output_curr_h)
        output = state_cur + update_gate * (transformed_output - state_cur)

        return output

    def ggnn_inference(self, init_input, forward_adj, backward_adj):

        prop_inputs = init_input
        for i_step in range(self.n_steps):
            in_edge_states, out_edge_states = self.propagate(prop_inputs)
            prop_inputs = self.GRU_update(in_edge_states, out_edge_states, prop_inputs, forward_adj, backward_adj)
        return self.ggnn_output_layer(prop_inputs)

    def forward(self, x, pred_polys, mode='train_ggnn', gt_polys=None, resnet_feature=None):
        """
        poly: [bs, self.max_time]
        """
        if gt_polys == None:
            gt_polys = np.zeros([pred_polys.shape[0], pred_polys.shape[1], 2], dtype=np.float32)

        d = utils.prepare_ggnn_component(pred_polys,
                                         gt_polys,
                                         self.poly_ce_grid_size,
                                         self.ggnn_grid_size,
                                         self.max_poly_len)

        adjacent = d['ggnn_adj_matrix'].to(device)
        init_poly_idx = d['ggnn_feature_indexs'].to(device)

        if self.use_separate_encoder:
            concat_feats,_ = self.encoder.forward(x)
            cnn_feats = self.extract_local_feature(concat_feats)
        else:
            # cnn_feats = self.encoder.forward(torch.Tensor(resnet_feature).to(device))
            cnn_feats = self.encoder.forward(resnet_feature)

        channels = cnn_feats.size(1)
        cnn_feats = cnn_feats.permute(0, 2, 3, 1).view(-1, 112 * 112, channels)

        feature_id = init_poly_idx.unsqueeze_(2).long().expand(init_poly_idx.size(0), init_poly_idx.size(1), cnn_feats.size(2)).detach()
        # ? , num of nodes, feat_size
        ggnn_cnn_feature = torch.gather(cnn_feats, 1, feature_id)

        if self.state_dim - channels > 0:
            dummy_tensor = torch.zeros(ggnn_cnn_feature.size(0), ggnn_cnn_feature.size(1), self.state_dim - cnn_feats.size(-1)).to(device)
            ggnn_cnn_feature = torch.cat((ggnn_cnn_feature, dummy_tensor), 2)

        ggnn_logits = self.ggnn_inference(ggnn_cnn_feature, adjacent[:, :, :self.num_nodes * self.n_edge_types], adjacent[:, :, self.num_nodes * self.n_edge_types: ])
        ggnn_init_poly = d['ggnn_fwd_poly'].to(device)

        out_dict = {}

        if 'train' in mode:
            out_dict['ggnn_logits'] = ggnn_logits
            out_dict['ggnn_local_prediction'] = d['ggnn_local_prediction']
            out_dict['ggnn_mask'] = d['ggnn_mask']

        with torch.no_grad():
            pred_x = []
            pred_y = []
            for t in range(self.num_nodes):
                logits = ggnn_logits[:, t, :]

                _, sampled_point = torch.max(logits, 1)

                x, y = utils.local_prediction_2xy(self.output_dim, sampled_point.detach())
                pred_x.append(x)
                pred_y.append(y)

            pred_x = torch.transpose(torch.stack(pred_x), 1, 0).float()
            pred_y = torch.transpose(torch.stack(pred_y), 1, 0).float()


            results = torch.stack([ggnn_init_poly[:, :, 0] + pred_x, ggnn_init_poly[:, :, 1] + pred_y],
                                     2).data.cpu().numpy()

            pred_polys = []
            for j in range(len(results)):
                pred_polys.append(utils.mask_and_flatten_poly(results[j], d['ggnn_mask'][j], self.ggnn_grid_size))

            pred_polys = torch.Tensor(np.stack(pred_polys, 0))
            
        out_dict['pred_polys'] = pred_polys

        return out_dict