import torch.nn as nn
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GgnnFeatureEncoder(nn.Module):
    def __init__(self, input_dim=256, final_dim=256):
        super(GgnnFeatureEncoder, self).__init__()

        conv_final_1 = nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1,  bias=False)
        bn_final_1 = nn.BatchNorm2d(input_dim)
        conv_final_2 = nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1, bias=False)
        bn_final_2 = nn.BatchNorm2d(input_dim)
        conv_final_3 = nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1, bias=False)
        bn_final_3 = nn.BatchNorm2d(input_dim)

        extract_local_feature = nn.Conv2d(
            in_channels=input_dim,
            out_channels=final_dim,
            kernel_size=15,
            padding=7,
            bias=True
        )
        self.conv_final = nn.Sequential(conv_final_1, bn_final_1, conv_final_2, bn_final_2,
            conv_final_3, bn_final_3, extract_local_feature)

    def reload(self, path):
        print "Reloading resnet from: ", path
        self.resnet.load_state_dict(torch.load(path))

    def forward(self, x):
        final_features = self.conv_final(x)

        return final_features