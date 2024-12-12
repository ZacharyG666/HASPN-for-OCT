import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from attention import *


# Image decomposer for making high-frequency counterparts, there are two ways to obtain them.
# 1.pre-process original images to prepare high-frequency counterparts. (Recommend)
# 2.mid-process original images i.e. process images in the training phase to prepare high-frequency counterparts.
def decomposer(img_tensor, kernel_size=(5, 5), sigma=1.5):
    img_tensor = img_tensor.detach().clone()
    img_numpy = img_tensor.cpu().numpy()

    blurred_imgs = []
    for img in img_numpy:
        img = np.transpose(img, (1, 2, 0))
        blurred_img = cv2.GaussianBlur(img, ksize=kernel_size, sigmaX=sigma)
        if blurred_img.ndim == 2:  # if the img is gray-scale
            blurred_img = np.expand_dims(blurred_img, axis=-1)
        # Convert back to channels-first format
        blurred_img = np.transpose(blurred_img, (2, 0, 1))
        blurred_imgs.append(blurred_img)

    blurred_img_tensor = torch.from_numpy(np.stack(blurred_imgs)).to(img_tensor.device)
    high_freq_components = img_tensor - blurred_img_tensor

    return high_freq_components


class ResBlock(nn.Module):

    def __init__(self, nch_ker, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(nch_ker, nch_ker, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(nch_ker, nch_ker, kernel_size=3, stride=1, padding=1)
        self.esa = Modified_ESA(n_feats=nch_ker, reduc_ratio=4)

    def forward(self, x):
        res = self.relu(self.conv1(x))
        res = self.conv2(res)
        res = self.esa(res)
        res *= self.res_scale
        res = res + x
        return res


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class MultiReceptionFieldModule(nn.Module):
    def __init__(self, nch_ker, gamma=16, M=5):
        super(MultiReceptionFieldModule, self).__init__()
        self.M = M
        self.resblock = ResBlock(nch_ker)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1x1 = nn.Conv2d(nch_ker, nch_ker, kernel_size=1, stride=1, padding=0)
        self.Dconv1x1_1 = nn.Conv2d(nch_ker, nch_ker // gamma, kernel_size=3, stride=1, padding=1, dilation=1)
        self.Dconv1x1_2 = nn.Conv2d(nch_ker // gamma, nch_ker, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv3x3 = nn.Conv2d(nch_ker, nch_ker, kernel_size=3, stride=1, padding=1)
        self.Dconv3x3_1 = nn.Conv2d(nch_ker, nch_ker // gamma, kernel_size=3, stride=1, padding=3, dilation=3)
        self.Dconv3x3_2 = nn.Conv2d(nch_ker // gamma, nch_ker, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv5x5 = nn.Conv2d(nch_ker, nch_ker, kernel_size=5, stride=1, padding=2)
        self.Dconv5x5_1 = nn.Conv2d(nch_ker, nch_ker // gamma, kernel_size=3, stride=1, padding=5, dilation=5)
        self.Dconv5x5_2 = nn.Conv2d(nch_ker // gamma, nch_ker, kernel_size=3, stride=1, padding=5, dilation=5)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()

        self.f_fusion = nn.Sequential(*[nn.Conv2d(3 * nch_ker, nch_ker, kernel_size=3, stride=1, padding=1),
                                        nn.Conv2d(nch_ker, nch_ker, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(nch_ker, nch_ker, kernel_size=3, stride=1, padding=1)])

    def forward(self, x):
        res = x
        for _ in range(self.M):
            res = self.resblock(x)
        branch1 = branch2 = branch3 = res.clone()
        # Adaptive dilated convolutional channel attention (ADCCA)
        # ----------branch1---------------
        branch1 = self.conv1x1(branch1)
        branch_x1 = branch1.clone()
        branch1 = self.maxpool(branch1)
        branch1 = self.Dconv1x1_1(branch1)
        branch1 = self.relu(branch1)
        branch1 = self.Dconv1x1_2(branch1)
        branch1 = self.gap(branch1)
        branch1 = self.sigmoid(branch1)
        branch_x1 *= branch1
        # ----------branch2---------------
        branch2 = self.conv3x3(branch2)
        branch_x2 = branch2.clone()
        branch2 = self.maxpool(branch2)
        branch2 = self.Dconv3x3_1(branch2)
        branch2 = self.relu(branch2)
        branch2 = self.Dconv3x3_2(branch2)
        branch2 = self.gap(branch2)
        branch2 = self.sigmoid(branch2)
        branch_x2 *= branch2
        # ----------branch3---------------
        branch3 = self.conv5x5(branch3)
        branch_x3 = branch3.clone()
        branch3 = self.maxpool(branch3)
        branch3 = self.Dconv5x5_1(branch3)
        branch3 = self.relu(branch3)
        branch3 = self.Dconv5x5_2(branch3)
        branch3 = self.gap(branch3)
        branch3 = self.sigmoid(branch3)
        branch_x3 *= branch3

        branch = torch.cat((branch_x1, branch_x2, branch_x3), dim=1)
        branch = self.f_fusion(branch)
        branch = branch + x
        return branch


# # upsample width
# class PixelShuffle1D(nn.Module):
#     def __init__(self, upscale_factor):
#         super(PixelShuffle1D, self).__init__()
#         self.upscale_factor = upscale_factor
#
#     def forward(self, x):
#         batch_size, channels, height, width = x.size()
#         channels //= self.upscale_factor
#         out_width = width * self.upscale_factor
#
#         # Reshape and transpose
#         x = x.contiguous().view(batch_size, channels, self.upscale_factor, height, width)
#         x = x.permute(0, 1, 3, 4, 2).contiguous()
#
#         # Merge the last dimension with width
#         x = x.view(batch_size, channels, height, out_width)
#         return x
#
# class Upsample_horizontal(nn.Sequential):
#     def __init__(self, scale, n_feats, bn=False, act=False, bias=True):
#         m = []
#         if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
#             for _ in range(int(math.log(scale, 2))):
#                 m.append(nn.Conv2d(n_feats, 2 * n_feats, 3, 1, 1, bias=bias))
#                 m.append(PixelShuffle1D(2))
#                 if bn: m.append(nn.BatchNorm2d(n_feats))
#
#                 if act == 'relu':
#                     m.append(nn.ReLU(True))
#                 elif act == 'prelu':
#                     m.append(nn.PReLU(n_feats))
#
#         elif scale == 3:
#             m.append(nn.Conv2d(n_feats, 3 * n_feats, 3, 1, 1, bias=bias))
#             m.append(PixelShuffle1D(3))
#             if bn: m.append(nn.BatchNorm2d(n_feats))
#
#             if act == 'relu':
#                 m.append(nn.ReLU(True))
#             elif act == 'prelu':
#                 m.append(nn.PReLU(n_feats))
#         else:
#             raise NotImplementedError
#
#         super(Upsample_horizontal, self).__init__(*m)


# class UPA_horizontal(nn.Module):
#     def __init__(self, nch_ker, up_scale):
#         super(UPA_horizontal, self).__init__()
#         self.up_scale = up_scale
#         self.nch_ker = nch_ker
#         self.conv = nn.Conv2d(nch_ker, nch_ker, kernel_size=3, stride=1, padding=1)
#         self.PA = PA(nch_ker)
#         self.HRconv = nn.Conv2d(nch_ker, nch_ker, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, x):
#         if (self.up_scale & (self.up_scale - 1)) == 0:  # Is scale = 2^n?
#             for _ in range(int(math.log(self.up_scale, 2))):
#                 # x = F.interpolate(x, scale_factor=(1, 2), mode='nearest')
#                 x = F.interpolate(x, scale_factor=(1, 2), mode='bilinear', align_corners=False)
#                 x = self.conv(x)
#                 x = self.PA(x)
#                 x = self.HRconv(x)
#         elif self.up_scale == 3:
#             # x = F.interpolate(x, scale_factor=(1, self.up_scale), mode='nearest')
#             x = F.interpolate(x, scale_factor=(1, self.up_scale), mode='bilinear', align_corners=False)
#             x = self.conv(x)
#             x = self.PA(x)
#             x = self.HRconv(x)
#         else:
#             raise NotImplementedError
#
#         return x


class ReconstructionModule(nn.Module):
    def __init__(self, nch_in, nch_out):
        super(ReconstructionModule, self).__init__()
        self.conv1 = nn.Conv2d(nch_in, nch_in, kernel_size=3, stride=1, padding=1)
        self.resblock1 = ResBlock(nch_in)
        self.conv2 = nn.Conv2d(nch_in, nch_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.resblock1(x)
        x = self.conv2(x)

        return x


class FusionModule(nn.Module):
    def __init__(self, nch_in, nch_out):
        super(FusionModule, self).__init__()
        self.conv1 = nn.Conv2d(nch_in, nch_in, kernel_size=3, stride=1, padding=1)
        self.Dconv2x2 = nn.Conv2d(nch_in, nch_in, kernel_size=3, stride=1, padding=2, dilation=2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(nch_in, nch_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Dconv2x2(x)
        x = self.relu(x)
        x = self.Dconv2x2(x)
        x = self.conv2(x)

        return x


# class Fusion(nn.Module):
#     '''Bi-directional Gated Feature Fusion.'''
#
#     def __init__(self, in_channels=56, out_channels=56):
#         super(Fusion, self).__init__()
#
#         self.structure_gate = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels + in_channels, out_channels=out_channels, kernel_size=3, stride=1,
#                       padding=1),
#             nn.Sigmoid()
#         )
#         self.texture_gate = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels + in_channels, out_channels=out_channels, kernel_size=3, stride=1,
#                       padding=1),
#             nn.Sigmoid()
#         )
#         self.structure_gamma = nn.Parameter(torch.zeros(1))
#         self.texture_gamma = nn.Parameter(torch.zeros(1))
#         self.conv = nn.Conv2d(in_channels=112, out_channels=56, kernel_size=1)
#
#     def forward(self, texture_feature, structure_feature):
#         energy = torch.cat((texture_feature, structure_feature), dim=1)
#
#         gate_structure_to_texture = self.structure_gate(energy)
#         gate_texture_to_structure = self.texture_gate(energy)
#
#         texture_feature = texture_feature + self.texture_gamma * (gate_structure_to_texture * structure_feature)
#         structure_feature = structure_feature + self.structure_gamma * (gate_texture_to_structure * texture_feature)
#         out = torch.cat((texture_feature, structure_feature), dim=1)
#         out = self.conv(out)
#
#         return out


class HASPN(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, up_scale=2, G=20):
        super(HASPN, self).__init__()
        self.G = G
        self.conv_in = nn.Conv2d(nch_in, nch_ker, kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(nch_ker, nch_ker, kernel_size=3, stride=1, padding=1)
        self.MFM = MultiReceptionFieldModule(nch_ker)
        self.RecM = ReconstructionModule(nch_ker, nch_out)
        self.fusionM = FusionModule(2 * nch_out, nch_out)

    def forward(self, LR, LR_T):
        LR = self.conv_in(LR)
        LR_T = self.conv_in(LR_T)
        identity_LR = LR.clone()
        identity_LR_T = LR_T.clone()
        for _ in range(self.G):
            LR = self.MFM(LR)
            LR = LR + identity_LR
            LR_T = self.MFM(LR_T)
            LR_T = LR_T + identity_LR_T
        LR = self.conv(LR)
        LR = LR + identity_LR
        LR_T = self.conv(LR_T)
        LR_T = LR_T + identity_LR_T
        
        LR = F.interpolate(LR, scale_factor=(1, self.up_scale), mode='bilinear', align_corners=True)
        HR_C = self.RecM(LR)
        LR_T = F.interpolate(LR_T, scale_factor=(1, self.up_scale), mode='bilinear', align_corners=True)
        HR_T = self.RecM(LR_T)
        
        HR = torch.cat([HR_C, HR_T], dim=1)
        HR = self.fusionM(HR)

        return HR_C, HR_T, HR


if __name__ == '__main__':
    model = HASPN(nch_in=1, nch_out=1, up_scale=4)
    print(model)
    x = torch.randn(1, 1, 256, 64)
    t = torch.randn(1, 1, 256, 64)
    _, _, HR = model(x, t)
    print(HR.size())
