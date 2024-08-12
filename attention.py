import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, stride=1, padding=None, bias=True, groups=1):
    if not padding and stride == 1:
        padding = kernel_size // 2
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)

class Modified_ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, n_feats, reduc_ratio, conv=default_conv):
        super(Modified_ESA, self).__init__()
        f = n_feats // reduc_ratio
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        self.attention = self.sigmoid(c4)
        return x * self.attention

    def get_attention(self):
        return self.attention




def mean_channels(F):
    """
    Compute the mean of each channel in the feature map.

    Parameters:
    F (tensor): The input feature map with shape (B, C, H, W).

    Returns:
    tensor: The mean value of each channel.
    """
    assert (F.dim() == 4)  # Ensure the feature map has 4 dimensions [B, C, H, W]
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)  # Sum over the spatial dimensions H and W
    return spatial_sum / (F.size(2) * F.size(3))  # Divide by the spatial area to get the mean


def stdv_channels(F):
    """
    Compute the standard deviation of each channel in the feature map.

    Parameters:
    F (tensor): The input feature map with shape (B, C, H, W).

    Returns:
    tensor: The standard deviation of each channel.
    """
    assert (F.dim() == 4)  # Ensure the feature map has 4 dimensions [B, C, H, W]
    F_mean = mean_channels(F)  # Compute the mean of each channel
    # Compute variance and standard deviation over the spatial dimensions H and W
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)  # Take the square root to get the standard deviation


# Contrast Channel Attention
class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        """
        Initializes the Contrast-aware Channel Attention (CCA) layer.

        Parameters:
        channel (int): The number of channels in the input feature map.
        reduction (int): The reduction ratio for the channel attention.
        """
        super(CCALayer, self).__init__()

        # Function to compute the contrast (standard deviation)
        self.contrast = stdv_channels

        # Adaptive average pooling to reduce spatial dimensions to 1x1 per channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Convolutional layers to learn the channel-wise attention from the contrast and average pooled features
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass of the CCA layer.

        Parameters:
        x (tensor): The input feature map with shape (B, C, H, W).

        Returns:
        tensor: The output feature map after applying channel attention.
        """
        # Compute the channel-wise contrast and mean, then combine them
        y = self.contrast(x) + self.avg_pool(x)

        # Apply the convolutional layers to learn the attention weights
        y = self.conv_du(y)

        # Apply the attention weights to the input feature map
        return x * y


class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out
