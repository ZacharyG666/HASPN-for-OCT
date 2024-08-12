import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from feature_extractor_vgg import *


class Overall_Loss(nn.Module):
    def __init__(self):
        super(Overall_Loss, self).__init__()
        self.loss = Loss()

    def forward(self, HR_C, HR_T, HR, GT, GT_T):
        l_1 = self.loss(HR_C, GT)
        l_2 = self.loss(HR_T, GT_T)
        l_3 = self.loss(HR, GT)

        return l_1 + l_2 + l_3

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.L1Loss = nn.L1Loss()
        self.MSELoss = nn.MSELoss()
        self.grad_loss = GradientLoss()  # sobel
        # self.grad_loss = gradientloss()  # Laplacian
        vgg = vgg19(pretrained=True)
        self.loss_network = nn.Sequential(*list(vgg.features.children())[:12])

    def forward(self, out_images, target_images):
        pixel_loss = self.L1Loss(out_images, target_images)

        perceptual_loss = self.MSELoss(self.loss_network(out_images), Variable(self.loss_network(target_images).data,
                                                                               requires_grad=False))

        grad_loss = self.grad_loss(out_images, target_images)

        return pixel_loss + perceptual_loss + grad_loss


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        # Define the kernels for gradient computation in x and y direction
        self.kernel_x = torch.tensor([[[[-1, 0, 1],
                                        [-2, 0, 2],
                                        [-1, 0, 1]]]], dtype=torch.float32)

        self.kernel_y = torch.tensor([[[[-1, -2, -1],
                                        [0,  0,  0],
                                        [1,  2,  1]]]], dtype=torch.float32)

    def image_gradients(self, image):
        # Ensure the input tensor is on the same device as the model
        self.kernel_x = self.kernel_x.to(image.device)
        self.kernel_y = self.kernel_y.to(image.device)

        # Compute the gradients for the output and target images
        grad_x = F.conv2d(image, self.kernel_x, padding=1)
        grad_y = F.conv2d(image, self.kernel_y, padding=1)

        return grad_x, grad_y

    def forward(self, out_images, target_images):
        grad_x_out, grad_y_out = self.image_gradients(out_images)
        grad_x_target, grad_y_target = self.image_gradients(target_images)

        # Calculate the gradient loss as the L1 norm of the gradient difference
        grad_loss = torch.mean(torch.abs(grad_x_out - grad_x_target) + torch.abs(grad_y_out - grad_y_target))
        return grad_loss


# Laplacian
class LaplaceAlogrithm(nn.Module):
    def __init__(self):
        super(LaplaceAlogrithm, self).__init__()

    def forward(self, image):
        assert torch.is_tensor(image) is True

        laplace_operator = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        laplace_operator = torch.from_numpy(laplace_operator).unsqueeze(0).unsqueeze(0).to(image.device)
        # laplace_operator = torch.from_numpy(laplace_operator).unsqueeze(0)#if no cuda
        image = image - F.conv2d(image, laplace_operator, padding=1, stride=1)
        return image

class gradientloss(nn.Module):
    def __init__(self):
        super(gradientloss, self).__init__()
        self.LaplaceAlogrithm = LaplaceAlogrithm()

    def forward(self, preds, labels):
        grad_img1 = self.LaplaceAlogrithm(preds)
        gt = self.LaplaceAlogrithm(labels)
        gt.requires_grad_(False)
        g_loss = F.l1_loss(grad_img1, gt, size_average=True, reduce=True)
        return g_loss
