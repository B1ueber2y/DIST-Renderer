import torch
import torch.nn as nn

def loss_ssim(img1, img2):
    '''
    Inputs:
    - img1	torch.Tensor (B, 3, H, W)
    - img2	torch.Tensor (B, 3, H, W)
    Returns:
    - loss_ssim	torch.Tensor (B)
    Examples:
    - For img1 and img2 both shaped (H, W, 3), you could use 
    - 'loss_ssim(img1.permute(2,0,1)[None,:,:,:], img2.permute(2,0,1)[None,:,:,:])[0]'
    '''
    ssim = SSIM(img1, img2)
    loss_ssim = torch.clamp((1.0 - ssim) / 2.0, 0, 1) 
    loss_ssim = loss_ssim.mean((1,2,3))
    return loss_ssim

def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1, padding=1)(x)
    mu_y = nn.AvgPool2d(3, 1, padding=1)(y)

    sigma_x = nn.AvgPool2d(3, 1, padding=1)(x ** 2) - mu_x ** 2
    sigma_y = nn.AvgPool2d(3, 1, padding=1)(y ** 2) - mu_y ** 2
    sigma_xy = nn.AvgPool2d(3, 1, padding=1)(x * y) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / (SSIM_d + 1e-12)
    return SSIM

