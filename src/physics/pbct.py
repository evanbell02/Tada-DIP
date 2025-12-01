import torch
import leaptorch
import math

class PBCT:

    def __init__(self, num_views, num_rows, num_cols, device='cpu', angles=None):
        self.img_size = num_cols

        num_cols = math.ceil(2**0.5 * num_cols)
        if num_cols % 2 == 1:
            num_cols += 1

        self.num_rows = num_rows
        self.num_cols = num_cols

        pixelHeight = 1
        pixelWidth = 1
        centerRow = num_rows//2
        centerCol = num_cols//2
        device = torch.device(device)

        if device == torch.device('cpu'):
            self.proj = leaptorch.Projector(forward_project=True, use_static=False, use_gpu=False, gpu_device=device, batch_size=1)
        else:
            self.proj = leaptorch.Projector(forward_project=True, use_static=False, use_gpu=True, gpu_device=device, batch_size=1)
        
        if angles is None:
            phis = self.proj.leapct.setAngleArray(num_views, 180.0)
        else:
            phis = angles

        self.proj.leapct.set_parallelbeam(num_views, num_rows, num_cols, pixelHeight, pixelWidth, centerRow, centerCol, phis)

        self.proj.leapct.set_diameterFOV(1.0e7)

        self.proj.set_default_volume()
        self.proj.allocate_batch_data()
        # self.proj.leapct.set_truncatedScan(True)

    def A(self, x):
        # Need to pad x to be the right size
        # x is [B, D, H, W]
        # Needs to be [B, D, num_cols, num_cols]
        pad = (self.num_cols - x.shape[3]) // 2
        x = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode='constant', value=0)
        x = x.float().contiguous()
        return self.proj(x).clone()

    def A_T(self, y):
        y = y.float().contiguous()
        self.proj.forward_project = False
        x = self.proj(y)
        self.proj.forward_project = True
        # Crop x back to original size
        pad = (x.shape[3] - self.img_size) // 2
        x = x[..., pad:-pad, pad:-pad]
        return x.clone()

    def A_pinv(self, y):
        y = y.float().contiguous()
        fbp = self.proj.fbp(y)
        # Crop fbp back to original size
        pad = (fbp.shape[3] - self.img_size) // 2
        fbp = fbp[..., pad:-pad, pad:-pad]
        return fbp.clone()