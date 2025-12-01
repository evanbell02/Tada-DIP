import argparse
import numpy as np
import glob
import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.physics.pbct import PBCT

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--num_views', type=int, default=30)
parser.add_argument('--separate_slices', action='store_true')
args = parser.parse_args()

files = sorted(glob.glob(args.data_path + '*.npy'))

for f in tqdm(files):
    vol = np.load(f)
    vol = torch.from_numpy(vol).float()
    vol = vol.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    vol = F.interpolate(vol, scale_factor=0.5, mode='trilinear', align_corners=False)
    vol = vol.squeeze(0) # (1, D, H, W)

    ct = PBCT(args.num_views, vol.shape[1], vol.shape[2], device='cpu')
    with torch.no_grad():
        sinogram = ct.A(vol)  # (1, num_views, D, num_rows)
        fbp = ct.A_pinv(sinogram)  # (1, D, H, W)

    if args.separate_slices:
        n_slices = vol.shape[1]
        print(vol.shape, fbp.shape)
        for slice_idx in range(n_slices):
            save_dict = {
                'volume': vol[:, slice_idx].clone(),
                'fbp': fbp[:, slice_idx].clone()
            }
            patient_num = f.split('/')[-1][:4]
            filename = f'{patient_num}_full_dose_slice_{slice_idx:03d}.pt'
            torch.save(save_dict, args.save_path + filename)
    
    else:
        save_dict = {
            'volume': vol,
            'fbp': fbp
        }

        patient_num = f.split('/')[-1][:4]
        filename = f'{patient_num}_full_dose_{vol.shape[1]}_slices.pt'
        torch.save(save_dict, args.save_path + filename)
        print(f'Saved FBP dataset for {filename}.')

