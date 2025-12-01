import torch
import torch.nn.functional as F
import numpy as np
import glob
from src.models.unet import UNet
from src.physics.pbct import PBCT
from src.physics.cbct import CBCT
from tqdm import tqdm
import pydicom
from torchmetrics.functional import peak_signal_noise_ratio as PSNR
import os
import argparse
import re

torch.manual_seed(123)


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/home/ebell34/Tada-DIP/aapm/full_1mm/')
parser.add_argument('--results_path', type=str, default='/home/ebell34/Tada-DIP/results/')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--base_filters', type=int, default=16)
parser.add_argument('--num_views', type=int, default=30)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=0.01)
parser.add_argument('--gamma', type=float, default=0.01)
parser.add_argument('--geometry', type=str, default='pbct', choices=['pbct', 'cbct'])
parser.add_argument('--case_num', type=str, default='L067')
args = parser.parse_args()

# Find next version number for results
os.makedirs(args.results_path, exist_ok=True)
existing_versions = [d for d in os.listdir(args.results_path) if re.match(r'version_\d+', d)]
if existing_versions:
    version_nums = [int(re.findall(r'version_(\d+)', d)[0]) for d in existing_versions]
    version = max(version_nums) + 1
else:
    version = 0
version_dir = os.path.join(args.results_path, f'version_{version}')
print(f'Saving results to {version_dir}')
os.makedirs(version_dir, exist_ok=True)

# Data loading
files = sorted(glob.glob(f'{args.data_path}/{args.case_num}/full_1mm/*'))
vol = []
for f in tqdm(files):
    img = pydicom.dcmread(f).pixel_array
    vol.append(img)
vol = np.stack(vol, axis=0)
vol = (vol - vol.min()) / (vol.max() - vol.min())
vol = torch.from_numpy(vol).float()
vol = vol[vol.shape[0]//2-256:vol.shape[0]//2+256,:,:]

x = vol[:512].unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
x = F.interpolate(x, size=(256,256,256), mode='trilinear', align_corners=False)
x = x.squeeze(1)
x = x[:,:]

# Device and model setup
device = torch.device(args.device)
x = x.to(device)
n_rows = x.shape[1]
n_cols = x.shape[2]
model = UNet(1, 1, depth=7, dim=3, base_filters=args.base_filters).to(device)
if args.geometry == 'pbct':
    ct = PBCT(args.num_views, n_rows, n_cols, device=device)
else:
    ct = CBCT(args.num_views, n_rows, n_cols, device=device)
y = ct.A(x)
fbp = ct.A_pinv(y)
output = fbp.clone()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
z = torch.randn(1, 1, 256, 256, 256).to(device)

dc_losses = []
psnrs = []
fbp_psnr = PSNR(fbp, x, data_range=(0.0, 1.0)).item()

psnrs = []
dc_losses = []
fbp_psnr = PSNR(fbp, x, data_range=(0.0, 1.0)).item()
out_avg = output.clone()
best_psnr = -float('inf')
best_output = None

num_steps = 50_000

for step in tqdm(range(num_steps)):
    
    optimizer.zero_grad()
    noise = args.alpha * z.abs().max() * torch.randn_like(z)
    out = model(z + noise)
    y_pred = ct.A(out.squeeze(1))
    dc_loss = F.l1_loss(y_pred, y) + args.beta * F.l1_loss(out, z)
    dc_loss.backward()
    optimizer.step()
    z = (1-args.gamma) * z + args.gamma * out.detach()
    out_avg = 0.99 * out_avg + 0.01 * out if step > 0 else out

    with torch.no_grad():
        psnr = PSNR(out_avg[0,0], x, data_range=(0.0, 1.0)).item()
        psnrs.append(psnr)
        dc_losses.append(dc_loss.item())
        if psnr > best_psnr:
            best_psnr = psnr
            best_output = out_avg.clone().cpu()
            torch.save(best_output, os.path.join(version_dir, 'best_output.pt'))
            torch.save(torch.tensor([best_psnr]), os.path.join(version_dir, 'best_psnr.pt'))

    # Save every 1000 steps for first 10k, then every 10k steps
    if step < 10000:
        save_interval = 1000
    else:
        save_interval = 10000
    if step % save_interval == 0 or step == num_steps - 1:
        # Save intermediate results as torch tensors
        torch.save(out_avg.cpu(), os.path.join(version_dir, f'out_avg_step{step}.pt'))
        torch.save(torch.tensor(psnrs), os.path.join(version_dir, f'psnrs_step{step}.pt'))
        torch.save(torch.tensor(dc_losses), os.path.join(version_dir, f'dc_losses_step{step}.pt'))

print(f'Best PSNR: {best_psnr}')
print(f'Best output saved to {os.path.join(version_dir, "best_output.pt")}')
