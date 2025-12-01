import argparse
import numpy as np
import pydicom
import glob
from tqdm import tqdm

# Script for loading the AAPM dataset and saving the volumes of full dose images

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--save_path', type=str, required=True)
args = parser.parse_args()

# Need to search for full dose images
# Only find dcm files with 'Full Dose Images' in the path
# Example suffix: L004/12-23-2021-40641/1.000000-Full Dose Images-63186/1-01.dcm
files = sorted(glob.glob(args.data_path + '*/*/*-Full Dose Images-*/*.dcm'))
volumes = {}

for f in tqdm(files):
    parts = f.split('/')
    patient_id = parts[-4]  # e.g., L004
    if patient_id not in volumes:
        volumes[patient_id] = []
    img = pydicom.dcmread(f).pixel_array
    volumes[patient_id].append(img)

# Save each volume as a numpy array
for patient_id, slices in volumes.items():
    vol = np.stack(slices, axis=0)
    vol = vol.astype(np.float32)
    vol -= vol.min()
    vol /= vol.max()
    np.save(args.save_path + f'{patient_id}_full_dose_{len(vol)}_slices.npy', vol)
    print(f'Saved volume for {patient_id} with {len(vol)} slices.')