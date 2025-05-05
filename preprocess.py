import os
import subprocess


import numpy as np
import nibabel as nib
import gcsfs
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# Define Google Cloud Storage path
data_dir = "gs://autism_images/MRI_Dataset"  # Replace with your bucket name
categories = ["autistic", "control"]
save_dir = "Processed_Data"

# Create local save directory for processed data
os.makedirs(save_dir, exist_ok=True)

# Initialize Google Cloud Storage filesystem
fs = gcsfs.GCSFileSystem()

def resize_image(img, target_size):
    img = img.astype(np.float32)  # Convert to float32 before resizing
    zoom_factors = [target_size[0] / img.shape[0], target_size[1] / img.shape[1], target_size[2] / img.shape[2]]
    resized_img = zoom(img, zoom_factors, order=3)
    return resized_img

for label, category in enumerate(categories):
    folder_path = f"{data_dir}/{category}"  # GCS path for category

    # List files in GCS directory
    nii_files = fs.ls(folder_path)  

    for file in nii_files:
        if file.endswith(".nii"):
            file_path = f"gs://{file}"  # Full GCS path
            
            # Load MRI image from GCS
            with fs.open(file_path, 'rb') as f:
                img = nib.FileHolder(fileobj=f)
                img = nib.Nifti1Image.from_file_map({'image': img}).get_fdata()

            # Normalize (0 to 1 range)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))

            # Resize
            img = resize_image(img, (160, 256, 256))

            # Save each processed image locally
            np.save(os.path.join(save_dir, f"{os.path.basename(file).replace('.nii', '')}.npy"), img)
            np.save(os.path.join(save_dir, f"{os.path.basename(file).replace('.nii', '')}_label.npy"), np.array(label, dtype=np.int8))

print("Preprocessing Complete! Images saved individually.")
