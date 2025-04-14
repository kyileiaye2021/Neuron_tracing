'''
Enhancing the contrast of the image and reconstructing back to 3D volume
'''

# before running, download the libraries 
import tifffile
import numpy as np
import cv2
import os

# change filenames and directories accordingly
input_path = 'Reslice of Image_bigbrain2_enhanced_contrast_xy.tif' 
output_path = 'enhanced_3d_volume.tif' 

# Load the volume
volume = tifffile.imread(input_path)

# Normalize
volume = (volume / np.max(volume) * 255).astype(np.uint8)

# CLAHE setup
# can change clipLimit to adjust the contrast but setting a large num can make noises
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) 

# Process each slice
enhanced_volume = np.zeros_like(volume)
for i in range(volume.shape[0]):
    slice_img = volume[i]

    if len(slice_img.shape) != 2:
        raise ValueError(f"Slice {i} is not 2D — shape: {slice_img.shape}")

    clahe_img = clahe.apply(slice_img)
    blurred = cv2.GaussianBlur(clahe_img, (3,3), 0) # denoising
    enhanced_volume[i] = blurred

# Write to TIFF
try:
    tifffile.imwrite(output_path, enhanced_volume, bigtiff=True)
    print(f"Saved enhanced volume to {output_path}")
except Exception as e:
    print(f"Error saving TIFF: {e}")
