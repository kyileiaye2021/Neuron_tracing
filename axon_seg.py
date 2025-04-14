'''
Axon Segmentation 
Matlab code from Ziv's paper (python version)
'''

# download the libraries first
import numpy as np
import cv2
import tifffile
from skimage.morphology import remove_small_objects

def apply_normalized_filter(image, kernel):
    filtered = cv2.filter2D(image.astype(np.float32), -1, kernel)
    return filtered

'''
creating different filters to detect different directions of axons (vertical, horizontal, diagonals )
'''
def create_directional_kernels():
    F = np.array([-1, -1, -1, 1, 1, 1, -1, -1, -1], dtype=np.float32)
    kernel = np.tile(F[:, None], (1, F.size))
    
    def normalize(k):
        k = k - np.mean(k)
        return k / np.sqrt(np.sum(k**2))

    top_bottom = normalize(kernel)
    left_right = normalize(kernel.T)
    
    diag1 = normalize(cv2.warpAffine(kernel, cv2.getRotationMatrix2D((4, 4), 45, 1), (9, 9)))
    diag2 = normalize(cv2.warpAffine(kernel, cv2.getRotationMatrix2D((4, 4), -45, 1), (9, 9)))
    
    return top_bottom, left_right, diag1, diag2

def OCT2Axons_py(tiff_path, axon_threshold_percentile=95, min_length=30):
    # Load 3D TIFF
    volume = tifffile.imread(tiff_path)
    h, w, d = volume.shape
    volume = volume.astype(np.uint8)

    # Initialize output arrays
    segmented = np.zeros((h, w, d), dtype=bool)
    
    # Directional filters
    top_bottom, left_right, diag1, diag2 = create_directional_kernels()

    for i in range(d): # for each slice, apply the filters
        img = volume[:, :, i]

        # Apply each directional filter
        filtered_tb = apply_normalized_filter(img, top_bottom)
        filtered_lr = apply_normalized_filter(img, left_right)
        filtered_d1 = apply_normalized_filter(img, diag1)
        filtered_d2 = apply_normalized_filter(img, diag2)

        # Threshold each filtered image using the percentile
        mask_tb = filtered_tb > np.percentile(filtered_tb, axon_threshold_percentile)
        mask_lr = filtered_lr > np.percentile(filtered_lr, axon_threshold_percentile)
        mask_d1 = filtered_d1 > np.percentile(filtered_d1, axon_threshold_percentile)
        mask_d2 = filtered_d2 > np.percentile(filtered_d2, axon_threshold_percentile)

        # Combine masks from all directions
        combined_mask = mask_tb | mask_lr | mask_d1 | mask_d2

        # Remove small objects (like MATLAB's bwareaopen)
        cleaned_mask = remove_small_objects(combined_mask, min_size=min_length)
        
        segmented[:, :, i] = cleaned_mask

    return segmented.astype(np.uint8) * 255  # for saving as binary TIFF

# change the file names accordingly
axon_mask = OCT2Axons_py('enhanced_3d_volume.tif', axon_threshold_percentile=95, min_length=30) 
tifffile.imwrite('axon_segmented.tif', axon_mask)