'''
Enhancing the contrast of the en face image slice
'''

import cv2
img = cv2.imread('slice_of_brain.png', cv2.IMREAD_GRAYSCALE) # change filename accordingly

# CLAHE: Contrast Limited Adaptive Histogram Equalization - to enhance contrast in the image
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8)) # can change clipLimit but setting a large num can make noises
blurred = cv2.GaussianBlur(img, (3,3), 0) # denoising
enhanced_img = clahe.apply(blurred)

# Save it to the file
cv2.imwrite('enhanced.png', enhanced_img)
