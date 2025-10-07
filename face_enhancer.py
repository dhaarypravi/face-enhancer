import cv2
import numpy as np
import os
import sys

# Define the input and output filenames
INPUT_IMAGE_NAME = 'blurred_face.jpg'
OUTPUT_IMAGE_NAME = 'enhanced_face_final_quality.jpg'

# --- Configuration Parameters for Tuning ---
# Denoising: The higher the h/hColor, the straonger the denoise.
DENOISE_H = 5
# Sharpening: Controls the intensity of the unsharp mask.
SHARPEN_WEIGHT = 1.5
BLUR_WEIGHT = -0.5
# Contrast: Controls the aggressiveness of CLAHE.
CLAHE_CLIP_LIMIT = 2.0 

def main():
    print("--- Refined Image Enhancement Pipeline ---")
    
    # Load image
    img = cv2.imread(INPUT_IMAGE_NAME)
    
    if img is None:
        print(f"Error: Could not load image '{INPUT_IMAGE_NAME}'.")
        print(f"Ensure the image is in the same folder and named '{INPUT_IMAGE_NAME}'.")
        # Exit gracefully if the image isn't found
        sys.exit(1)

    # Step 1: Denoising (Slightly stronger to remove pre-existing noise)
    # The higher the DENOISE_H (h and hColor), the more noise is removed, but detail is also lost.
    # The default values (3, 3, 7, 21) are often too weak for very blurry/noisy photos.
    print(f"1. Applying Denoising (h={DENOISE_H})...")
    denoised = cv2.fastNlMeansDenoisingColored(img, None, DENOISE_H, DENOISE_H, 7, 21)

    # Step 2: Sharpening using the Unsharp Mask technique
    # We use a weight of 1.5 for the original and -0.5 for the blurred version (1.5 + (-0.5) = 1.0 overall)
    # The kernel size (0, 0) tells OpenCV to determine the size based on sigma (1.0).
    print(f"2. Applying Unsharp Mask (Weight={SHARPEN_WEIGHT})...")
    gaussian = cv2.GaussianBlur(denoised, (0, 0), 1.0)
    sharpened = cv2.addWeighted(denoised, SHARPEN_WEIGHT, gaussian, BLUR_WEIGHT, 0)
    
    # Step 3: Enhance contrast with CLAHE
    # CLAHE is essential for correcting overexposed or flat images.
    print(f"3. Applying Contrast Enhancement (CLAHE Clip={CLAHE_CLIP_LIMIT})...")
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # clipLimit determines how aggressive the contrast enhancement is. 
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    contrast = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Step 4: Final output is the contrast-enhanced image. 
    # Removed the final kernel filter2D step as it caused excessive edge enhancement.
    final = contrast

    # Step 5: Show and save
    cv2.imshow("Original Image", img)
    cv2.imshow("Enhanced Face (Final Quality)", final)
    cv2.imwrite(OUTPUT_IMAGE_NAME, final)
    print(f"\nEnhanced image saved as {OUTPUT_IMAGE_NAME}")

    print("Close the image windows to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
