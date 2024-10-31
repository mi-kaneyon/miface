import cv2
import mediapipe as mp
from PIL import Image, ImageEnhance
import numpy as np

def enhance_resize_and_white_background(input_path, output_path, scale_factor=2.0, sharpness=2.0):
    # Set up MediaPipe for segmentation
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        
        # Load the image and convert it to RGB
        img = cv2.imread(input_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect the person in the image with MediaPipe
        results = selfie_segmentation.process(img_rgb)
        mask = results.segmentation_mask

        # Create a white background where the mask is applied
        condition = np.stack((mask,) * 3, axis=-1) > 0.5
        white_background = np.ones(img_rgb.shape, dtype=np.uint8) * 255
        img_with_white_bg = np.where(condition, img_rgb, white_background)

        # Convert to Pillow format and apply sharpness adjustment
        pil_img = Image.fromarray(img_with_white_bg)
        enhancer = ImageEnhance.Sharpness(pil_img)
        img_sharp = enhancer.enhance(sharpness)  # Adjust sharpness based on parameter
        
        # Resize the image with the specified scaling factor (supports float)
        new_size = (int(img_sharp.width * scale_factor), int(img_sharp.height * scale_factor))
        img_resized = img_sharp.resize(new_size, Image.LANCZOS)
        
        # Save the processed image
        img_resized.save(output_path)
        print(f"Processing completed. Saved to: {output_path}")

# Example usage
input_path = "input.jpg"  # Path to the input image
output_path = "output.png"     # Path to save the output image
enhance_resize_and_white_background(input_path, output_path, scale_factor=2.5, sharpness=3.0)  # Adjust sharpness and scaling factor
