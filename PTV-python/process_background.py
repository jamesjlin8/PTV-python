import os
from pathlib import Path
from config import ImageConfig, BackgroundConfig, TrackingConfig, AnalysisConfig
from image_processing import process_images
import cv2

# Image settings
IMAGE_DIR = "/Users/jameslin/Downloads/Helgeson/PTV/PVA/Q2_0.1mlmin_In_Q1_0.1mlmin_Wi_run1"  # Directory containing images
BASE_NAME = "frame_"  # Base name of image files
NUM_IMAGES = 500  # Number of images to process (None for all)
OUTPUT_DIR = "/Users/jameslin/Downloads/Helgeson/PTV/PVA/Q2_0.1mlmin_In_Q1_0.1mlmin_Wi_run1/background"  # Directory to save results

# Background settings
BACKGROUND_METHOD = "median"  # Options: "static", "median", "mean"
BACKGROUND_WINDOW_SIZE = 500  # Window size for rolling background

def main():
    """
    Process images with background subtraction and save background and maximum intensity images.
    """
    try:
        # Create configuration objects
        image_config = ImageConfig(
            image_dir=IMAGE_DIR,
            base_name=BASE_NAME,
            num_images=NUM_IMAGES,
            image_resolution=(1024, 544),  # Required but not used for background
            length_per_pixel=1,            # Required but not used for background
            frame_rate=1,                  # Required but not used for background
            length_unit="mm",              # Required but not used for background
            time_unit="sec"               # Required but not used for background
        )
        
        background_config = BackgroundConfig(
            method=BACKGROUND_METHOD,
            background_image=None,
            window_size=BACKGROUND_WINDOW_SIZE
        )
        
        # Minimal tracking config (required but not used for background)
        tracking_config = TrackingConfig(
            diameter=11,
            minmass=10,
            search_range=4,
            memory=10,
            min_track_length=10
        )
        
        config = AnalysisConfig(
            image=image_config,
            background=background_config,
            tracking=tracking_config,
            output=OUTPUT_DIR
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Process images and get background-subtracted images
        print("Processing images...")
        processed_images, background = process_images(config)
        
        # Save background image
        background_path = os.path.join(OUTPUT_DIR, "background.png")
        cv2.imwrite(background_path, background)
        print(f"Background saved to: {background_path}")
        
        print("\nProcessing complete!")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    main() 