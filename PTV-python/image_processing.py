import pims
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
from typing import List, Union, Optional, Tuple
from config import BackgroundConfig, AnalysisConfig

def load_image_sequence(image_dir: str, base_name: str, num_images: int) -> pims.ImageSequence:
    """
    Load sequence of images using PIMS.
    
    Args:
        image_dir: Directory containing images
        base_name: Base name of image files
        num_images: Number of images to load
        
    Returns:
        PIMS ImageSequence object
    """
    # Construct the file pattern for PIMS
    pattern = str(Path(image_dir) / f"{base_name}*.tif")
    print(f"Loading images matching pattern: {pattern}")
    
    # Load images using PIMS
    images = pims.open(pattern)
    
    # Set the number of frames to process
    if num_images is not None:
        images = images[:num_images]

    return images

def create_background(images: Union[pims.ImageSequence, List[np.ndarray]], config: BackgroundConfig) -> np.ndarray:
    """
    Create background image based on specified method.
    
    Args:
        images: PIMS ImageSequence or list of image arrays
        config: Background configuration
        
    Returns:
        Background image as numpy array
    """
    print(f"Creating background using {config.method} method...")
    
    # Convert to list if PIMS sequence
    if isinstance(images, pims.ImageSequence):
        frames = list(images)
    else:
        frames = images
    
    if config.method == "static" and config.background_image is not None:
        # Load static background image
        background = plt.imread(config.background_image)
        return background
    
    elif config.method == "median":
        # Calculate median background
        if len(frames) > config.window_size:
            # Use a subset of frames for efficiency
            step = len(frames) // config.window_size
            subset = frames[::step][:config.window_size]
            background = np.median(subset, axis=0)
        else:
            background = np.median(frames, axis=0)
        return background
    
    elif config.method == "mean":
        # Calculate mean background
        if len(frames) > config.window_size:
            # Use a subset of frames for efficiency
            step = len(frames) // config.window_size
            subset = frames[::step][:config.window_size]
            background = np.mean(subset, axis=0)
        else:
            background = np.mean(frames, axis=0)
        return background
    
    else:
        # Default to black background
        return np.zeros_like(frames[0])

def subtract_background(images: Union[pims.ImageSequence, List[np.ndarray]], background: np.ndarray) -> List[np.ndarray]:
    """
    Subtract background from images.
    
    Args:
        images: PIMS ImageSequence or list of image arrays
        background: Background image to subtract
        
    Returns:
        List of processed images
    """
    print("Subtracting background from images...")
    
    # Convert to list if PIMS sequence
    if isinstance(images, pims.ImageSequence):
        frames = list(images)
    else:
        frames = images
    
    # Subtract background from each frame
    processed_frames = []
    for frame in frames:
        # Ensure frame and background have the same data type
        frame = frame.astype(np.float32)
        background = background.astype(np.float32)
        
        # Subtract background
        processed = frame - background
        
        # Clip negative values to 0
        processed = np.clip(processed, 0, None)
        
        processed_frames.append(processed)
    
    return processed_frames

def create_max_image(processed_images: List[np.ndarray], output_path: str) -> None:
    """
    Create and save a maximum intensity image of all processed frames.
    
    Args:
        processed_images: List of processed image arrays
        output_path: Path to save the maximum intensity image
    """
    print("Creating maximum intensity image of all processed frames...")
    
    # Calculate the maximum intensity image
    max_image = np.max(processed_images, axis=0)
    
    # Normalize to 0-255 range for better visualization
    max_image = (max_image - max_image.min()) / (max_image.max() - max_image.min()) * 255
    max_image = max_image.astype(np.uint8)
    
    # Save the maximum intensity image
    plt.imsave(output_path, max_image, cmap='gray')
    print(f"Maximum intensity image saved to {output_path}")

def process_images(config: AnalysisConfig) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Process images according to configuration.
    
    Args:
        config: Analysis configuration
        
    Returns:
        Tuple of (processed images, background image, maximum intensity image)
    """
    # Load image sequence
    images = load_image_sequence(
        config.image.image_dir,
        config.image.base_name,
        config.image.num_images
    )
    
    # Create background
    background = create_background(images, config.background)
    
    # Subtract background
    processed_images = subtract_background(images, background)
    
    # Create maximum intensity image
    max_intensity = np.max(processed_images, axis=0)
    
    # Save maximum intensity image
    os.makedirs(config.output, exist_ok=True)
    plt.imsave(os.path.join(config.output, 'max_intensity.png'), max_intensity, cmap='gray')
    
    return processed_images, background, max_intensity 