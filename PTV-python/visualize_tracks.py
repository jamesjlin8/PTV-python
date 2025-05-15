import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from visualization import (
    plot_tracks,
    plot_velocity_magnitude,
    plot_flow_type_parameter,
    plot_magnitude_velocity_gradient,
    plot_deformation_rate
)

# Configuration settings
TRACKS_PATH = "/Users/jameslin/Downloads/Helgeson/PTV/PVA/Q2_0.1mlmin_In_Q1_0.1mlmin_Wi_run1/results/tracks.csv"  # Path to tracks.csv file
OUTPUT_DIR = "/Users/jameslin/Downloads/Helgeson/PTV/PVA/Q2_0.1mlmin_In_Q1_0.1mlmin_Wi_run1/results"  # Directory to save results
BACKGROUND_PATH = "/Users/jameslin/Downloads/Helgeson/PTV/PVA/Q2_0.1mlmin_In_Q1_0.1mlmin_Wi_run1/results/background.png"  # Path to background image (optional)
MAX_INTENSITY_PATH = "/Users/jameslin/Downloads/Helgeson/PTV/PVA/Q2_0.1mlmin_In_Q1_0.1mlmin_Wi_run1/results/max_intensity.png"  # Path to maximum intensity image (optional)
IMAGE_SIZE = (340, 340)  # Image size (width, height) - set to None to auto-detect
LENGTH_PER_PIXEL = 1/250  # Physical length per pixel in mm

def load_image(image_path: Optional[str]) -> Optional[np.ndarray]:
    """
    Load image if provided.
    
    Args:
        image_path: Path to image
        
    Returns:
        Image as numpy array or None
    """
    if image_path and os.path.exists(image_path):
        return plt.imread(image_path)
    return None

def visualize_from_csv(
    tracks_path: str,
    output_dir: str,
    background_path: Optional[str] = None,
    max_intensity_path: Optional[str] = None,
    image_size: Optional[Tuple[int, int]] = None,
    length_per_pixel: float = 0.1,
    bin_size: int = 8
) -> None:
    """
    Visualize tracks from an existing CSV file.
    
    Args:
        tracks_path: Path to tracks.csv file
        output_dir: Directory to save results
        background_path: Optional path to background image
        max_intensity_path: Optional path to maximum intensity image
        image_size: Optional tuple of (width, height) for image size
        length_per_pixel: Physical length per pixel in mm
        bin_size: Size of bins for averaging
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tracks
    print(f"Loading tracks from {tracks_path}...")
    tracks = pd.read_csv(tracks_path)
    
    # Load images if provided
    background = load_image(background_path)
    max_intensity = load_image(max_intensity_path)
    
    # Determine image size if not provided
    if image_size is None:
        # Use the maximum x and y coordinates as image size
        width = int(tracks['x'].max()) + 100
        height = int(tracks['y'].max()) + 100
        image_size = (width, height)
        print(f"Using inferred image size: {image_size}")
    
    # Calculate center coordinates
    center_x = image_size[0] / 2
    center_y = image_size[1] / 2
    
    # Save images if provided
    if background is not None:
        plt.imsave(os.path.join(output_dir, 'background.png'), background, cmap='gray')
    
    if max_intensity is not None:
        plt.imsave(os.path.join(output_dir, 'max_intensity.png'), max_intensity, cmap='gray')
    
    # Plot tracks
    plot_tracks(tracks, image_size, os.path.join(output_dir, 'tracks.png'), max_intensity, length_per_pixel)
    
    # Plot velocity magnitude
    plot_velocity_magnitude(tracks, image_size, os.path.join(output_dir, 'velocity_magnitude.png'), max_intensity, bin_size, length_per_pixel)
    
    # Plot flow type parameter
    plot_flow_type_parameter(tracks, image_size, os.path.join(output_dir, 'flow_type_parameter.png'), max_intensity, bin_size, length_per_pixel)
    
    # Plot magnitude velocity gradient tensor
    plot_magnitude_velocity_gradient(tracks, image_size, os.path.join(output_dir, 'magnitude_velocity_gradient.png'), max_intensity, bin_size, length_per_pixel)
    
    # Plot deformation rate
    plot_deformation_rate(tracks, image_size, os.path.join(output_dir, 'deformation_rate.png'), max_intensity, bin_size, length_per_pixel)
    
    print(f"\nResults saved to {output_dir}")

def main():
    """
    Visualize tracks from a CSV file using the configuration settings defined at the top of the file.
    """
    # Visualize tracks using the configuration settings
    visualize_from_csv(
        tracks_path=TRACKS_PATH,
        output_dir=OUTPUT_DIR,
        background_path=BACKGROUND_PATH,
        max_intensity_path=MAX_INTENSITY_PATH,
        image_size=IMAGE_SIZE,
        length_per_pixel=LENGTH_PER_PIXEL,
        bin_size=8
    )

if __name__ == "__main__":
    main() 