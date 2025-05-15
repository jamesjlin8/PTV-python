import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trackpy as tp
from pathlib import Path
from typing import Dict, Optional, Tuple
import os

def bin_data(tracks: pd.DataFrame, bin_size: int = 8, length_per_pixel: float = 1.0) -> pd.DataFrame:
    """
    Bin the data into a grid and calculate average values for each bin.
    
    Args:
        tracks: DataFrame with particle tracks
        bin_size: Size of bins for averaging
        length_per_pixel: Physical length per pixel in mm
        
    Returns:
        DataFrame with binned data
    """
    # Create bin edges
    x_bins = np.arange(tracks['x'].min(), tracks['x'].max() + bin_size, bin_size)
    y_bins = np.arange(tracks['y'].min(), tracks['y'].max() + bin_size, bin_size)
    
    # Digitize the data
    x_indices = np.digitize(tracks['x'], x_bins) - 1
    y_indices = np.digitize(tracks['y'], y_bins) - 1
    
    # Initialize list for binned data
    binned_data = []
    
    # Calculate average values in each bin
    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            # Get particles in this bin
            mask = (x_indices == i) & (y_indices == j)
            bin_particles = tracks[mask]
            
            if len(bin_particles) > 0:
                # Calculate average values for this bin
                binned_data.append({
                    'x': (x_bins[i] + x_bins[i+1]) / 2,
                    'y': (y_bins[j] + y_bins[j+1]) / 2,
                    'speed': bin_particles['speed'].mean(),
                    'ftp': bin_particles['ftp'].mean(),
                    'mvgt': bin_particles['mvgt'].mean(),
                    'deformation_rate': bin_particles['deformation_rate'].mean()
                })
    
    return pd.DataFrame(binned_data)

def plot_tracks(tracks: pd.DataFrame, image_size: Tuple[int, int], output_path: str, background: Optional[np.ndarray] = None, length_per_pixel: float = 1.0) -> None:
    """
    Plot particle tracks on the image.
    
    Args:
        tracks: DataFrame with particle tracks
        image_size: Size of the image (width, height)
        output_path: Path to save the plot
        background: Optional background image
        length_per_pixel: Physical length per pixel in mm
    """
    plt.figure(figsize=(10, 10 * image_size[1] / image_size[0]))
    
    if background is not None:
        plt.imshow(background, cmap='gray', extent=[0, image_size[0]*length_per_pixel, image_size[1]*length_per_pixel, 0])
    
    # Plot tracks
    for particle_id, group in tracks.groupby('particle'):
        plt.plot(group['x']*length_per_pixel, group['y']*length_per_pixel, '-', linewidth=1, alpha=0.5)
    
    plt.xlim(0, image_size[0]*length_per_pixel)
    plt.ylim(image_size[1]*length_per_pixel, 0)  # Invert y-axis to match image coordinates
    plt.axis('equal')  # Ensure equal aspect ratio
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_velocity_magnitude(tracks: pd.DataFrame, image_size: Tuple[int, int], output_path: str, background: Optional[np.ndarray] = None, bin_size: int = 8, length_per_pixel: float = 1.0) -> None:
    """
    Plot velocity magnitude using binned data.
    
    Args:
        tracks: DataFrame with particle tracks
        image_size: Size of the image (width, height)
        output_path: Path to save the plot
        background: Optional background image
        bin_size: Size of bins for averaging
        length_per_pixel: Physical length per pixel in mm
    """
    plt.figure(figsize=(10, 10 * image_size[1] / image_size[0]))
    
    if background is not None:
        plt.imshow(background, cmap='gray', extent=[0, image_size[0]*length_per_pixel, image_size[1]*length_per_pixel, 0])
    
    # Bin the data
    binned_data = bin_data(tracks, bin_size, length_per_pixel)
    
    # Get the maximum velocity magnitude for scaling
    max_speed = binned_data['speed'].max()
    
    # Plot velocity magnitude with full color range using square markers
    scatter = plt.scatter(binned_data['x']*length_per_pixel, binned_data['y']*length_per_pixel, 
                         c=binned_data['speed'], 
                         marker='s',  # Use square markers
                         s=bin_size*4,  # Marker size proportional to bin size
                         cmap='jet', 
                         alpha=0.7, 
                         vmin=0, 
                         vmax=max_speed) #max_speed
    plt.colorbar(scatter, label='Velocity Magnitude (mm/s)')
    
    plt.xlim(0, image_size[0]*length_per_pixel)
    plt.ylim(image_size[1]*length_per_pixel, 0)  # Invert y-axis to match image coordinates
    plt.axis('equal')  # Ensure equal aspect ratio
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_flow_type_parameter(tracks: pd.DataFrame, image_size: Tuple[int, int], output_path: str, background: Optional[np.ndarray] = None, bin_size: int = 8, length_per_pixel: float = 1.0) -> None:
    """
    Plot flow type parameter using binned data.
    
    Args:
        tracks: DataFrame with particle tracks
        image_size: Size of the image (width, height)
        output_path: Path to save the plot
        background: Optional background image
        bin_size: Size of bins for averaging
        length_per_pixel: Physical length per pixel in mm
    """
    plt.figure(figsize=(10, 10 * image_size[1] / image_size[0]))
    
    if background is not None:
        plt.imshow(background, cmap='gray', extent=[0, image_size[0]*length_per_pixel, image_size[1]*length_per_pixel, 0])
    
    # Bin the data
    binned_data = bin_data(tracks, bin_size, length_per_pixel)
    
    # Plot flow type parameter with fixed range from -1 to 1 using square markers
    scatter = plt.scatter(binned_data['x']*length_per_pixel, binned_data['y']*length_per_pixel, 
                         c=binned_data['ftp'], 
                         marker='s',  # Use square markers
                         s=bin_size*4,  # Marker size proportional to bin size
                         cmap='jet', 
                         alpha=0.7, 
                         vmin=-1, 
                         vmax=1)
    plt.colorbar(scatter, label='Flow Type Parameter')
    
    plt.xlim(0, image_size[0]*length_per_pixel)
    plt.ylim(image_size[1]*length_per_pixel, 0)  # Invert y-axis to match image coordinates
    plt.axis('equal')  # Ensure equal aspect ratio
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_magnitude_velocity_gradient(tracks: pd.DataFrame, image_size: Tuple[int, int], output_path: str, background: Optional[np.ndarray] = None, bin_size: int = 8, length_per_pixel: float = 1.0) -> None:
    """
    Plot magnitude velocity gradient tensor using binned data.
    
    Args:
        tracks: DataFrame with particle tracks
        image_size: Size of the image (width, height)
        output_path: Path to save the plot
        background: Optional background image
        bin_size: Size of bins for averaging
        length_per_pixel: Physical length per pixel in mm
    """
    plt.figure(figsize=(10, 10 * image_size[1] / image_size[0]))
    
    if background is not None:
        plt.imshow(background, cmap='gray', extent=[0, image_size[0]*length_per_pixel, image_size[1]*length_per_pixel, 0])
    
    # Bin the data
    binned_data = bin_data(tracks, bin_size, length_per_pixel)

    # Get the maximum velocity gradient tensor for scaling
    max_mvgt = binned_data['mvgt'].max()
    
    # Plot magnitude velocity gradient tensor with fixed color range
    scatter = plt.scatter(binned_data['x']*length_per_pixel, binned_data['y']*length_per_pixel, 
                         c=binned_data['mvgt'], 
                         marker='s',  # Use square markers
                         s=bin_size*4,  # Marker size proportional to bin size
                         cmap='jet', 
                         alpha=0.7, 
                         vmin=0, 
                         vmax=75) #max_mvgt
    plt.colorbar(scatter, label='Magnitude of Velocity Gradient Tensor (s⁻¹)')
    
    plt.xlim(0, image_size[0]*length_per_pixel)
    plt.ylim(image_size[1]*length_per_pixel, 0)  # Invert y-axis to match image coordinates
    plt.axis('equal')  # Ensure equal aspect ratio
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_deformation_rate(tracks: pd.DataFrame, image_size: Tuple[int, int], output_path: str, background: Optional[np.ndarray] = None, bin_size: int = 8, length_per_pixel: float = 1.0) -> None:
    """
    Plot the deformation rate G = MVGT/sqrt(1 + FTP²).
    
    Args:
        tracks: DataFrame with particle tracks
        image_size: Size of the image (width, height)
        output_path: Path to save the plot
        background: Optional background image
        bin_size: Size of bins for averaging
        length_per_pixel: Physical length per pixel in mm
    """
    plt.figure(figsize=(10, 10 * image_size[1] / image_size[0]))
    
    if background is not None:
        plt.imshow(background, cmap='gray', extent=[0, image_size[0]*length_per_pixel, image_size[1]*length_per_pixel, 0])
    
    # Bin the data
    binned_data = bin_data(tracks, bin_size, length_per_pixel)
    
    # Get the maximum deformation rate for scaling
    max_deformation_rate = binned_data['deformation_rate'].max()
    
    # Plot deformation rate with fixed color range using square markers
    scatter = plt.scatter(binned_data['x']*length_per_pixel, binned_data['y']*length_per_pixel, 
                         c=binned_data['deformation_rate'], 
                         marker='s',  # Use square markers
                         s=bin_size*4,  # Marker size proportional to bin size
                         cmap='jet', 
                         alpha=0.7, 
                         vmin=0, 
                         vmax=100) #max_deformation_rate
    plt.colorbar(scatter, label='Deformation Rate (s⁻¹)')
    
    plt.xlim(0, image_size[0]*length_per_pixel)
    plt.ylim(image_size[1]*length_per_pixel, 0)  # Invert y-axis to match image coordinates
    plt.axis('equal')  # Ensure equal aspect ratio
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_results(
    analyzed_tracks: pd.DataFrame,
    image_size: Tuple[int, int],
    output_dir: str,
    background: Optional[np.ndarray] = None,
    max_intensity: Optional[np.ndarray] = None,
    bin_size: int = 8,
    length_per_pixel: float = 1.0
) -> None:
    """
    Visualize the results of particle tracking and analysis.
    
    Args:
        analyzed_tracks: DataFrame with analyzed tracks
        image_size: Size of the image (width, height)
        output_dir: Directory to save results
        background: Optional background image
        max_intensity: Optional maximum intensity image
        bin_size: Size of bins for averaging
        length_per_pixel: Physical length per pixel in mm
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save tracks to CSV
    print("Saving tracks to CSV...")
    analyzed_tracks.to_csv(os.path.join(output_dir, 'tracks.csv'), index=False)
    
    # Save background image if provided
    if background is not None:
        plt.imsave(os.path.join(output_dir, 'background.png'), background, cmap='gray')
    
    # Save maximum intensity image if provided
    if max_intensity is not None:
        plt.imsave(os.path.join(output_dir, 'max_intensity.png'), max_intensity, cmap='gray')
    
    # Plot tracks
    plot_tracks(analyzed_tracks, image_size, os.path.join(output_dir, 'tracks.png'), max_intensity, length_per_pixel)
    
    # Plot velocity magnitude
    plot_velocity_magnitude(analyzed_tracks, image_size, os.path.join(output_dir, 'velocity_magnitude.png'), max_intensity, bin_size, length_per_pixel)
    
    # Plot flow type parameter
    plot_flow_type_parameter(analyzed_tracks, image_size, os.path.join(output_dir, 'flow_type_parameter.png'), max_intensity, bin_size, length_per_pixel)
    
    # Plot magnitude velocity gradient tensor
    plot_magnitude_velocity_gradient(analyzed_tracks, image_size, os.path.join(output_dir, 'magnitude_velocity_gradient.png'), max_intensity, bin_size, length_per_pixel)
    
    # Plot deformation rate
    plot_deformation_rate(analyzed_tracks, image_size, os.path.join(output_dir, 'deformation_rate.png'), max_intensity, bin_size, length_per_pixel)
    
    # Print FTP and MVGT statistics to terminal
    print("\nFlow Type Parameter (FTP) Statistics:")
    print(f"Average FTP: {analyzed_tracks['ftp'].mean():.3f} ± {analyzed_tracks['ftp'].std():.3f}")
    print(f"Average MVGT: {analyzed_tracks['mvgt'].mean():.3f} ± {analyzed_tracks['mvgt'].std():.3f} s⁻¹")
    print(f"Average Deformation Rate: {analyzed_tracks['deformation_rate'].mean():.3f} ± {analyzed_tracks['deformation_rate'].std():.3f} s⁻¹")
    print(f"\nResults saved to {output_dir}")