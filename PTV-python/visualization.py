import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from scipy.ndimage import zoom


def get_scientific_colormap(field: str) -> str:
    """
    Get appropriate colormap for scientific visualization.
    
    Args:
        field: Field name to determine colormap
        
    Returns:
        Colormap name
    """
    colormaps = {
        'ftp': 'jet',            # Jet colormap for FTP
        'speed': 'viridis',      # Sequential for speed
        'deformation_rate': 'viridis',  # Sequential for rates
        'mvgt': 'viridis'        # Sequential for magnitude
    }
    return colormaps.get(field, 'viridis')


def create_aligned_colorbar(fig, ax, im, field: str) -> None:
    """
    Create properly aligned colorbar with same size as grid.
    
    Args:
        fig: Matplotlib figure
        ax: Matplotlib axes
        im: Image object
        field: Field name for labeling
    """
    # Use shrink=1 for normal sized colorbars
    cbar = fig.colorbar(im, ax=ax, shrink=1.0)
    cbar.set_label(field, rotation=270, labelpad=20)


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
    
    flipped_background = np.fliplr(background)
    if background is not None:
        plt.imshow(flipped_background, cmap='gray', extent=[0, image_size[0]*length_per_pixel, 0, image_size[1]*length_per_pixel])
    
    # Plot tracks with horizontal flip
    for particle_id, group in tracks.groupby('particle'):
        plt.plot((image_size[0] - group['x'])*length_per_pixel, (image_size[1] - group['y'])*length_per_pixel, '-', linewidth=1, alpha=0.5)
    
    plt.xlim(0, image_size[0]*length_per_pixel)
    plt.ylim(0, image_size[1]*length_per_pixel)  # Keep top-left origin for display
    plt.axis('equal')  # Ensure equal aspect ratio
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_binned_heatmap(binned_data: pd.DataFrame, output_path: str, field: str, 
                       vmin: Optional[float] = None, vmax: Optional[float] = None, 
                       n_bins: int = 48, title: Optional[str] = None) -> None:
    """
    Plot a heatmap from binned data with axes 0 to n_bins-1 and square bins.
    
    Args:
        binned_data: DataFrame with binned data containing x_bin, y_bin, and field columns
        output_path: Path to save the plot
        field: Field name to plot (e.g., 'speed', 'ftp', 'mvgt', 'deformation_rate')
        vmin: Minimum value for color scaling
        vmax: Maximum value for color scaling
        n_bins: Number of bins in each dimension
        title: Optional title for the plot
    """
    grid = np.full((n_bins, n_bins), np.nan)
    
    for _, row in binned_data.iterrows():
        xi = int(row['x_bin'])
        yi = int(row['y_bin'])
        if 0 <= xi < n_bins and 0 <= yi < n_bins:
            grid[yi, xi] = row[field]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use scientific colormap
    cmap_name = get_scientific_colormap(field)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad('black')

    # Use origin='lower' for bottom-left origin coordinate system
    im = ax.imshow(grid, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, 
                   extent=[0, n_bins, 0, n_bins], aspect='equal')

    # Create properly aligned colorbar with same size as grid
    create_aligned_colorbar(fig, ax, im, field)
    plt.xlabel('x_bin')
    plt.ylabel('y_bin')
    plt.xlim(0, n_bins-1)
    plt.ylim(0, n_bins-1)
    
    if title:
        plt.title(title)
    else:
        plt.title(f'{field} (binned)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_velocity_magnitude_heatmap(binned_data: pd.DataFrame, output_path: str, n_bins: int = 48) -> None:
    """
    Plot velocity magnitude heatmap from binned data.
    
    Args:
        binned_data: DataFrame with binned velocity data
        output_path: Path to save the plot
        n_bins: Number of bins in each dimension
    """
    max_speed = binned_data['speed'].quantile(0.98)
    plot_binned_heatmap(binned_data, output_path, 'speed', vmin=0, vmax=max_speed, 
                       n_bins=n_bins, title='Velocity Magnitude (mm/s)')

def plot_flow_type_parameter_heatmap(binned_data: pd.DataFrame, output_path: str, n_bins: int = 48) -> None:
    """
    Plot flow type parameter heatmap from binned data.
    
    Args:
        binned_data: DataFrame with binned velocity data
        output_path: Path to save the plot
        n_bins: Number of bins in each dimension
    """
    plot_binned_heatmap(binned_data, output_path, 'ftp', vmin=-1, vmax=1, 
                       n_bins=n_bins, title='Flow Type Parameter')

def plot_magnitude_velocity_gradient_heatmap(binned_data: pd.DataFrame, output_path: str, n_bins: int = 48) -> None:
    """
    Plot magnitude velocity gradient tensor heatmap from binned data.
    
    Args:
        binned_data: DataFrame with binned velocity data
        output_path: Path to save the plot
        n_bins: Number of bins in each dimension
    """
    max_mvgt = binned_data['mvgt'].quantile(0.98)
    plot_binned_heatmap(binned_data, output_path, 'mvgt', vmin=0, vmax=max_mvgt, 
                       n_bins=n_bins, title='Magnitude of Velocity Gradient Tensor (s⁻¹)')

def plot_deformation_rate_heatmap(binned_data: pd.DataFrame, output_path: str, n_bins: int = 48) -> None:
    """
    Plot deformation rate heatmap from binned data.
    
    Args:
        binned_data: DataFrame with binned velocity data
        output_path: Path to save the plot
        n_bins: Number of bins in each dimension
    """
    max_deformation = binned_data['deformation_rate'].quantile(0.98)
    plot_binned_heatmap(binned_data, output_path, 'deformation_rate', vmin=0, vmax=max_deformation, 
                       n_bins=n_bins, title='Deformation Rate (s⁻¹)')


def plot_saved_streamlines(streamline_coords: pd.DataFrame, output_path: str = None, background: np.ndarray = None, 
                          image_size: Tuple[int, int] = None, length_per_pixel: float = 1.0,
                          is_custom_background: bool = False) -> None:
    """
    Plot saved streamlines with optional background image.
    Handles both standard (grayscale) and custom (color/grayscale) backgrounds.
    
    Args:
        streamline_coords: DataFrame with saved streamline coordinates
        output_path: Path to save the plot (if None, display only)
        background: Optional background image (typically max_intensity or custom image)
        image_size: Size of the image (width, height) for proper scaling
        length_per_pixel: Physical length per pixel in mm
        is_custom_background: If True, handles color images and different normalization
    """
    if streamline_coords.empty:
        print("No streamline data to plot")
        return
    
    # Use same figure size as tracks plot
    if image_size:
        plt.figure(figsize=(10, 10 * image_size[1] / image_size[0]))
    else:
        plt.figure(figsize=(10, 10))
    
    # Determine actual grid size from the data
    max_x_bin = int(streamline_coords['x_bin'].max()) + 1
    max_y_bin = int(streamline_coords['y_bin'].max()) + 1

    # Create proper bin grid for plotting (matching actual data size)
    x_bins = np.arange(0, max_x_bin)
    y_bins = np.arange(0, max_y_bin)
    X_bin_grid, Y_bin_grid = np.meshgrid(x_bins, y_bins)

    # Plot background if provided (resized to bin grid)
    if background is not None:
        if is_custom_background:
            # Handle custom background (color images, different normalization)
            flipped_background = np.flipud(background)
            # Ensure proper data type and value range for display
            if flipped_background.dtype == np.float64 or flipped_background.dtype == np.float32:
                # If image is in [0,1] range, keep it as is
                if flipped_background.max() > 1.0:
                    # If image is in [0,255] range but stored as float, normalize
                    flipped_background = flipped_background / 255.0
            elif flipped_background.dtype in [np.uint8, np.uint16]:
                # Convert integer images to [0,1] range
                flipped_background = flipped_background.astype(np.float64)
                if flipped_background.max() > 1.0:
                    flipped_background = flipped_background / 255.0
            
            # Resize background to match bin grid shape
            y_scale = max_y_bin / flipped_background.shape[0]
            x_scale = max_x_bin / flipped_background.shape[1]
            
            # Handle both color and grayscale images
            if len(flipped_background.shape) == 3:
                # For color images, zoom each channel separately
                background_resized = np.zeros((max_y_bin, max_x_bin, flipped_background.shape[2]))
                for i in range(flipped_background.shape[2]):
                    background_resized[:, :, i] = zoom(flipped_background[:, :, i], (y_scale, x_scale), order=0)
            else:
                # For grayscale images
                background_resized = zoom(flipped_background, (y_scale, x_scale), order=0)
            
            plt.imshow(background_resized, extent=[0, max_x_bin, 0, max_y_bin], origin='lower', alpha=0.8)
        else:
            # Handle standard background (grayscale, typically max_intensity)
            flipped_background = np.fliplr(np.flipud(background))
            # Resize background to match bin grid shape
            y_scale = max_y_bin / flipped_background.shape[0]
            x_scale = max_x_bin / flipped_background.shape[1]
            background_resized = zoom(flipped_background, (y_scale, x_scale), order=0)
            plt.imshow(background_resized, cmap='gray', extent=[0, max_x_bin, 0, max_y_bin], origin='lower', alpha=0.8)
    
    # Group streamlines by streamline_id and plot each one
    unique_streamlines = streamline_coords['streamline_id'].unique()
    streamline_color = 'black'

    for i, streamline_id in enumerate(unique_streamlines):
        streamline_data = streamline_coords[streamline_coords['streamline_id'] == streamline_id]
        # Sort by point_index to ensure correct order
        streamline_data = streamline_data.sort_values('point_index')
        # Use saved pixel coordinates directly
        x_pixel_coords = streamline_data['x_pixel']
        y_pixel_coords = streamline_data['y_pixel']
        plt.plot(x_pixel_coords, y_pixel_coords, color=streamline_color, linewidth=1, alpha=0.8)
    
    plt.title(f'Saved Streamlines ({len(unique_streamlines)} streamlines)')
    plt.xlabel('X (bin index)')
    plt.ylabel('Y (bin index)')

    # Set axis limits to match bin grid
    plt.xlim(0, max_x_bin)
    plt.ylim(0, max_y_bin)
    plt.axis('equal')

    # Set ticks to show every bin boundary
    plt.xticks(np.arange(0, max_x_bin + 1, 1))
    plt.yticks(np.arange(0, max_y_bin + 1, 1))

    # Add grid to show every bin boundary
    plt.grid(True, alpha=0.3, color='white', linewidth=0.5)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_saved_streamlines_with_custom_background(streamline_coords: pd.DataFrame, output_path: str = None, 
                                                  background: np.ndarray = None, image_size: Tuple[int, int] = None, 
                                                  length_per_pixel: float = 1.0) -> None:
    """
    Plot saved streamlines with a custom background image.
    This is a convenience wrapper around plot_saved_streamlines with is_custom_background=True.
    
    Args:
        streamline_coords: DataFrame with saved streamline coordinates
        output_path: Path to save the plot (if None, display only)
        background: Custom background image to overlay streamlines on
        image_size: Size of the image (width, height) for proper scaling
        length_per_pixel: Physical length per pixel in mm
    """
    plot_saved_streamlines(streamline_coords, output_path, background, image_size, length_per_pixel, 
                          is_custom_background=True)


def plot_ftp_with_streamlines(binned_data: pd.DataFrame, streamline_coords: pd.DataFrame, output_path: str, n_bins: int = 48) -> None:
    """
    Plot flow type parameter heatmap with streamlines overlay.
    
    Args:
        binned_data: DataFrame with binned velocity data
        streamline_coords: DataFrame with streamline coordinates to overlay
        output_path: Path to save the plot
        n_bins: Number of bins in each dimension
    """
    if streamline_coords.empty:
        print("No streamline data to overlay")
        return
    
    # Create the FTP heatmap grid
    grid = np.full((n_bins, n_bins), np.nan)
    
    for _, row in binned_data.iterrows():
        xi = int(row['x_bin'])
        yi = int(row['y_bin'])
        if 0 <= xi < n_bins and 0 <= yi < n_bins:
            grid[yi, xi] = row['ftp']
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use scientific colormap for FTP
    cmap_name = get_scientific_colormap('ftp')
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad('black')

    # Use origin='lower' for bottom-left origin coordinate system
    im = ax.imshow(grid, origin='lower', cmap=cmap, vmin=-1, vmax=1, 
                   extent=[0, n_bins, 0, n_bins], aspect='equal')

    # Create properly aligned colorbar with same size as grid
    create_aligned_colorbar(fig, ax, im, 'ftp')
    
    # Overlay streamlines
    unique_streamlines = streamline_coords['streamline_id'].unique()
    streamline_color = 'black'
    
    for streamline_id in unique_streamlines:
        streamline_data = streamline_coords[streamline_coords['streamline_id'] == streamline_id]
        # Sort by point_index to ensure correct order
        streamline_data = streamline_data.sort_values('point_index')
        # Use saved pixel coordinates directly
        x_pixel_coords = streamline_data['x_pixel']
        y_pixel_coords = streamline_data['y_pixel']
        ax.plot(x_pixel_coords, y_pixel_coords, color=streamline_color, linewidth=1, alpha=0.8)
    
    ax.set_title('Flow Type Parameter with Streamlines')
    ax.set_xlabel('X (bin index)')
    ax.set_ylabel('Y (bin index)')
    
    # Set axis limits to match bin grid
    ax.set_xlim(0, n_bins)
    ax.set_ylim(0, n_bins)
    
    # Set ticks to show every bin boundary with smaller font size
    ax.set_xticks(np.arange(0, n_bins + 1, 1))
    ax.set_yticks(np.arange(0, n_bins + 1, 1))
    
    # Add grid to show every bin boundary
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_deformation_rate_with_streamlines(binned_data: pd.DataFrame, streamline_coords: pd.DataFrame, output_path: str, n_bins: int = 48) -> None:
    """
    Plot deformation rate heatmap with streamlines overlay.
    
    Args:
        binned_data: DataFrame with binned velocity data
        streamline_coords: DataFrame with streamline coordinates to overlay
        output_path: Path to save the plot
        n_bins: Number of bins in each dimension
    """
    if streamline_coords.empty:
        print("No streamline data to overlay")
        return
    
    # Create the deformation rate heatmap grid
    grid = np.full((n_bins, n_bins), np.nan)
    
    for _, row in binned_data.iterrows():
        xi = int(row['x_bin'])
        yi = int(row['y_bin'])
        if 0 <= xi < n_bins and 0 <= yi < n_bins:
            grid[yi, xi] = row['deformation_rate']
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use scientific colormap for deformation rate
    cmap_name = get_scientific_colormap('deformation_rate')
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad('black')

    # Calculate max deformation rate for color scaling (use 98th percentile to avoid outliers)
    max_deformation = binned_data['deformation_rate'].quantile(0.98)

    # Use origin='lower' for bottom-left origin coordinate system
    im = ax.imshow(grid, origin='lower', cmap=cmap, vmin=0, vmax=max_deformation, 
                   extent=[0, n_bins, 0, n_bins], aspect='equal')

    # Create properly aligned colorbar with same size as grid
    create_aligned_colorbar(fig, ax, im, 'Deformation Rate (s⁻¹)')
    
    # Overlay streamlines
    unique_streamlines = streamline_coords['streamline_id'].unique()
    streamline_color = 'black'
    
    for streamline_id in unique_streamlines:
        streamline_data = streamline_coords[streamline_coords['streamline_id'] == streamline_id]
        # Sort by point_index to ensure correct order
        streamline_data = streamline_data.sort_values('point_index')
        # Use saved pixel coordinates directly
        x_pixel_coords = streamline_data['x_pixel']
        y_pixel_coords = streamline_data['y_pixel']
        ax.plot(x_pixel_coords, y_pixel_coords, color=streamline_color, linewidth=1, alpha=0.8)
    
    ax.set_title('Deformation Rate with Streamlines', fontsize=12)
    ax.set_xlabel('X (bin index)')
    ax.set_ylabel('Y (bin index)')
    
    # Set axis limits to match bin grid
    ax.set_xlim(0, n_bins)
    ax.set_ylim(0, n_bins)
    
    # Set ticks to show every bin boundary with smaller font size
    ax.set_xticks(np.arange(0, n_bins + 1, 1))
    ax.set_yticks(np.arange(0, n_bins + 1, 1))
    
    # Add grid to show every bin boundary
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def save_streamline_csv(streamline_coordinates: pd.DataFrame, output_dir: str) -> None:
    """
    Save streamline bin coordinates to CSV file.
    
    Args:
        streamline_coordinates: DataFrame with streamline bin coordinate data
        output_dir: Directory to save the CSV file
    """
    
    output_path = os.path.join(output_dir, 'streamline_bin_coordinates.csv')
    streamline_coordinates.to_csv(output_path, index=False)


def create_bin_time_map(streamline_coordinates: pd.DataFrame) -> pd.DataFrame:
    """
    Create a bin-time map that aggregates time along streamlines for each bin.
    When multiple streamlines pass through the same bin, uses the mean time.
    
    Args:
        streamline_coordinates: DataFrame with streamline coordinates including time_along_streamline_s
        
    Returns:
        DataFrame with columns: x_bin, y_bin, time_along_streamline_s
    """
    if len(streamline_coordinates) == 0 or 'time_along_streamline_s' not in streamline_coordinates.columns:
        print("Warning: No time data available for bin-time map")
        return pd.DataFrame(columns=['x_bin', 'y_bin', 'time_along_streamline_s'])
    
    # Group by bin coordinates and take mean time (handles multiple streamlines passing through same bin)
    bin_time_map = streamline_coordinates.groupby(['x_bin', 'y_bin'])['time_along_streamline_s'].mean().reset_index()
    
    return bin_time_map


def plot_bin_time_map_heatmap(bin_time_map: pd.DataFrame, output_path: str, 
                               streamline_coords: Optional[pd.DataFrame] = None) -> None:
    """
    Plot a heatmap of time along streamlines for each bin using Jet colormap.
    Optionally overlay streamlines on the heatmap.
    
    Args:
        bin_time_map: DataFrame with columns x_bin, y_bin, time_along_streamline_s
        output_path: Path to save the plot
        streamline_coords: Optional DataFrame with streamline coordinates to overlay
    """
    if len(bin_time_map) == 0:
        print("No data to plot in bin-time map")
        return
    
    # Determine the grid size based on max bins
    x_max = int(bin_time_map['x_bin'].max())
    y_max = int(bin_time_map['y_bin'].max())
    
    # Create a full grid filled with NaNs
    grid_shape = (y_max + 1, x_max + 1)
    time_grid = np.full(grid_shape, np.nan)
    
    # Fill the grid with the time values
    for _, row in bin_time_map.iterrows():
        x = int(row['x_bin'])
        y = int(row['y_bin'])
        time_grid[y, x] = row['time_along_streamline_s']
    
    # Plotting with Jet colormap
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create the heatmap with black for NaN values
    # origin='upper' places (0,0) at the top-left, consistent with image coords
    cmap = plt.get_cmap('jet').copy()
    cmap.set_bad('black')  # Set NaN values to black
    im = ax.imshow(time_grid, cmap=cmap, interpolation='nearest', origin='upper')
    
    # Add Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Time along Streamline (s)', rotation=270, labelpad=15)
    
    # Overlay streamlines if provided
    if streamline_coords is not None and len(streamline_coords) > 0:
        unique_streamlines = streamline_coords['streamline_id'].unique()
        streamline_color = 'white'  # White streamlines for visibility on colored background
        
        for streamline_id in unique_streamlines:
            streamline_data = streamline_coords[streamline_coords['streamline_id'] == streamline_id]
            # Sort by point_index to ensure correct order
            streamline_data = streamline_data.sort_values('point_index')
            # Use saved pixel coordinates directly
            x_pixel_coords = streamline_data['x_pixel']
            y_pixel_coords = streamline_data['y_pixel']
            ax.plot(x_pixel_coords, y_pixel_coords, color=streamline_color, linewidth=1, alpha=0.8)
    
    # Formatting
    title = 'Coordinate Map: Time along Streamline'
    if streamline_coords is not None:
        title += ' with Streamlines'
    ax.set_title(title)
    ax.set_xlabel('x_bin')
    ax.set_ylabel('y_bin')
    
    # Set axis limits
    ax.set_xlim(-0.5, x_max + 0.5)
    ax.set_ylim(-0.5, y_max + 0.5)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_results(
    analyzed_tracks: pd.DataFrame,
    binned_data: pd.DataFrame,
    image_size: Tuple[int, int],
    output_dir: str,
    background: Optional[np.ndarray] = None,
    max_intensity: Optional[np.ndarray] = None,
    length_per_pixel: float = 1.0,
    n_bins: int = 48,
    streamline_results: Optional[Dict] = None,
    custom_streamline_background: Optional[np.ndarray] = None) -> None:
    """
    Visualize the results of particle tracking and analysis using binned data.
    
    Args:
        analyzed_tracks: DataFrame with analyzed tracks
        binned_data: DataFrame with binned velocity data
        image_size: Size of the image (width, height)
        output_dir: Directory to save results
        background: Optional background image
        max_intensity: Optional maximum intensity image
        length_per_pixel: Physical length per pixel in mm
        n_bins: Number of bins in each dimension for heatmaps
        streamline_results: Dictionary containing streamline analysis results (optional)
        custom_streamline_background: Optional custom background image for additional streamline plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    binned_data.to_csv(os.path.join(output_dir, 'binned_velocity_data.csv'), index=False)
    
    if background is not None:
        plt.imsave(os.path.join(output_dir, 'background.png'), background, cmap='gray')
    
    if max_intensity is not None:
        plt.imsave(os.path.join(output_dir, 'max_intensity.png'), max_intensity, cmap='gray')
    
    plot_tracks(analyzed_tracks, image_size, os.path.join(output_dir, 'tracks.png'), max_intensity, length_per_pixel)
    
    plot_velocity_magnitude_heatmap(binned_data, os.path.join(output_dir, 'velocity_magnitude.png'), n_bins)
    plot_flow_type_parameter_heatmap(binned_data, os.path.join(output_dir, 'flow_type_parameter.png'), n_bins)
    plot_deformation_rate_heatmap(binned_data, os.path.join(output_dir, 'deformation_rate.png'), n_bins)
    
    # Generate streamline analysis if results are provided
    if streamline_results is not None:
        # Create streamline plot with max intensity background
        plot_saved_streamlines(
            streamline_coords=streamline_results['streamline_coordinates'],
            output_path=os.path.join(output_dir, 'streamlines.png'),
            background=max_intensity,
            image_size=image_size,
            length_per_pixel=length_per_pixel
        )
        
        # Create additional streamline plot with custom background if provided
        if custom_streamline_background is not None:
            plot_saved_streamlines_with_custom_background(
                streamline_coords=streamline_results['streamline_coordinates'],
                output_path=os.path.join(output_dir, 'streamlines_custom_background.png'),
                background=custom_streamline_background,
                image_size=image_size,
                length_per_pixel=length_per_pixel
            )
        
        
        # Create FTP plot with streamlines overlay
        plot_ftp_with_streamlines(
            binned_data=binned_data,
            streamline_coords=streamline_results['streamline_coordinates'],
            output_path=os.path.join(output_dir, 'flow_type_parameter_with_streamlines.png'),
            n_bins=n_bins
        )
        
        # Create deformation rate plot with streamlines overlay
        plot_deformation_rate_with_streamlines(
            binned_data=binned_data,
            streamline_coords=streamline_results['streamline_coordinates'],
            output_path=os.path.join(output_dir, 'deformation_rate_with_streamlines.png'),
            n_bins=n_bins
        )
        
        # Always save CSV data
        save_streamline_csv(streamline_results['streamline_coordinates'], output_dir)
        
        # Create and plot bin-time map heatmap
        if 'time_along_streamline_s' in streamline_results['streamline_coordinates'].columns:
            bin_time_map = create_bin_time_map(streamline_results['streamline_coordinates'])
            
            # Plot bin-time map heatmap without streamlines
            if len(bin_time_map) > 0:
                plot_bin_time_map_heatmap(
                    bin_time_map,
                    os.path.join(output_dir, 'time_map.png')
                )
                
                # Plot bin-time map heatmap with streamlines overlay
                plot_bin_time_map_heatmap(
                    bin_time_map,
                    os.path.join(output_dir, 'time_map_with_streamlines.png'),
                    streamline_coords=streamline_results['streamline_coordinates']
                )