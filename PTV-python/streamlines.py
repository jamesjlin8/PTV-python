import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
from config import AnalysisConfig


def build_uniform_grids_from_binned_data(binned_data: pd.DataFrame, image_size: Tuple[int, int], 
                                       bin_size: int, n_bins: int = 48) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build uniform grids from binned velocity data, including additional property grids.
    
    Args:
        binned_data: DataFrame with binned velocity data
        image_size: Tuple of (width, height) in pixels
        bin_size: Size of bins used in analysis
        n_bins: Number of bins in each dimension for the output grid
        
    Returns:
        Tuple of (X_grid, Y_grid, U_grid, V_grid, Speed_grid, FTP_grid, DefRate_grid)
        where each grid is shape (n_bins, n_bins)
    """
    num_bins = n_bins
    
    # Create bin center coordinates (0 to 47)
    x_bins = np.arange(num_bins)
    y_bins = np.arange(num_bins)
    
    X_grid, Y_grid = np.meshgrid(x_bins, y_bins)
    
    U_grid = np.full((num_bins, num_bins), np.nan)
    V_grid = np.full((num_bins, num_bins), np.nan)
    Speed_grid = np.full((num_bins, num_bins), np.nan)
    FTP_grid = np.full((num_bins, num_bins), np.nan)
    DefRate_grid = np.full((num_bins, num_bins), np.nan)
    
    # Vectorized mapping of binned data to the 48x48 grid
    try:
        # Convert bin indices to integers
        x_bins = binned_data['x_bin'].values.astype(int)
        y_bins = binned_data['y_bin'].values.astype(int)
        
        # Create valid mask for bins within bounds
        valid_mask = (x_bins >= 0) & (x_bins < num_bins) & (y_bins >= 0) & (y_bins < num_bins)
        
        # Apply valid mask
        x_bins_valid = x_bins[valid_mask]
        y_bins_valid = y_bins[valid_mask]
        vx_valid = binned_data['vx'].values[valid_mask]
        vy_valid = binned_data['vy'].values[valid_mask]
        speed_valid = binned_data['speed'].values[valid_mask] if 'speed' in binned_data.columns else np.full_like(vx_valid, np.nan)
        ftp_valid = binned_data['ftp'].values[valid_mask] if 'ftp' in binned_data.columns else np.full_like(vx_valid, np.nan)
        defrate_valid = binned_data['deformation_rate'].values[valid_mask] if 'deformation_rate' in binned_data.columns else np.full_like(vx_valid, np.nan)

        # Create masks for non-NaN values
        vx_mask = ~np.isnan(vx_valid)
        vy_mask = ~np.isnan(vy_valid)
        speed_mask = ~np.isnan(speed_valid)
        ftp_mask = ~np.isnan(ftp_valid)
        defrate_mask = ~np.isnan(defrate_valid)

        # Assign values to grids
        U_grid[y_bins_valid[vx_mask], x_bins_valid[vx_mask]] = vx_valid[vx_mask]
        V_grid[y_bins_valid[vy_mask], x_bins_valid[vy_mask]] = vy_valid[vy_mask]
        Speed_grid[y_bins_valid[speed_mask], x_bins_valid[speed_mask]] = speed_valid[speed_mask]
        FTP_grid[y_bins_valid[ftp_mask], x_bins_valid[ftp_mask]] = ftp_valid[ftp_mask]
        DefRate_grid[y_bins_valid[defrate_mask], x_bins_valid[defrate_mask]] = defrate_valid[defrate_mask]

    except Exception as e:
        # Fallback to row-by-row method if vectorized approach fails
        for _, row in binned_data.iterrows():
            try:
                x_bin = int(row['x_bin'])
                y_bin = int(row['y_bin'])
                if 0 <= x_bin < num_bins and 0 <= y_bin < num_bins:
                    if not np.isnan(row['vx']):
                        U_grid[y_bin, x_bin] = row['vx']
                    if not np.isnan(row['vy']):
                        V_grid[y_bin, x_bin] = row['vy']
                    if 'speed' in row and not np.isnan(row['speed']):
                        Speed_grid[y_bin, x_bin] = row['speed']
                    if 'ftp' in row and not np.isnan(row['ftp']):
                        FTP_grid[y_bin, x_bin] = row['ftp']
                    if 'deformation_rate' in row and not np.isnan(row['deformation_rate']):
                        DefRate_grid[y_bin, x_bin] = row['deformation_rate']
            except (ValueError, TypeError) as row_error:
                # Skip invalid rows
                continue

    return X_grid, Y_grid, U_grid, V_grid, Speed_grid, FTP_grid, DefRate_grid


def extract_streamline_coordinates_from_streamplot(U_grid: np.ndarray, V_grid: np.ndarray, X_grid: np.ndarray, Y_grid: np.ndarray, 
                                                 binned_data: pd.DataFrame, min_bins: int = 8, density: int = 2,
                                                 Speed_grid: np.ndarray = None,
                                                 FTP_grid: np.ndarray = None,
                                                 DefRate_grid: np.ndarray = None) -> pd.DataFrame:
    """
    Extract streamline coordinates from streamplot (the accurate method).
    Filters short streamlines, removes duplicates, and adds flow properties.
    
    Args:
        U_grid: U velocity component grid
        V_grid: V velocity component grid
        X_grid: X coordinate grid (bin coordinates)
        Y_grid: Y coordinate grid (bin coordinates)
        binned_data: DataFrame with binned velocity data including flow properties
        min_bins: Minimum number of bins a streamline must pass through
        density: Density of streamlines (same as streamplot)
        
    Returns:
        DataFrame with streamline bin coordinates and flow properties
    """
    # Generate streamlines using the same method as the plot
    fig, ax = plt.subplots()
    streamlines = ax.streamplot(X_grid, Y_grid, U_grid, V_grid, density=density, 
                               linewidth=1.0, color='red', arrowsize=1.0, broken_streamlines=False)
    plt.close(fig)
    
    # Extract streamline paths from LineCollection
    raw_streamlines = []
    
    # streamlines.lines is a LineCollection, we need to iterate through its segments
    for line_segment in streamlines.lines.get_segments():
        if len(line_segment) > 1:  # Only include paths with multiple points
            # Convert to bin coordinates and remove duplicates
            streamline_points = []
            seen_coords = set()

            for k, (x, y) in enumerate(line_segment):
                # Find the nearest bin indices using rounding and clamping
                x_bin_idx = int(round(x))
                y_bin_idx = int(round(y))
                # Clamp to grid boundaries
                x_bin_idx = max(0, min(X_grid.shape[1] - 1, x_bin_idx))
                y_bin_idx = max(0, min(Y_grid.shape[0] - 1, y_bin_idx))

                # Check for duplicate coordinates
                coord_key = (x_bin_idx, y_bin_idx)
                if coord_key not in seen_coords:
                    seen_coords.add(coord_key)
                    streamline_points.append({
                        'x_bin': x_bin_idx,
                        'y_bin': y_bin_idx,
                        'x_pixel': x,
                        'y_pixel': y,
                        'point_index': len(streamline_points)
                    })

            # Add streamlines that meet minimum length requirement
            if len(streamline_points) >= min_bins:
                raw_streamlines.append(streamline_points)
    
    # Add flow properties and create final DataFrame
    streamline_data = []
    streamline_id = 0
    
    for streamline in raw_streamlines:
        # Ensure point indices are properly ordered and continuous
        streamline_sorted = sorted(streamline, key=lambda p: p['point_index'])

        for i, point in enumerate(streamline_sorted):
            # Get flow properties from precomputed grids
            flow_props = _get_flow_properties_at_bin(
                point['x_bin'], point['y_bin'],
                U_grid, V_grid, Speed_grid, FTP_grid, DefRate_grid
            )

            streamline_data.append({
                'streamline_id': f"streamline_{streamline_id}",
                'streamline_index': streamline_id,
                'x_bin': point['x_bin'],
                'y_bin': point['y_bin'],
                'x_pixel': point['x_pixel'],
                'y_pixel': point['y_pixel'],
                'point_index': i,  # Ensure continuous indexing starting from 0
                'speed': flow_props['speed'],
                'ftp': flow_props['ftp'],
                'deformation_rate': flow_props['deformation_rate']
            })
        streamline_id += 1
    
    print(f"Extracted {streamline_id} streamlines from streamplot (min {min_bins} bins)")
    
    return pd.DataFrame(streamline_data)


def _get_flow_properties_at_bin(x_bin: int, y_bin: int,
                                U_grid: np.ndarray, V_grid: np.ndarray,
                                Speed_grid: np.ndarray, FTP_grid: np.ndarray, DefRate_grid: np.ndarray) -> dict:
    """
    Get flow properties (u, v, speed, ftp, deformation_rate) at a specific bin location from precomputed grids.
    Args:
        x_bin: X bin coordinate
        y_bin: Y bin coordinate
        U_grid, V_grid, Speed_grid, FTP_grid, DefRate_grid: property grids
    Returns:
        Dictionary with flow properties
    """
    num_y, num_x = U_grid.shape
    # Clamp indices to grid boundaries
    x_bin = max(0, min(num_x - 1, int(x_bin)))
    y_bin = max(0, min(num_y - 1, int(y_bin)))
    return {
        'u': U_grid[y_bin, x_bin],
        'v': V_grid[y_bin, x_bin],
        'speed': Speed_grid[y_bin, x_bin],
        'ftp': FTP_grid[y_bin, x_bin],
        'deformation_rate': DefRate_grid[y_bin, x_bin]
    }



def _calculate_time_along_streamlines(streamline_df: pd.DataFrame, length_per_pixel: float = 1.0) -> pd.DataFrame:
    """
    Calculate time along streamlines based on distance and speed.
    
    For each streamline, calculates:
    - Distance between consecutive points (in physical units)
    - Time increment: dt = distance / speed
    - Cumulative time along streamline
    
    Args:
        streamline_df: DataFrame with streamline coordinates and speed
        length_per_pixel: Physical length per pixel (mm/pixel)
        
    Returns:
        DataFrame with added 'distance_mm', 'dt_s', and 'time_along_streamline_s' columns
    """
    if len(streamline_df) == 0:
        return streamline_df
    
    # Make a copy to avoid modifying original
    df = streamline_df.copy()
    
    # Initialize time columns
    df['distance_mm'] = np.nan
    df['dt_s'] = np.nan
    df['time_along_streamline_s'] = np.nan
    
    # Process each streamline separately
    for streamline_idx in df['streamline_index'].unique():
        streamline_mask = df['streamline_index'] == streamline_idx
        streamline_points = df[streamline_mask].sort_values('point_index').copy()
        
        if len(streamline_points) < 2:
            # Single point streamline - no time calculation possible
            continue
        
        # Calculate distances between consecutive points
        # Convert pixel coordinates to physical units
        x_mm = streamline_points['x_pixel'].values * length_per_pixel
        y_mm = streamline_points['y_pixel'].values * length_per_pixel
        
        # Calculate distance increments
        dx = np.diff(x_mm)
        dy = np.diff(y_mm)
        distances_mm = np.sqrt(dx**2 + dy**2)
        
        # First point has no distance (starting point)
        distances_array = np.concatenate([[0.0], distances_mm])
        df.loc[streamline_mask, 'distance_mm'] = distances_array
        
        # Calculate time increments using speed
        # Use average speed between consecutive points for better accuracy
        speeds = streamline_points['speed'].values
        avg_speeds = (speeds[:-1] + speeds[1:]) / 2.0
        
        # Time increment: dt = distance / speed
        # Handle zero or NaN speeds
        dt_array = np.zeros_like(distances_array)
        dt_array[1:] = np.where(
            (avg_speeds > 0) & np.isfinite(avg_speeds),
            distances_mm / avg_speeds,
            0.0
        )
        
        df.loc[streamline_mask, 'dt_s'] = dt_array
        
        # Calculate cumulative time along streamline
        time_along = np.cumsum(dt_array)
        df.loc[streamline_mask, 'time_along_streamline_s'] = time_along
    
    return df


def calculate_streamlines_from_binned_data(binned_data: pd.DataFrame, image_size: Tuple[int, int],
                                         bin_size: int = None, density: int = 2, 
                                         min_bins: int = 8, n_bins: int = 48,
                                         length_per_pixel: float = 1.0) -> dict:
    """
    Calculate streamlines directly from binned velocity data using streamplot.
    This is the simplified, more accurate approach that avoids streamfunction calculation.
    
    Args:
        binned_data: DataFrame with binned velocity data
        image_size: Tuple of (width, height) in pixels
        bin_size: Size of bins used in analysis
        density: Density of streamlines (same as streamplot)
        min_bins: Minimum number of bins a streamline must pass through
        n_bins: Number of bins in each dimension for the visualization grid
        length_per_pixel: Physical length per pixel (mm/pixel) for time calculations
        
    Returns:
        Dictionary containing all computed fields and grids
    """
    # Infer bin size from data if not provided
    if bin_size is None:
        if 'x_bin' in binned_data.columns and 'y_bin' in binned_data.columns:
            bin_size = int(np.diff(binned_data['x'].unique()).min())
        else:
            bin_size = 1

    # Step 1: Build uniform grids from binned data
    X_grid, Y_grid, U_grid, V_grid, Speed_grid, FTP_grid, DefRate_grid = build_uniform_grids_from_binned_data(
        binned_data, image_size, bin_size, n_bins=n_bins
    )

    # Step 2: Extract streamline coordinates directly from streamplot
    streamline_coords = extract_streamline_coordinates_from_streamplot(
        U_grid, V_grid, X_grid, Y_grid, binned_data,
        min_bins=min_bins, density=density,
        Speed_grid=Speed_grid, FTP_grid=FTP_grid, DefRate_grid=DefRate_grid
    )
    
    # Step 3: Calculate time along streamlines
    if len(streamline_coords) > 0:
        streamline_coords = _calculate_time_along_streamlines(streamline_coords, length_per_pixel)

    # Compile results for return
    results = {
        'X_grid': X_grid,
        'Y_grid': Y_grid,
        'U_grid': U_grid,
        'V_grid': V_grid,
        'Speed_grid': Speed_grid,
        'FTP_grid': FTP_grid,
        'DefRate_grid': DefRate_grid,
        'method': 'streamplot_direct',
        'streamline_coordinates': streamline_coords,
        'bin_size': bin_size,
        'density': density
    }

    return results

