import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Tuple, Dict, List
from config import AnalysisConfig
from tqdm import tqdm

def calculate_velocities(tracks: pd.DataFrame, frame_rate: float, length_per_pixel: float) -> pd.DataFrame:
    """
    Calculate particle velocities from trajectories.
    
    Args:
        tracks: DataFrame with linked trajectories
        frame_rate: Camera frame rate in fps
        length_per_pixel: Physical length per pixel in mm
        
    Returns:
        DataFrame with added velocity columns (vx, vy, speed in mm/s)
    """
    print("Calculating velocities...")
    
    # Create a copy of the tracks DataFrame to avoid modifying the original
    tracks_with_velocity = tracks.copy()
    tracks_with_velocity = tracks_with_velocity.reset_index(drop=True)

    # Sort by particle and frame to ensure chronological order
    tracks_with_velocity = tracks_with_velocity.sort_values(['particle', 'frame'])
    
    # Group by particle
    particle_groups = list(tracks_with_velocity.groupby('particle'))
    for particle_id, group in tqdm(particle_groups, desc="Calculating velocities"):
        # Get frames for this particle
        frames = group['frame'].values

        # Calculate velocities for each frame
        for i in range(len(frames) - 1):
            current_frame = frames[i]
            next_frame = frames[i + 1]
            
            # Get positions at current and next frames
            current_pos = group[group['frame'] == current_frame].iloc[0]
            next_pos = group[group['frame'] == next_frame].iloc[0]
            
            # Calculate time difference in seconds
            dt = (next_frame - current_frame) / frame_rate
            
            # Convert position differences from pixels to physical units (mm)
            dx_physical = (next_pos['x'] - current_pos['x']) * length_per_pixel
            dy_physical = (next_pos['y'] - current_pos['y']) * length_per_pixel
            
            # Calculate velocity as (next_pos - current_pos) / dt
            # Now in mm/s
            vx = dx_physical / dt
            vy = dy_physical / dt
            
            # Assign velocity to the current frame
            tracks_with_velocity.loc[(tracks_with_velocity['particle'] == particle_id) & 
                                    (tracks_with_velocity['frame'] == current_frame), 'vx'] = vx
            tracks_with_velocity.loc[(tracks_with_velocity['particle'] == particle_id) & 
                                    (tracks_with_velocity['frame'] == current_frame), 'vy'] = vy
    
    # Calculate speed in mm/s
    tracks_with_velocity['speed'] = np.sqrt(tracks_with_velocity['vx']**2 + tracks_with_velocity['vy']**2)
    
    return tracks_with_velocity

def calculate_ftp_and_mvgt(tracks: pd.DataFrame, 
                          bin_size: int = 8, 
                          edge_cutoff: int = 1,
                          length_per_pixel: float = 1.0,
                          gradient_method: str = 'central_differences',
                          loess_radius_factor: float = 2.0) -> pd.DataFrame:
    """
    Calculate Flow Type Parameter (FTP) and Mean Velocity Gradient Tensor (MVGT) from particle trajectories.
    
    Args:
        tracks: DataFrame with particle trajectories and velocities
        bin_size: Size of bins for velocity gradient calculation in pixels
        edge_cutoff: Number of bins to cut off from the edges
        frame_rate: Camera frame rate in fps (not used in calculations)
        length_per_pixel: Physical length per pixel in mm
        gradient_method: Method to calculate velocity gradients ('central_differences' or 'loess')
        loess_radius_factor: Factor to multiply bin_size by to determine LOESS search radius
        
    Returns:
        DataFrame with added FTP and MVGT columns
    """
    print(f"Calculating FTP and MVGT using {gradient_method} method...")
    # Create a copy of the tracks DataFrame to avoid modifying the original
    tracks_with_parameters = tracks.copy()
    
    # Initialize columns for FTP and MVGT data
    tracks_with_parameters['ftp'] = np.nan
    tracks_with_parameters['mvgt'] = np.nan
    
    # Extract velocity data
    velocity_data = tracks[['x', 'y', 'vx', 'vy']].copy()
    
    # Bin the velocity data first (common for both methods)
    print("Binning velocity data...")
    x_bins = np.arange(velocity_data['x'].min(), velocity_data['x'].max() + bin_size, bin_size)
    y_bins = np.arange(velocity_data['y'].min(), velocity_data['y'].max() + bin_size, bin_size)
    
    # Create a 2D histogram of velocity data
    x_indices = np.digitize(velocity_data['x'], x_bins) - 1
    y_indices = np.digitize(velocity_data['y'], y_bins) - 1
    
    # Initialize arrays for binned velocity data
    binned_velocities = []
    
    # Calculate average velocity in each bin
    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            # Skip edge bins if edge_cutoff > 0
            if (i < edge_cutoff or i >= len(x_bins) - 1 - edge_cutoff or 
                j < edge_cutoff or j >= len(y_bins) - 1 - edge_cutoff):
                continue
                
            # Get particles in this bin
            mask = (x_indices == i) & (y_indices == j)
            bin_particles = velocity_data[mask]
            
            if len(bin_particles) > 0:
                # Calculate average velocity in this bin
                avg_vx = bin_particles['vx'].mean()
                avg_vy = bin_particles['vy'].mean()
                
                # Store bin center and average velocity
                binned_velocities.append({
                    'x': (x_bins[i] + x_bins[i+1]) / 2,
                    'y': (y_bins[j] + y_bins[j+1]) / 2,
                    'vx': avg_vx,
                    'vy': avg_vy
                })
    
    # Convert to DataFrame
    binned_velocities_df = pd.DataFrame(binned_velocities)
    
    # Initialize gradient columns
    tracks_with_parameters['dvx_dx'] = np.nan
    tracks_with_parameters['dvx_dy'] = np.nan
    tracks_with_parameters['dvy_dx'] = np.nan
    tracks_with_parameters['dvy_dy'] = np.nan
    
    if gradient_method == 'loess':
        print("Performing LOESS regression on binned data...")
        # Calculate search radius based on bin size
        search_radius = bin_size * loess_radius_factor
        
        # For each bin, find neighbors within the search radius and perform LOESS
        for i in tqdm(range(len(binned_velocities_df)), desc="LOESS fit"):
            x0, y0 = binned_velocities_df.iloc[i]['x'], binned_velocities_df.iloc[i]['y']
            
            # Find neighbors within search radius
            neighbors = binned_velocities_df[
                (abs(binned_velocities_df['x'] - x0) <= search_radius) & 
                (abs(binned_velocities_df['y'] - y0) <= search_radius) &
                (binned_velocities_df['x'] != x0) & 
                (binned_velocities_df['y'] != y0)
            ]
            
            if len(neighbors) >= 6:  # Need at least 6 points for second-order polynomial
                # Calculate distances for weighting
                distances = np.sqrt((neighbors['x'] - x0)**2 + (neighbors['y'] - y0)**2)
                
                # Relative coordinates for the fit
                x_rel = neighbors['x'] - x0
                y_rel = neighbors['y'] - y0
                x_rel *= length_per_pixel
                y_rel *= length_per_pixel

                # Design matrix for third-order polynomial: [1, x, y, x², y², xy, x³, y³, x²y, xy²]
                X = np.column_stack([
                    np.ones(len(neighbors)),
                    x_rel,
                    y_rel,
                    x_rel**2,
                    y_rel**2,
                    x_rel * y_rel])
                '''
                    x_rel**3,
                    y_rel**3,
                    x_rel**2 * y_rel,
                    x_rel * y_rel**2
                ])'''
                
                # Tricube weights
                max_d = distances.max()
                if max_d == 0:
                    w = np.ones_like(distances)
                else:
                    w = (1 - (distances / max_d)**3)**3
                    w[distances > max_d] = 0
                
                # Fit local regression for vx
                y_vx = neighbors['vx']
                model_vx = sm.WLS(y_vx, X, weights=w).fit()
                
                # Extract gradients (coefficients of x and y terms)
                dvx_dx = model_vx.params.iloc[1]  # Coefficient of x
                dvx_dy = model_vx.params.iloc[2]  # Coefficient of y
                
                # Fit local regression for vy
                y_vy = neighbors['vy']
                model_vy = sm.WLS(y_vy, X, weights=w).fit()
                
                # Extract gradients
                dvy_dx = model_vy.params.iloc[1]  # Coefficient of x
                dvy_dy = model_vy.params.iloc[2]  # Coefficient of y
                
                # Find particles in this bin
                bin_mask = (
                    (tracks_with_parameters['x'] >= x0 - bin_size/2) & 
                    (tracks_with_parameters['x'] < x0 + bin_size/2) & 
                    (tracks_with_parameters['y'] >= y0 - bin_size/2) & 
                    (tracks_with_parameters['y'] < y0 + bin_size/2)
                )
                
                # Apply gradients to all particles in this bin
                for idx in tracks_with_parameters[bin_mask].index:
                    tracks_with_parameters.loc[idx, 'dvx_dx'] = dvx_dx
                    tracks_with_parameters.loc[idx, 'dvx_dy'] = dvx_dy
                    tracks_with_parameters.loc[idx, 'dvy_dx'] = dvy_dx
                    tracks_with_parameters.loc[idx, 'dvy_dy'] = dvy_dy
                    
                    # Calculate FTP and MVGT
                    calculate_ftp_mvgt(tracks_with_parameters, idx, dvx_dx, dvx_dy, dvy_dx, dvy_dy)
    
    elif gradient_method == 'central_differences':
        print("Calculating gradients using central differences...")
        # Calculate velocity gradients using central differences
        for i in tqdm(range(len(binned_velocities_df)), desc="Central differences"):
            x = binned_velocities_df.iloc[i]['x']
            y = binned_velocities_df.iloc[i]['y']
            
            # Find neighboring bins
            neighbors = binned_velocities_df[
                (abs(binned_velocities_df['x'] - x) <= bin_size * 1.5) & 
                (abs(binned_velocities_df['y'] - y) <= bin_size * 1.5) &
                (binned_velocities_df['x'] != x) & 
                (binned_velocities_df['y'] != y)
            ]
            
            if len(neighbors) >= 4:  # Need at least 4 neighbors for gradient calculation
                # Calculate velocity gradients using central differences
                # Convert bin_size from pixels to physical units (mm)
                dx = bin_size * length_per_pixel
                dy = bin_size * length_per_pixel
                
                # Find right, left, top, and bottom neighbors
                right = neighbors[neighbors['x'] > x].sort_values('x').iloc[0] if len(neighbors[neighbors['x'] > x]) > 0 else None
                left = neighbors[neighbors['x'] < x].sort_values('x', ascending=False).iloc[0] if len(neighbors[neighbors['x'] < x]) > 0 else None
                top = neighbors[neighbors['y'] > y].sort_values('y').iloc[0] if len(neighbors[neighbors['y'] > y]) > 0 else None
                bottom = neighbors[neighbors['y'] < y].sort_values('y', ascending=False).iloc[0] if len(neighbors[neighbors['y'] < y]) > 0 else None
                
                # Calculate velocity gradients
                if right is not None and left is not None and top is not None and bottom is not None:
                    # Velocities are already in mm/s, and dx/dy are in mm
                    # So gradients will be in (mm/s)/mm = 1/s
                    dvx_dx = (right['vx'] - left['vx']) / (2 * dx)
                    dvx_dy = (top['vx'] - bottom['vx']) / (2 * dy)
                    dvy_dx = (right['vy'] - left['vy']) / (2 * dx)
                    dvy_dy = (top['vy'] - bottom['vy']) / (2 * dy)
                    
                    # Find particles in this bin
                    bin_mask = (
                        (tracks_with_parameters['x'] >= x - bin_size/2) & 
                        (tracks_with_parameters['x'] < x + bin_size/2) & 
                        (tracks_with_parameters['y'] >= y - bin_size/2) & 
                        (tracks_with_parameters['y'] < y + bin_size/2)
                    )
                    
                    # Apply gradients to all particles in this bin
                    for idx in tracks_with_parameters[bin_mask].index:
                        tracks_with_parameters.loc[idx, 'dvx_dx'] = dvx_dx
                        tracks_with_parameters.loc[idx, 'dvx_dy'] = dvx_dy
                        tracks_with_parameters.loc[idx, 'dvy_dx'] = dvy_dx
                        tracks_with_parameters.loc[idx, 'dvy_dy'] = dvy_dy
                        
                        # Calculate FTP and MVGT
                        calculate_ftp_mvgt(tracks_with_parameters, idx, dvx_dx, dvx_dy, dvy_dx, dvy_dy)
    
    else:
        raise ValueError(f"Unknown gradient method: {gradient_method}. Use 'central_differences' or 'loess'.")
    
    return tracks_with_parameters

def calculate_ftp_mvgt(tracks_df: pd.DataFrame, idx: int, dvx_dx: float, dvx_dy: float, dvy_dx: float, dvy_dy: float) -> None:
    """
    Calculate Flow Type Parameter (FTP) and Mean Velocity Gradient Tensor (MVGT) from velocity gradients.
    
    Args:
        tracks_df: DataFrame with particle trajectories
        idx: Index of the particle in the DataFrame
        dvx_dx, dvx_dy, dvy_dx, dvy_dy: Velocity gradient components
    """
    # Calculate velocity gradient tensor
    grad_tensor = np.array([[dvx_dx, dvx_dy], [dvy_dx, dvy_dy]])
    
    # Calculate strain rate tensor (symmetric part)
    strain_rate = 0.5 * (grad_tensor + grad_tensor.T)
    
    # Calculate vorticity tensor (antisymmetric part)
    vorticity = 0.5 * (grad_tensor - grad_tensor.T)
    
    # Calculate flow type parameter (FTP)
    mag_e = np.sqrt(dvx_dx**2 + dvy_dy**2 + 0.5*(dvx_dy + dvy_dx)**2)
    mag_o = np.sqrt(0.25*((dvx_dy - dvy_dx)**2 + (dvy_dx - dvx_dy)**2))
    
    if mag_e + mag_o != 0:
        ftp = (mag_e - mag_o)/(mag_e + mag_o)
    else:
        ftp = 0
    
    # Calculate mean velocity gradient tensor (MVGT)
    mvgt = np.sqrt(dvx_dx**2 + dvx_dy**2 + dvy_dx**2 + dvy_dy**2)
    
    # Store FTP and MVGT
    tracks_df.loc[idx, 'ftp'] = ftp
    tracks_df.loc[idx, 'mvgt'] = mvgt

def analyze_particles(tracks: pd.DataFrame, config: AnalysisConfig, image_size: Tuple[int, int]) -> pd.DataFrame:
    """
    Analyze particle tracks to calculate velocities, FTP, and MVGT.
    
    Args:
        tracks: DataFrame with particle tracks
        config: Analysis configuration
        image_size: Size of the image (width, height)
        
    Returns:
        DataFrame with analyzed tracks
    """
    print("Analyzing tracks...")
    # Get image dimensions for center coordinates
    image_height, image_width = image_size
    center_x = image_width / 2
    center_y = image_height / 2
    
    # Calculate velocities using frame rate and length per pixel
    tracks_with_velocity = calculate_velocities(tracks, config.image.frame_rate, config.image.length_per_pixel)
    
    # Calculate FTP and MVGT
    analyzed_tracks = calculate_ftp_and_mvgt(tracks_with_velocity, 
                                            length_per_pixel=config.image.length_per_pixel,
                                            gradient_method=config.gradient_method)
    
    # Calculate deformation rate G = MVGT/sqrt(1 + FTP²)
    print("Calculating deformation rate...")
    analyzed_tracks['deformation_rate'] = analyzed_tracks['mvgt'] / np.sqrt(1 + analyzed_tracks['ftp']**2)
 
    return analyzed_tracks 