import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Tuple, Dict, List, Optional
from config import AnalysisConfig
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from scipy.spatial import cKDTree


def _robust_loess_fit(X: np.ndarray, y: np.ndarray, weights: np.ndarray, 
                     regularization: float = 1e-6) -> Tuple[float, float]:
    """
    Robust LOESS fitting with regularization and condition number checking.
    
    Args:
        X: Design matrix
        y: Response variable
        weights: Weights for weighted least squares
        regularization: Regularization parameter
        
    Returns:
        Tuple of (gradient_x, gradient_y)
    """
    try:
        # Convert to numpy arrays to avoid pandas indexing issues
        X = np.asarray(X)
        y = np.asarray(y)
        weights = np.asarray(weights)
        
        # Add regularization to diagonal
        XTX = X.T @ (weights[:, None] * X)
        XTX += regularization * np.eye(XTX.shape[0])
        
        # Check condition number
        cond_num = np.linalg.cond(XTX)
        if cond_num > 1e12:
            regularization *= 10
            XTX += regularization * np.eye(XTX.shape[0])
        
        XTy = X.T @ (weights * y)
        params = np.linalg.solve(XTX, XTy)
        
        return params[1], params[2]  # Coefficients of x and y terms
        
    except np.linalg.LinAlgError as e:
        # Fallback to simple linear regression
        try:
            model = sm.WLS(y, X, weights=weights).fit()
            return model.params.iloc[1], model.params.iloc[2]
        except Exception as fallback_error:
            # Return NaN if all methods fail, but log the error for debugging
            return np.nan, np.nan


def calculate_gaussian_weights_numba(distances: np.ndarray, sigma: float) -> np.ndarray:
    """
    Calculate Gaussian weights using numpy (Numba removed due to compatibility issues).
    
    Args:
        distances: Array of distances
        sigma: Standard deviation for Gaussian weighting
        
    Returns:
        Array of Gaussian weights
    """
    return np.exp(-0.5 * (distances / sigma)**2)


def _calculate_particle_velocity_vectorized(args):
    """
    Vectorized velocity calculation for a single particle.
    Much more efficient than the original loop-based approach.
    
    Args:
        args: Tuple of (particle_id, group, frame_rate, length_per_pixel)
        
    Returns:
        DataFrame with velocity data for this particle
    """
    particle_id, group, frame_rate, length_per_pixel = args
    
    try:
        # Sort by frame to ensure chronological order
        group = group.sort_values('frame').copy()
        
        if len(group) < 2:
            group['vx'] = np.nan
            group['vy'] = np.nan
            group['speed'] = np.nan
            return group
        
        # Calculate time step
        dt = 1.0 / frame_rate
        
        # Vectorized velocity calculation using pandas diff()
        group['vx'] = group['x'].diff() * length_per_pixel / dt
        group['vy'] = group['y'].diff() * length_per_pixel / dt
        
        # Calculate speed
        group['speed'] = np.sqrt(group['vx']**2 + group['vy']**2)
        
        return group
        
    except Exception as e:
        # Return NaN velocities if calculation fails
        group['vx'] = np.nan
        group['vy'] = np.nan
        group['speed'] = np.nan
        return group

def calculate_velocities(tracks: pd.DataFrame, frame_rate: float, length_per_pixel: float, n_cores: int = None) -> pd.DataFrame:
    """
    Calculate particle velocities from trajectories using multiprocessing.
    
    Args:
        tracks: DataFrame with linked trajectories
        frame_rate: Camera frame rate in fps
        length_per_pixel: Physical length per pixel in mm
        n_cores: Number of CPU cores to use (None for all available)
        
    Returns:
        DataFrame with added velocity columns (vx, vy, speed in mm/s)
    """
    tracks_with_velocity = tracks.copy()
    tracks_with_velocity = tracks_with_velocity.reset_index(drop=True)
    tracks_with_velocity = tracks_with_velocity.sort_values(['particle', 'frame'])
    
    particle_groups = list(tracks_with_velocity.groupby('particle'))
    
    if n_cores is None:
        n_cores = mp.cpu_count()
    n_cores = min(n_cores, len(particle_groups), mp.cpu_count())
    
    args_list = [(particle_id, group, frame_rate, length_per_pixel) 
                 for particle_id, group in particle_groups]
    
    with mp.Pool(processes=n_cores) as pool:
        results = list(tqdm(
            pool.imap(_calculate_particle_velocity_vectorized, args_list),
            total=len(args_list),
            desc="Calculating velocities"
        ))
    
    tracks_with_velocity = pd.concat(results, ignore_index=True)
    # Speed already calculated in _calculate_particle_velocity_vectorized, no need to recalculate
    
    return tracks_with_velocity

def _calculate_loess_gradients(args):
    """
    Helper function to calculate LOESS gradients for a single bin.
    Used for multiprocessing.
    
    Args:
        args: Tuple of (bin_index, binned_velocities_df, search_radius, length_per_pixel, bin_size)
        
    Returns:
        Dictionary with gradient results for this bin
    """
    bin_index, binned_velocities_df, search_radius, length_per_pixel, bin_size = args
    
    x0, y0 = binned_velocities_df.iloc[bin_index]['x'], binned_velocities_df.iloc[bin_index]['y']
    
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

        # Design matrix for second-order polynomial: [1, x, y, x², y², xy]
        X = np.column_stack([
            np.ones(len(neighbors)),
            x_rel,
            y_rel,
            x_rel**2,
            y_rel**2,
            x_rel * y_rel])
        
        # Tricube weights
        max_d = distances.max()
        if max_d == 0:
            w = np.ones_like(distances)
        else:
            w = (1 - (distances / max_d)**3)**3
            w[distances > max_d] = 0
        
        # Fit local regression for vx with regularization
        y_vx = neighbors['vx']
        dvx_dx, dvx_dy = _robust_loess_fit(X, y_vx, w)
        
        # Fit local regression for vy with regularization
        y_vy = neighbors['vy']
        dvy_dx, dvy_dy = _robust_loess_fit(X, y_vy, w)
        
        return {
            'bin_index': bin_index,
            'dvx_dx': dvx_dx,
            'dvx_dy': dvx_dy,
            'dvy_dx': dvy_dx,
            'dvy_dy': dvy_dy,
            'success': True
        }
    else:
        return {
            'bin_index': bin_index,
            'dvx_dx': np.nan,
            'dvx_dy': np.nan,
            'dvy_dx': np.nan,
            'dvy_dy': np.nan,
            'success': False
        }

def _calculate_gaussian_weighted_velocity(bin_center_x: float, bin_center_y: float, 
                                        all_particles: pd.DataFrame, 
                                        sigma: float = None,
                                        kdtree: Optional[cKDTree] = None,
                                        particle_coords: Optional[np.ndarray] = None) -> Dict:
    """
    Calculate Gaussian weighted average velocity for a bin using particles from the entire field.
    
    This implements a discrete approximation of a convolution integral where the velocity
    field is convolved with a Gaussian kernel. The mathematical form is:
    
    v_weighted(x₀, y₀) = ∫∫ v(x, y) * G(x-x₀, y-y₀) dx dy
    
    Where G(x, y) = (1/(2πσ²)) * exp(-(x² + y²)/(2σ²)) is the 2D Gaussian kernel.
    
    For discrete particles, this becomes:
    v_weighted = Σᵢ wᵢ * vᵢ / Σᵢ wᵢ
    
    Where wᵢ = exp(-0.5 * (rᵢ/σ)²) and rᵢ is the distance from particle i to bin center.
    
    Args:
        bin_center_x: X-coordinate of bin center (pixels)
        bin_center_y: Y-coordinate of bin center (pixels)
        all_particles: DataFrame of ALL particles in the field (not just those in this bin)
        sigma: Standard deviation for Gaussian weighting in pixels (if None, uses default)
        kdtree: Optional pre-computed KDTree for spatial queries (for optimization)
        particle_coords: Optional pre-computed particle coordinates array (for optimization)
        
    Returns:
        Dictionary with weighted average velocities and other statistics
        Velocities are in physical units (mm/s) as stored in the DataFrame
    """
    if len(all_particles) == 0:
        return {
            'vx': np.nan,
            'vy': np.nan,
            'speed': np.nan,
            'num_particles': 0
        }
    
    # Use KDTree for efficient spatial queries if provided
    # Note: When KDTree is provided, all_particles should already be filtered to valid particles
    if kdtree is not None and particle_coords is not None:
        # Find particles within 3*sigma radius for efficiency
        search_radius = 3 * sigma if sigma is not None else 30.0
        query_point = np.array([[bin_center_x, bin_center_y]])
        indices = kdtree.query_ball_point(query_point, r=search_radius)[0]
        
        if len(indices) == 0:
            return {
                'vx': np.nan,
                'vy': np.nan,
                'speed': np.nan,
                'num_particles': 0
            }
        
        # Get particle coordinates and velocities for nearby particles
        # KDTree indices correspond directly to all_particles when KDTree is provided
        nearby_coords = particle_coords[indices]
        nearby_particles = all_particles.iloc[indices]
        
        # Calculate distances
        distances = np.sqrt((nearby_coords[:, 0] - bin_center_x)**2 + 
                           (nearby_coords[:, 1] - bin_center_y)**2)
    else:
        # Fallback: filter out particles with NaN velocities
        valid_velocity_mask = np.isfinite(all_particles['vx']) & np.isfinite(all_particles['vy'])
        if not np.any(valid_velocity_mask):
            return {
                'vx': np.nan,
                'vy': np.nan,
                'speed': np.nan,
                'num_particles': 0
            }
        valid_particles = all_particles.loc[valid_velocity_mask]
        
        # Calculate distances from all valid particles
        distances = np.sqrt((valid_particles['x'] - bin_center_x)**2 + 
                           (valid_particles['y'] - bin_center_y)**2)
        nearby_particles = valid_particles
    
    # If sigma not provided, use a reasonable default
    if sigma is None:
        sigma = 10.0  # Default sigma value in pixels
    
    # Calculate Gaussian weights
    weights = calculate_gaussian_weights_numba(distances, sigma)
    
    # Only consider particles with significant weights (within ~3 sigma)
    significant_mask = weights > 0.01  # Only particles with weight > 1% of max weight
    
    if not np.any(significant_mask):
        return {
            'vx': np.nan,
            'vy': np.nan,
            'speed': np.nan,
            'num_particles': 0
        }
    
    # Filter particles and weights to only significant ones
    significant_particles = nearby_particles.iloc[significant_mask] if kdtree is not None else nearby_particles[significant_mask]
    significant_weights = weights[significant_mask]
    
    # Normalize weights to ensure Σᵢ wᵢ = 1
    significant_weights = significant_weights / np.sum(significant_weights)
    
    # Calculate weighted averages: v_weighted = Σᵢ wᵢ * vᵢ
    # Note: vx and vy are already in physical units (mm/s) from velocity calculation
    weighted_vx = np.sum(significant_weights * significant_particles['vx'].values)
    weighted_vy = np.sum(significant_weights * significant_particles['vy'].values)
    weighted_speed = np.sqrt(weighted_vx**2 + weighted_vy**2)
    
    return {
        'vx': weighted_vx,
        'vy': weighted_vy,
        'speed': weighted_speed,
        'num_particles': len(significant_particles)
    }

def calculate_gradients(tracks: pd.DataFrame, 
                          bin_size: int = 8, 
                          length_per_pixel: float = 1.0,
                          loess_radius_factor: float = 2.0,
                          image_size: Tuple[int, int] = None,
                          min_particles: int = 1,
                          gaussian_sigma: float = None,
                          n_cores: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate gradients from particle trajectories using LOESS regression.
    Bins across the entire image frame with Gaussian weighted averaging.
    
    All velocity gradients are calculated in physical units:
    - dvx_dx, dvy_dy: units of s⁻¹ (velocity gradient in x and y directions)
    - dvx_dy, dvy_dx: units of s⁻¹ (cross-derivatives)
    
    Args:
        tracks: DataFrame with particle trajectories and velocities (vx, vy in mm/s)
        bin_size: Size of bins for velocity gradient calculation in pixels
        length_per_pixel: Physical length per pixel in mm
        loess_radius_factor: Factor to multiply bin_size by to determine LOESS search radius
        image_size: Size of the image (width, height) in pixels
        min_particles: Minimum number of particles required in a bin for calculation
        gaussian_sigma: Standard deviation for Gaussian weighting in pixels (if None, uses default)
        n_cores: Number of CPU cores to use for multiprocessing
        
    Returns:
        Tuple of (DataFrame with added FTP, MVGT, and deformation_rate columns, 
                 DataFrame with binned velocity data)
    """
    tracks_with_parameters = tracks.copy()
    tracks_with_parameters['ftp'] = np.nan
    tracks_with_parameters['mvgt'] = np.nan
    tracks_with_parameters['deformation_rate'] = np.nan
    
    velocity_data = tracks[['x', 'y', 'vx', 'vy']].copy()
    
    # Use full-frame binning
    width, height = image_size if image_size is not None else (
        int(velocity_data['x'].max()) + 1,
        int(velocity_data['y'].max()) + 1
    )
    
    # Validate inputs
    if bin_size <= 0:
        raise ValueError(f"bin_size must be positive, got {bin_size}")
    if length_per_pixel <= 0:
        raise ValueError(f"length_per_pixel must be positive, got {length_per_pixel}")
    
    x_bins = np.arange(0, width + 1, bin_size)
    y_bins = np.arange(0, height + 1, bin_size)
    
    # Build KDTree for efficient spatial queries in Gaussian weighting
    # Filter to valid particles first, then build KDTree
    valid_mask = np.isfinite(velocity_data['vx']) & np.isfinite(velocity_data['vy'])
    valid_particles = velocity_data[valid_mask].copy()
    
    if len(valid_particles) > 0:
        particle_coords = valid_particles[['x', 'y']].values
        kdtree = cKDTree(particle_coords)
        # Pass the filtered valid_particles DataFrame for KDTree indexing
        velocity_data_for_kdtree = valid_particles
    else:
        kdtree = None
        particle_coords = None
        velocity_data_for_kdtree = None
    
    # Set default sigma if not provided
    if gaussian_sigma is None:
        gaussian_sigma = bin_size / 4.0  # Default: 1/4 of bin size
    
    # Create binned velocities with consistent coordinate system
    # Store bin indices for visualization (0-indexed from bottom-left)
    binned_velocities = []
    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            # Calculate bin center coordinates in pixel space
            bin_center_x = (x_bins[i] + x_bins[i+1]) / 2
            bin_center_y = (y_bins[j] + y_bins[j+1]) / 2

            # Use Gaussian weighting with ALL particles in the field for proper smoothing
            # When KDTree is available, use the pre-filtered valid particles
            particles_for_weighting = velocity_data_for_kdtree if kdtree is not None else velocity_data
            weighted_velocities = _calculate_gaussian_weighted_velocity(
                bin_center_x, bin_center_y, particles_for_weighting, gaussian_sigma,
                kdtree=kdtree, particle_coords=particle_coords
            )
            
            # Store bin indices (i, j) for grid mapping
            # Only include bins that have significant particle contributions
            if weighted_velocities['num_particles'] >= min_particles:
                binned_velocities.append({
                    'x_bin': i,
                    'y_bin': j,
                    'x': bin_center_x,
                    'y': bin_center_y,
                    'vx': weighted_velocities['vx'],
                    'vy': weighted_velocities['vy'],
                    'speed': weighted_velocities['speed'],
                    'num_particles': weighted_velocities['num_particles']
                })
            else:
                # No significant particle contributions - leave as NaN
                binned_velocities.append({
                    'x_bin': i,
                    'y_bin': j,
                    'x': bin_center_x,
                    'y': bin_center_y,
                    'vx': np.nan,
                    'vy': np.nan,
                    'speed': np.nan,
                    'num_particles': 0
                })
    
    binned_velocities_df = pd.DataFrame(binned_velocities)
    
    # Initialize gradient columns in tracks_with_parameters
    tracks_with_parameters['dvx_dx'] = np.nan
    tracks_with_parameters['dvx_dy'] = np.nan
    tracks_with_parameters['dvy_dx'] = np.nan
    tracks_with_parameters['dvy_dy'] = np.nan
    
    # Initialize gradient columns in binned_velocities_df for gradient calculation
    binned_velocities_df['dvx_dx'] = np.nan
    binned_velocities_df['dvx_dy'] = np.nan
    binned_velocities_df['dvy_dx'] = np.nan
    binned_velocities_df['dvy_dy'] = np.nan
    
    # Initialize flow parameter columns in binned_velocities_df
    binned_velocities_df['ftp'] = np.nan
    binned_velocities_df['mvgt'] = np.nan
    binned_velocities_df['deformation_rate'] = np.nan
    
    if n_cores is None:
        n_cores = mp.cpu_count()
    n_cores = min(n_cores, len(binned_velocities_df), mp.cpu_count())
    # Calculate search radius based on bin size
    search_radius = bin_size * loess_radius_factor
    
    # Prepare arguments for multiprocessing
    args_list = [(i, binned_velocities_df, search_radius, length_per_pixel, bin_size) 
                 for i in range(len(binned_velocities_df))]
    
    # Use multiprocessing to calculate gradients
    with mp.Pool(processes=n_cores) as pool:
        results = list(tqdm(
            pool.imap(_calculate_loess_gradients, args_list),
            total=len(args_list),
            desc="LOESS fit"
        ))
    
    # Apply results to binned data and tracks efficiently
    successful_results = [r for r in results if r['success']]
    
    # Create a mapping of bin coordinates to gradient values for faster lookup
    gradient_map = {}
    for result in successful_results:
        bin_index = result['bin_index']
        x0, y0 = binned_velocities_df.iloc[bin_index]['x'], binned_velocities_df.iloc[bin_index]['y']
        gradient_map[(x0, y0)] = {
            'dvx_dx': result['dvx_dx'],
            'dvx_dy': result['dvx_dy'],
            'dvy_dx': result['dvy_dx'],
            'dvy_dy': result['dvy_dy']
        }
        
        # Add gradients to binned data
        binned_velocities_df.loc[bin_index, 'dvx_dx'] = result['dvx_dx']
        binned_velocities_df.loc[bin_index, 'dvx_dy'] = result['dvx_dy']
        binned_velocities_df.loc[bin_index, 'dvy_dx'] = result['dvy_dx']
        binned_velocities_df.loc[bin_index, 'dvy_dy'] = result['dvy_dy']
    
    # Apply gradients to all particles at once using vectorized operations
    
    for (x0, y0), gradients in gradient_map.items():
        # Find particles in this bin
        bin_mask = (
            (tracks_with_parameters['x'] >= x0 - bin_size/2) & 
            (tracks_with_parameters['x'] < x0 + bin_size/2) & 
            (tracks_with_parameters['y'] >= y0 - bin_size/2) & 
            (tracks_with_parameters['y'] < y0 + bin_size/2)
        )
        
        # Apply gradients to all particles in this bin at once
        tracks_with_parameters.loc[bin_mask, 'dvx_dx'] = gradients['dvx_dx']
        tracks_with_parameters.loc[bin_mask, 'dvx_dy'] = gradients['dvx_dy']
        tracks_with_parameters.loc[bin_mask, 'dvy_dx'] = gradients['dvy_dx']
        tracks_with_parameters.loc[bin_mask, 'dvy_dy'] = gradients['dvy_dy']
    
    calculate_ftp_mvgt_deformation_vectorized(tracks_with_parameters)
    calculate_ftp_mvgt_deformation_vectorized(binned_velocities_df)
    
    return tracks_with_parameters, binned_velocities_df

def calculate_ftp_mvgt_deformation_vectorized(tracks_df: pd.DataFrame) -> None:
    """
    Calculate Flow Type Parameter (FTP), Mean Velocity Gradient Tensor (MVGT), and deformation rate 
    from velocity gradients using vectorized operations for all rows at once.
    
    Args:
        tracks_df: DataFrame with particle trajectories and gradient columns
    """
    # Get gradient columns
    dvx_dx = tracks_df['dvx_dx'].values
    dvx_dy = tracks_df['dvx_dy'].values
    dvy_dx = tracks_df['dvy_dx'].values
    dvy_dy = tracks_df['dvy_dy'].values
    
    # Create mask for valid gradients (not NaN)
    valid_mask = ~np.isnan(dvx_dx)
    
    if not np.any(valid_mask):
        return
    
    # Calculate flow parameters only for valid gradients
    dvx_dx_valid = dvx_dx[valid_mask]
    dvx_dy_valid = dvx_dy[valid_mask]
    dvy_dx_valid = dvy_dx[valid_mask]
    dvy_dy_valid = dvy_dy[valid_mask]
    
    # Calculate flow type parameter (FTP)
    mag_e = np.sqrt(dvx_dx_valid**2 + dvy_dy_valid**2 + 0.5*(dvx_dy_valid + dvy_dx_valid)**2)
    mag_o = np.sqrt(0.25*((dvx_dy_valid - dvy_dx_valid)**2 + (dvy_dx_valid - dvx_dy_valid)**2))
    
    # Avoid division by zero
    denominator = mag_e + mag_o
    ftp = np.where(denominator != 0, (mag_e - mag_o) / denominator, 0)
    
    # Calculate mean velocity gradient tensor (MVGT)
    mvgt = np.sqrt(dvx_dx_valid**2 + dvx_dy_valid**2 + dvy_dx_valid**2 + dvy_dy_valid**2)
    
    # Calculate deformation rate G = MVGT/sqrt(1 + FTP²)
    deformation_rate = mvgt / np.sqrt(1 + ftp**2)
    
    # Store results back in DataFrame
    tracks_df.loc[valid_mask, 'ftp'] = ftp
    tracks_df.loc[valid_mask, 'mvgt'] = mvgt
    tracks_df.loc[valid_mask, 'deformation_rate'] = deformation_rate


def analyze_particles(tracks: pd.DataFrame, config: AnalysisConfig, image_size: Tuple[int, int], n_cores: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze particle tracks to calculate velocities, FTP, MVGT, and deformation rate.
    
    Args:
        tracks: DataFrame with particle tracks
        config: Analysis configuration
        image_size: Size of the image (width, height)
        n_cores: Number of CPU cores to use for multiprocessing (None for all available)
        
    Returns:
        Tuple of (analyzed tracks DataFrame, binned data DataFrame)
    """
    # Step 1: Calculate particle velocities from trajectory data
    tracks_with_velocity = calculate_velocities(tracks, config.image.frame_rate, config.image.length_per_pixel, n_cores)
    
    # Step 2: Calculate velocity gradients using LOESS regression for high accuracy
    # This provides the dvx_dx, dvx_dy, dvy_dx, dvy_dy fields needed for streamlines
    analyzed_tracks, binned_data = calculate_gradients(
        tracks_with_velocity,
        bin_size=config.bin_size,
        length_per_pixel=config.image.length_per_pixel,
        loess_radius_factor=config.loess_radius_factor,
        image_size=image_size,
        min_particles=config.min_particles,
        gaussian_sigma=config.gaussian_sigma,
        n_cores=n_cores
    )
 
    return analyzed_tracks, binned_data