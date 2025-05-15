import trackpy as tp
import numpy as np
import pandas as pd
import pims
from typing import List, Dict, Tuple, Union, Optional
from config import TrackingConfig

def locate_particles(frame: Union[np.ndarray, pims.Frame], config: TrackingConfig) -> pd.DataFrame:
    """
    Locate particles in a single frame using trackpy.
    
    Args:
        frame: Image array or PIMS Frame
        config: Tracking configuration
        
    Returns:
        DataFrame with particle locations
    """
    return tp.locate(
        frame,
        diameter=config.diameter,
        minmass=config.minmass
    )

def link_particles(frames: List[pd.DataFrame], config: TrackingConfig) -> pd.DataFrame:
    """
    Link particle locations across frames to form trajectories.
    
    Args:
        frames: List of DataFrames with particle locations
        config: Tracking configuration
        
    Returns:
        DataFrame with linked trajectories
    """
    print("Linking particles across frames...")
    # Concatenate all frames into a single DataFrame
    all_particles = pd.concat(frames, ignore_index=True)
    
    # Create a predictor that uses the nearest velocity
    predictor = tp.predict.NearestVelocityPredict(span=1)
    
    # Use tp.link to link particles across frames with velocity prediction
    linked_tracks = predictor.link_df(
        all_particles,
        search_range=config.search_range,
        memory=config.memory,
        neighbor_strategy='KDTree',
        link_strategy='auto'
    )
    
    return linked_tracks

def filter_tracks(tracks: pd.DataFrame, config: TrackingConfig) -> pd.DataFrame:
    """
    Filter tracks based on minimum length using trackpy's filter_stubs.
    
    Args:
        tracks: DataFrame with linked trajectories
        config: Tracking configuration
        
    Returns:
        Filtered DataFrame
    """
    print("Filtering tracks...")
    # Use trackpy's filter_stubs function
    filtered_tracks = tp.filtering.filter_stubs(tracks, threshold=config.min_track_length)
    
    return filtered_tracks

def track_particles(images: Union[pims.ImageSequence, List[np.ndarray]], 
                     config: TrackingConfig, 
                     frame_rate: float) -> pd.DataFrame:
    """
    Process entire image sequence to track particles.
    
    Args:
        images: PIMS ImageSequence or list of image arrays
        config: Tracking configuration
        frame_rate: Camera frame rate in fps
        
    Returns:
        DataFrame with particle trajectories
    """
    print("Locating particles in frames...")
    
    # Convert to list if PIMS sequence
    if isinstance(images, pims.ImageSequence):
        frames = list(images)
    else:
        frames = images
    
    # Locate particles in each frame
    particle_frames = []
    
    for i, frame in enumerate(frames):
        # Locate particles in the current frame
        frame_particles = locate_particles(frame, config)
        
        # Add frame number to the DataFrame
        if not frame_particles.empty:
            frame_particles['frame'] = i
        
        particle_frames.append(frame_particles)

    tracks = link_particles(particle_frames, config)
    
    # Filter tracks and save the filtered tracks
    filtered_tracks = filter_tracks(tracks, config)
    
    return filtered_tracks