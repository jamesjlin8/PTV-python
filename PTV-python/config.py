from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class ImageConfig:
    """Configuration for image processing"""
    image_dir: str
    base_name: str
    num_images: Optional[int]
    length_per_pixel: float
    frame_rate: int
    length_unit: str
    time_unit: str

@dataclass
class TrackingConfig:
    """Configuration for particle tracking"""
    diameter: int
    minmass: float
    search_range: int
    memory: int
    min_track_length: int

@dataclass
class BackgroundConfig:
    """Configuration for background subtraction"""
    method: str  # 'median', 'mean', or 'static'
    background_image: Optional[str]
    window_size: int

@dataclass
class AnalysisConfig:
    """Main configuration class"""
    image: ImageConfig
    background: BackgroundConfig
    tracking: TrackingConfig
    output: str
    gradient_method: str  # 'central_differences' or 'loess'