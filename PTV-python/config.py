from dataclasses import dataclass, field
from typing import Optional, Tuple

@dataclass
class ImageConfig:
    """Configuration for image processing"""
    image_dir: str
    base_name: str
    num_images: Optional[int]
    length_per_pixel: float
    frame_rate: int
    length_unit: str = "mm"
    time_unit: str = "sec"
    
    def __post_init__(self):
        """Validate image configuration"""
        if self.length_per_pixel <= 0:
            raise ValueError(f"length_per_pixel must be positive, got {self.length_per_pixel}")
        if self.frame_rate <= 0:
            raise ValueError(f"frame_rate must be positive, got {self.frame_rate}")
        if self.num_images is not None and self.num_images <= 0:
            raise ValueError(f"num_images must be positive, got {self.num_images}")

@dataclass
class TrackingConfig:
    """Configuration for particle tracking"""
    diameter: int
    minmass: float
    search_range: int
    memory: int
    min_track_length: int
    
    def __post_init__(self):
        """Validate tracking configuration"""
        if self.diameter <= 0:
            raise ValueError(f"diameter must be positive, got {self.diameter}")
        if self.minmass < 0:
            raise ValueError(f"minmass must be non-negative, got {self.minmass}")
        if self.search_range <= 0:
            raise ValueError(f"search_range must be positive, got {self.search_range}")
        if self.memory < 0:
            raise ValueError(f"memory must be non-negative, got {self.memory}")
        if self.min_track_length <= 0:
            raise ValueError(f"min_track_length must be positive, got {self.min_track_length}")

@dataclass
class BackgroundConfig:
    """Configuration for background subtraction"""
    method: str  # 'median', 'mean', or 'static'
    background_image: Optional[str] = None
    window_size: int = 1000
    
    def __post_init__(self):
        """Validate background configuration"""
        if self.method not in ['median', 'mean', 'static']:
            raise ValueError(f"method must be 'median', 'mean', or 'static', got {self.method}")
        if self.method == 'static' and self.background_image is None:
            raise ValueError("background_image must be provided when method='static'")
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")

@dataclass
class StreamlineConfig:
    """Configuration for streamline calculation"""
    density: int = 2
    min_bins: int = 20
    n_bins: int = 48  # Number of bins for visualization grid
    
    def __post_init__(self):
        """Validate streamline configuration"""
        if self.density <= 0:
            raise ValueError(f"density must be positive, got {self.density}")
        if self.min_bins <= 0:
            raise ValueError(f"min_bins must be positive, got {self.min_bins}")
        if self.n_bins <= 0:
            raise ValueError(f"n_bins must be positive, got {self.n_bins}")

@dataclass
class AnalysisConfig:
    """Main configuration class"""
    image: ImageConfig
    background: BackgroundConfig
    tracking: TrackingConfig
    output: str
    bin_size: int = 6  # Size of bins for velocity and gradient calculations in pixels
    loess_radius_factor: float = 2.0  # Factor to multiply bin_size by for LOESS search radius
    min_particles: int = 1  # Minimum number of particles required in a bin for calculation
    gaussian_sigma: Optional[float] = None  # Standard deviation for Gaussian weighting in pixels (None for auto)
    n_cores: Optional[int] = None  # Number of CPU cores to use (None for all available)
    streamline: StreamlineConfig = field(default_factory=StreamlineConfig)
    custom_streamline_background: Optional[str] = None  # Path to custom background image for streamlines
    
    def __post_init__(self):
        """Validate analysis configuration"""
        if self.bin_size <= 0:
            raise ValueError(f"bin_size must be positive, got {self.bin_size}")
        if self.loess_radius_factor <= 0:
            raise ValueError(f"loess_radius_factor must be positive, got {self.loess_radius_factor}")
        if self.min_particles < 0:
            raise ValueError(f"min_particles must be non-negative, got {self.min_particles}")
        if self.n_cores is not None and self.n_cores <= 0:
            raise ValueError(f"n_cores must be positive, got {self.n_cores}")