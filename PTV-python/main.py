from config import ImageConfig, TrackingConfig, BackgroundConfig, AnalysisConfig, StreamlineConfig
from image_processing import process_images
from tracking import track_particles
from analysis import analyze_particles
from visualization import visualize_results
from streamlines import calculate_streamlines_from_binned_data
from pathlib import Path

# Configuration - all settings consolidated here
IMAGE_DIR = "./CTAB/r2_6_24_25_CTAB_0.5I_0.5W_525fps/frames"  # Directory containing images
BASE_NAME = "frame"  # Base name of image files (e.g., "frame001.tif")
NUM_IMAGES = 1000           # Number of images to process (None for all)
OUTPUT_DIR = "./CTAB/r2_6_24_25_CTAB_0.5I_0.5W_525fps/results_PTV2"    # Directory to save results

# Image processing settings
LENGTH_PER_PIXEL = 1/250  # mm/pixel
FRAME_RATE = 525  # fps
LENGTH_UNIT = "mm"
TIME_UNIT = "sec"

# Background settings
BACKGROUND_METHOD = "median"  # Options: "static", "median", "mean"
BACKGROUND_IMAGE = None       # Path to static background image (if method="static")
BACKGROUND_WINDOW_SIZE = 1000   # Window size for rolling background

# Custom background for streamlines
CUSTOM_STREAMLINE_BACKGROUND = None  # Path to custom background image for streamlines

# Tracking settings
PARTICLE_DIAMETER = 5     # Diameter of particles in pixels
MINMASS = 3              # Minimum intensity for particle detection
SEARCH_RANGE = 3          # Maximum distance to search for particles between frames
MEMORY = 1                 # Number of frames a particle can disappear before being forgotten
MIN_TRACK_LENGTH = 10      # Minimum number of frames for a valid track

# Analysis settings
BIN_SIZE = 7  # Size of bins for velocity and gradient calculations
LOESS_RADIUS_FACTOR = 2  # Factor to multiply bin_size by for LOESS search radius
MIN_PARTICLES = 3  # Minimum number of particles required in a bin for calculation
GAUSSIAN_SIGMA = 6 # Standard deviation for Gaussian weighting (None for auto)

# Streamline settings
STREAMLINE_DENSITY = 1.5  # Density of streamlines (higher = more streamlines)
STREAMLINE_MIN_BINS = 20  # Minimum number of bins a streamline must pass through
STREAMLINE_N_BINS = 48  # Number of bins for visualization grid

# Multiprocessing settings
N_CORES = 2  # Number of CPU cores to use (None for all available, or specify a number)

def main():
    """
    Run PTV analysis with the configuration specified above.
    """
    # Validate input directory
    if not Path(IMAGE_DIR).exists():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")
    
    # Create configuration objects
    image_config = ImageConfig(
        image_dir=IMAGE_DIR,
        base_name=BASE_NAME,
        num_images=NUM_IMAGES,
        length_per_pixel=LENGTH_PER_PIXEL,
        frame_rate=FRAME_RATE,
        length_unit=LENGTH_UNIT,
        time_unit=TIME_UNIT
    )
    
    background_config = BackgroundConfig(
        method=BACKGROUND_METHOD,
        background_image=BACKGROUND_IMAGE,
        window_size=BACKGROUND_WINDOW_SIZE
    )
    
    tracking_config = TrackingConfig(
        diameter=PARTICLE_DIAMETER,
        minmass=MINMASS,
        search_range=SEARCH_RANGE,
        memory=MEMORY,
        min_track_length=MIN_TRACK_LENGTH
    )
    
    streamline_config = StreamlineConfig(
        density=STREAMLINE_DENSITY,
        min_bins=STREAMLINE_MIN_BINS,
        n_bins=STREAMLINE_N_BINS
    )
    
    config = AnalysisConfig(
        image=image_config,
        background=background_config,
        tracking=tracking_config,
        output=OUTPUT_DIR,
        bin_size=BIN_SIZE,
        loess_radius_factor=LOESS_RADIUS_FACTOR,
        min_particles=MIN_PARTICLES,
        gaussian_sigma=GAUSSIAN_SIGMA,
        n_cores=N_CORES,
        streamline=streamline_config,
        custom_streamline_background=CUSTOM_STREAMLINE_BACKGROUND
    )
    
    # Run analysis
    run(config)

def run(config: AnalysisConfig) -> None:
    """
    Run the complete PTV analysis pipeline with streamline analysis.
    
    Workflow:
    1. Process images and extract background
    2. Track particles across frames
    3. Analyze trajectories and calculate LOESS gradients
    4. Generate standard PTV visualizations
    5. Calculate streamfunction and extract streamlines
    
    Args:
        config: Analysis configuration
    """
    # Step 1: Process images and extract background
    processed_images, background, max_intensity = process_images(config)
    
    # Step 2: Track particles across frames
    tracks = track_particles(processed_images, config.tracking, config.image.frame_rate)
    
    # Step 3: Analyze particle trajectories and calculate gradients
    image_size = (processed_images[0].shape[1], processed_images[0].shape[0])
    analyzed_tracks, binned_data = analyze_particles(tracks, config, image_size, config.n_cores)
    
    # Step 4: Analyze streamlines using direct streamplot method
    streamline_results = calculate_streamlines_from_binned_data(
        binned_data=binned_data,
        image_size=image_size,
        bin_size=config.bin_size,
        density=config.streamline.density,
        min_bins=config.streamline.min_bins,
        n_bins=config.streamline.n_bins,
        length_per_pixel=config.image.length_per_pixel
    )
    
    # Step 5: Load custom background for streamlines if specified
    custom_streamline_background = None
    if config.custom_streamline_background is not None:
        import matplotlib.pyplot as plt
        custom_streamline_background = plt.imread(config.custom_streamline_background)
    
    # Step 6: Generate standard PTV visualizations and streamline analysis
    visualize_results(
        analyzed_tracks=analyzed_tracks,
        binned_data=binned_data,
        image_size=image_size,
        output_dir=config.output,
        background=background,
        max_intensity=max_intensity,
        length_per_pixel=config.image.length_per_pixel,
        n_bins=config.streamline.n_bins,
        streamline_results=streamline_results,
        custom_streamline_background=custom_streamline_background
    )
    
    print("Results saved to ", config.output)

if __name__ == "__main__":
    main()