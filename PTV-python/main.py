from config import ImageConfig, TrackingConfig, BackgroundConfig, AnalysisConfig
from image_processing import process_images
from tracking import track_particles
from analysis import analyze_particles
from visualization import visualize_results

# Image settings
IMAGE_DIR = "./CTAB/extension/frames/"  # Directory containing images
BASE_NAME = "frame"  # Base name of image files (e.g., "frame001.tif")
NUM_IMAGES = 500           # Number of images to process (None for all)
OUTPUT_DIR = "./CTAB/extension/results/"     # Directory to save results

# Image processing settings
LENGTH_PER_PIXEL = 1/270  # mm/pixel
FRAME_RATE = 560  # fps
LENGTH_UNIT = "mm"
TIME_UNIT = "sec"

# Background settings
BACKGROUND_METHOD = "median"  # Options: "static", "median", "mean"
BACKGROUND_IMAGE = None       # Path to static background image (if method="static")
BACKGROUND_WINDOW_SIZE = 500   # Window size for rolling background

# Tracking settings
PARTICLE_DIAMETER = 5     # Diameter of particles in pixels
MINMASS = 10              # Minimum intensity for particle detection
SEARCH_RANGE = 3          # Maximum distance to search for particles between frames
MEMORY = 2                 # Number of frames a particle can disappear before being forgotten
MIN_TRACK_LENGTH = 10      # Minimum number of frames for a valid track

# Analysis settings
GRADIENT_METHOD = "loess"  # Options: "central_differences" or "loess"

def main():
    """
    Run PTV analysis with the configuration specified above.
    """
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
    
    config = AnalysisConfig(
        image=image_config,
        background=background_config,
        tracking=tracking_config,
        output=OUTPUT_DIR,
        gradient_method=GRADIENT_METHOD
    )
    
    # Run analysis
    run(config)

def run(config: AnalysisConfig) -> None:
    """
    Run the complete PTV analysis pipeline.
    
    Args:
        config: Analysis configuration
    """
    # Step 1: Process images
    processed_images, background, max_intensity = process_images(config)
    
    # Step 2: Track particles
    tracks = track_particles(processed_images, config.tracking, config.image.frame_rate)
    
    # Step 3: Analyze particles
    image_size = (processed_images[0].shape[1], processed_images[0].shape[0])
    analyzed_tracks = analyze_particles(tracks, config, image_size)
    
    # Step 4: Visualize results
    print("Visualizing results...")
    image_height, image_width = image_size
    center_x = image_width / 2
    center_y = image_height / 2
    
    visualize_results(
        analyzed_tracks=analyzed_tracks,
        image_size=image_size,
        output_dir=config.output,
        background=background,
        max_intensity=max_intensity,
        bin_size=8,
        length_per_pixel=config.image.length_per_pixel
    )
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()