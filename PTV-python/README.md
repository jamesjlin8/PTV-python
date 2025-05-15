# Particle Tracking Velocimetry using trackpy

This package provides a Python implementation of Particle Tracking Velocimetry (PTV) using the `trackpy` library. It includes background subtraction, particle tracking, and visualization capabilities.

## Features

- Image sequence loading and preprocessing
- Background subtraction (median, mean, or static background)
- Particle detection and tracking using trackpy
- Velocity calculation
- Visualization of trajectories and velocity fields
- Results export in various formats

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd trackpy
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

The package can be used from the command line:

```bash
python -m src.main --image-dir /path/to/images \
                  --base-name "image_sequence" \
                  --num-images 200 \
                  --output-dir results
```

### Python API

You can also use the package in your Python code:

```python
from trackpy.src import (
    load_image_sequence,
    create_background,
    subtract_background,
    process_sequence,
    save_results,
    DEFAULT_CONFIG
)

# Load images
images = load_image_sequence(
    image_dir="/path/to/images",
    base_name="image_sequence",
    num_images=200
)

# Create and subtract background
background = create_background(images, DEFAULT_CONFIG.background)
processed_images = subtract_background(images, background)

# Track particles
tracks = process_sequence(
    processed_images,
    DEFAULT_CONFIG.tracking,
    DEFAULT_CONFIG.image.frame_rate
)

# Save results
save_results(tracks, "results")
```

## Configuration

The package uses a configuration system to customize the analysis. You can modify the default configuration in `src/config.py` or create your own configuration file.

Example configuration:
```python
from trackpy.src import AnalysisConfig, ImageConfig, TrackingConfig, BackgroundConfig

config = AnalysisConfig(
    image=ImageConfig(
        length_per_pixel=2/187,  # mm/pixel
        frame_rate=280,  # fps
        image_resolution=(1024, 544)
    ),
    tracking=TrackingConfig(
        particle_diameter=10,
        min_intensity=30,
        search_radius=4
    ),
    background=BackgroundConfig(
        method="median",
        window_size=10
    )
)
```

## Output

The analysis produces the following outputs in the specified output directory:

- `trajectories.csv`: Raw trajectory data
- `summary.txt`: Summary statistics
- `trajectories.png`: Plot of particle trajectories
- `velocity_field.png`: Velocity field visualization
- `speed_distribution.png`: Distribution of particle speeds

## License

This project is licensed under the MIT License - see the LICENSE file for details. 