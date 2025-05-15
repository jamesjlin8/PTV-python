import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import os

# Default parameters - modify these values for your specific case
DEFAULT_VIDEO = "./CTAB/extension/converted.mp4"  # Change this to your video path
DEFAULT_OUTPUT_DIR = "./CTAB/extension/frames/"  # Change this to your desired output directory
DEFAULT_BASE_NAME = "frame_"
DEFAULT_START_FRAME = 1000
DEFAULT_END_FRAME = 1500  # Change this based on your video length
DEFAULT_ROI = (436, 28, 1617, 1060)  # Shouldn't matter
DEFAULT_INVERT = True  # Set to True if particles are dark on light background
DEFAULT_NORMALIZE = True
DEFAULT_PREVIEW_FRAME = 500  # Frame number to preview for ROI selection

def select_roi(video_path: str = DEFAULT_VIDEO, frame_number: int = DEFAULT_PREVIEW_FRAME) -> Optional[tuple]:
    """
    Preview video frame and select ROI.
    
    Args:
        video_path: Path to video file
        frame_number: Frame number to preview
        
    Returns:
        Tuple of (x_min, y_min, x_max, y_max) for selected ROI or None if cancelled
    """
    # Check if video file exists
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not video_path.is_file():
        raise ValueError(f"Path is not a file: {video_path}")
    
    print(f"Opening video file: {video_path}")
    
    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read frame
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"Could not read frame {frame_number}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Show frame and select ROI
    print("\nInstructions:")
    print("1. A window will open showing the video frame")
    print("2. Click and drag to select the region of interest")
    print("3. Press Enter to confirm selection")
    print("4. Press 'q' to quit without selecting")
    
    # Select ROI
    roi = cv2.selectROI("Select Region of Interest", gray, False)
    
    # Convert ROI to tuple of integers
    roi = tuple(map(int, roi))
    
    # Draw rectangle on frame
    x, y, w, h = roi
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Show frame with ROI
    cv2.imshow("Selected ROI", frame)
    
    # Wait for key press
    key = cv2.waitKey(0)
    
    # Close windows
    cv2.destroyAllWindows()
    
    # Release video capture
    cap.release()
    
    if key == ord('q'):
        print("\nROI selection cancelled")
        return None
    
    # Convert ROI from (x, y, w, h) to (x_min, y_min, x_max, y_max)
    roi = (x, y, x + w, y + h)
    
    # Print ROI values
    print("\nSelected region:")
    print(f"Top-left corner: ({x}, {y})")
    print(f"Bottom-right corner: ({x + w}, {y + h})")
    
    return roi

def video_to_images(
    video_path: str = DEFAULT_VIDEO,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    base_name: str = DEFAULT_BASE_NAME,
    start_frame: int = DEFAULT_START_FRAME,
    end_frame: Optional[int] = DEFAULT_END_FRAME,
    roi: Optional[tuple] = DEFAULT_ROI,
    invert: bool = DEFAULT_INVERT,
    normalize: bool = DEFAULT_NORMALIZE
) -> None:
    """
    Convert video file to sequence of TIFF images.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save images
        base_name: Base name for output files
        start_frame: First frame to process
        end_frame: Last frame to process (None for all frames)
        roi: Region of interest (x_min, y_min, x_max, y_max)
        invert: Whether to invert the image
        normalize: Whether to normalize pixel values
    """
    # Check if video file exists
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not video_path.is_file():
        raise ValueError(f"Path is not a file: {video_path}")
    
    print(f"Opening video file: {video_path}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Set end frame if not specified
    if end_frame is None:
        end_frame = total_frames
    
    # Validate frame range
    if start_frame < 0:
        start_frame = 0
    if end_frame > total_frames:
        end_frame = total_frames
    if start_frame >= end_frame:
        raise ValueError("start_frame must be less than end_frame")
    
    print(f"Processing frames {start_frame} to {end_frame}")
    
    # Set initial frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Process frames
    frame_count = 0
    for i in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {i}")
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply ROI if specified
        if roi is not None:
            x_min, y_min, x_max, y_max = roi
            gray = gray[y_min:y_max, x_min:x_max]
        
        # Invert if specified
        if invert:
            gray = 255 - gray
        
        # Normalize if specified
        if normalize:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Save as TIFF
        output_file = output_path / f"{base_name}{frame_count:04d}.tif"
        cv2.imwrite(str(output_file), gray)
        
        frame_count += 1
        
        # Print progress
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")
    
    # Release video capture
    cap.release()
    
    print(f"\nConversion complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Output directory: {output_path}")

def main():
    """
    Main function to run the video processing pipeline.
    """
    try:
        print("Video Processing Pipeline")
        print("=======================")
        
        # Step 1: Select ROI
        print("\nStep 1: Select Region of Interest")
        roi = select_roi()
        if roi is None:
            print("ROI selection cancelled. Using default ROI.")
            roi = DEFAULT_ROI
        
        # Step 2: Convert video to images
        print("\nStep 2: Converting Video to Images")
        video_to_images(roi=roi)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    main() 