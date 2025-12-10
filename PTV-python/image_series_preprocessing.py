#!/usr/bin/env python3
"""
Image series preprocessing utilities for PTV analysis.
Interactive workflow: Rotation → Grid Size → Grid Position → Grayscale Inversion
"""

import cv2
import numpy as np
from pathlib import Path
import glob
import os
import argparse

# Default settings
DEFAULT_IMAGE_DIR = "./CTAB/r2_6_24_25_CTAB_0.5I_0.5W_525fps/raw"
DEFAULT_OUTPUT_DIR = "./CTAB/r2_6_24_25_CTAB_0.5I_0.5W_525fps/frames2"
DEFAULT_PATTERN = "*.tiff"

def rotate_image(img, rotation):
    """
    Rotate image by specified angle.
    
    Args:
        img: Input image
        rotation: Rotation angle in degrees
        
    Returns:
        Rotated image
    """
    if rotation == 0:
        return img
    
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
    
    # Perform rotation
    rotated = cv2.warpAffine(img, rotation_matrix, (width, height))
    
    return rotated

def draw_grid_overlay(img, grid_size, offset_x=0, offset_y=0, color=(0, 255, 0)):
    """
    Draw grid overlay on image.
    
    Args:
        img: Input image
        grid_size: Size of grid squares
        offset_x: X offset for grid
        offset_y: Y offset for grid
        color: Grid color (BGR)
        
    Returns:
        Image with grid overlay
    """
    h, w = img.shape[:2]
    display_img = img.copy()
    
    # Draw vertical lines
    for x in range(offset_x % grid_size, w, grid_size):
        cv2.line(display_img, (x, 0), (x, h), color, 1)
    
    # Draw horizontal lines
    for y in range(offset_y % grid_size, h, grid_size):
        cv2.line(display_img, (0, y), (w, y), color, 1)
    
    return display_img

def step1_rotation_selection(image_path):
    """
    Step 1: Interactive rotation selection with 1-degree increments.
    
    Args:
        image_path: Path to sample image
        
    Returns:
        Selected rotation angle in degrees
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to grayscale for preview
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()
    
    rotation = 0
    
    print("\n=== STEP 1: ROTATION SELECTION ===")
    print("Controls:")
    print("  '+' - Rotate 1° clockwise")
    print("  '-' - Rotate 1° counter-clockwise")
    print("  'r' - Rotate 90° clockwise")
    print("  'l' - Rotate 90° counter-clockwise")
    print("  'n' - Next step (keep current rotation)")
    print("  'q' - Quit")
    print(f"Current rotation: {rotation}°")
    
    cv2.namedWindow('Rotation Preview', cv2.WINDOW_NORMAL)
    
    while True:
        # Apply rotation
        rotated_img = rotate_image(img_gray, rotation)
        
        # Add text overlay
        display_img = cv2.cvtColor(rotated_img, cv2.COLOR_GRAY2BGR)
        cv2.putText(display_img, f"Rotation: {rotation}°", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_img, "Use +/- for 1° or r/l for 90°", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Rotation Preview', display_img)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            cv2.destroyAllWindows()
            return None
        elif key == ord('+') or key == ord('='):
            rotation = (rotation + 1) % 360
            print(f"Rotation: {rotation}°")
        elif key == ord('-'):
            rotation = (rotation - 1) % 360
            print(f"Rotation: {rotation}°")
        elif key == ord('r'):
            rotation = (rotation + 90) % 360
            print(f"Rotation: {rotation}°")
        elif key == ord('l'):
            rotation = (rotation - 90) % 360
            print(f"Rotation: {rotation}°")
        elif key == ord('n'):
            break
    
    cv2.destroyAllWindows()
    print(f"Selected rotation: {rotation}°")
    return rotation

def step2_bin_size_selection(image_path, rotation):
    """
    Step 2: Bin size selection for grid calculation.
    
    Args:
        image_path: Path to sample image
        rotation: Selected rotation angle
        
    Returns:
        Selected bin size in pixels
    """
    # Load and rotate image
    img = cv2.imread(image_path)
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()
    
    rotated_img = rotate_image(img_gray, rotation)
    h, w = rotated_img.shape
    
    bin_size = 6  # Default bin size
    
    print("\n=== STEP 2: BIN SIZE SELECTION ===")
    print("Controls:")
    print("  '+' - Increase bin size")
    print("  '-' - Decrease bin size")
    print("  'n' - Next step (keep current bin size)")
    print("  'q' - Quit")
    print(f"Current bin size: {bin_size}px")
    print(f"Image size: {w}x{h}px")
    
    cv2.namedWindow('Bin Size Selection', cv2.WINDOW_NORMAL)
    
    while True:
        # Create display image
        display_img = rotated_img.copy()
        display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
        
        # Draw grid overlay to show bin size
        display_img = draw_grid_overlay(display_img, bin_size, 0, 0, (0, 255, 255))
        
        # Add text overlay
        cv2.putText(display_img, f"Bin Size: {bin_size}px", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(display_img, "Use +/- to adjust bin size", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Bin Size Selection', display_img)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            cv2.destroyAllWindows()
            return None
        elif key == ord('+') or key == ord('='):
            bin_size = min(bin_size + 1, 20)
            print(f"Bin size: {bin_size}px")
        elif key == ord('-'):
            bin_size = max(bin_size - 1, 1)
            print(f"Bin size: {bin_size}px")
        elif key == ord('n'):
            break
    
    cv2.destroyAllWindows()
    print(f"Selected bin size: {bin_size}px")
    return bin_size

def step3_grid_and_crop_selection(image_path, rotation, bin_size):
    """
    Step 3: Grid size and crop positioning combined.
    
    Args:
        image_path: Path to sample image
        rotation: Selected rotation angle
        bin_size: Selected bin size in pixels
        
    Returns:
        Tuple of (num_bins, crop_x, crop_y) for grid and crop position
    """
    # Load and rotate image
    img = cv2.imread(image_path)
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()
    
    rotated_img = rotate_image(img_gray, rotation)
    h, w = rotated_img.shape
    
    # Start with default grid size
    num_bins = 48  # Default number of bins
    total_crop_size = num_bins * bin_size
    
    # Start with center crop
    crop_x = (w - total_crop_size) // 2
    crop_y = (h - total_crop_size) // 2
    
    print("\n=== STEP 3: GRID SIZE AND CROP POSITIONING ===")
    print("Controls:")
    print("  '+' - Increase number of bins")
    print("  '-' - Decrease number of bins")
    print("  'w' - Move crop up")
    print("  's' - Move crop down")
    print("  'a' - Move crop left")
    print("  'd' - Move crop right")
    print("  'c' - Center crop")
    print("  'n' - Next step (keep current settings)")
    print("  'q' - Quit")
    print(f"Bin size: {bin_size}px")
    print(f"Number of bins: {num_bins}x{num_bins}")
    print(f"Total crop size: {total_crop_size}x{total_crop_size}px")
    
    cv2.namedWindow('Grid Size and Crop Positioning', cv2.WINDOW_NORMAL)
    
    while True:
        # Calculate total crop size
        total_crop_size = num_bins * bin_size
        
        # Ensure crop position is valid
        crop_x = max(0, min(crop_x, w - total_crop_size))
        crop_y = max(0, min(crop_y, h - total_crop_size))
        
        # Create display image
        display_img = rotated_img.copy()
        display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
        
        # Draw bin grid overlay within crop area
        crop_x1 = crop_x
        crop_y1 = crop_y
        crop_x2 = crop_x + total_crop_size
        crop_y2 = crop_y + total_crop_size
        
        # Draw the main crop area
        cv2.rectangle(display_img, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 255, 0), 2)
        
        # Draw bin grid within crop area
        for x in range(crop_x1, crop_x2, bin_size):
            cv2.line(display_img, (x, crop_y1), (x, crop_y2), (0, 255, 0), 1)
        for y in range(crop_y1, crop_y2, bin_size):
            cv2.line(display_img, (crop_x1, y), (crop_x2, y), (0, 255, 0), 1)
        
        # Add text overlay
        cv2.putText(display_img, f"Bins: {num_bins}x{num_bins} ({bin_size}px each)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_img, f"Total: {total_crop_size}x{total_crop_size}px", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(display_img, f"Position: ({crop_x}, {crop_y})", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(display_img, "Use +/- for bins, WASD for position", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Grid Size and Crop Positioning', display_img)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            cv2.destroyAllWindows()
            return None
        elif key == ord('+') or key == ord('='):
            num_bins = min(num_bins + 1, 100)
            print(f"Number of bins: {num_bins}x{num_bins}")
        elif key == ord('-'):
            num_bins = max(num_bins - 1, 1)
            print(f"Number of bins: {num_bins}x{num_bins}")
        elif key == ord('w'):  # W key for up
            crop_y = max(crop_y - bin_size, 0)
            print(f"Crop position: ({crop_x}, {crop_y})")
        elif key == ord('s'):  # S key for down
            crop_y = min(crop_y + bin_size, h - total_crop_size)
            print(f"Crop position: ({crop_x}, {crop_y})")
        elif key == ord('a'):  # A key for left
            crop_x = max(crop_x - bin_size, 0)
            print(f"Crop position: ({crop_x}, {crop_y})")
        elif key == ord('d'):  # D key for right
            crop_x = min(crop_x + bin_size, w - total_crop_size)
            print(f"Crop position: ({crop_x}, {crop_y})")
        elif key == ord('c'):
            crop_x = (w - total_crop_size) // 2
            crop_y = (h - total_crop_size) // 2
            print("Crop centered")
        elif key == ord('n'):
            break
    
    cv2.destroyAllWindows()
    print(f"Selected grid: {num_bins}x{num_bins} bins ({bin_size}px each)")
    print(f"Selected crop position: ({crop_x}, {crop_y})")
    return (num_bins, crop_x, crop_y)

def step4_grayscale_inversion_preview(image_path, rotation, bin_size, num_bins, crop_x, crop_y):
    """
    Step 4: Preview grayscale inversion on cropped area.
    
    Args:
        image_path: Path to sample image
        rotation: Selected rotation angle
        bin_size: Selected bin size in pixels
        num_bins: Number of bins in each dimension
        crop_x: Crop X position
        crop_y: Crop Y position
        
    Returns:
        Boolean indicating if inversion should be applied
    """
    # Load and rotate image
    img = cv2.imread(image_path)
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()
    
    rotated_img = rotate_image(img_gray, rotation)
    h, w = rotated_img.shape
    
    invert = False
    total_crop_size = num_bins * bin_size
    
    print("\n=== STEP 4: GRAYSCALE INVERSION ===")
    print("Controls:")
    print("  'i' - Toggle inversion on/off")
    print("  'n' - Next step (keep current setting)")
    print("  'q' - Quit")
    
    cv2.namedWindow('Grayscale Inversion Preview', cv2.WINDOW_NORMAL)
    
    while True:
        # Create display image
        display_img = rotated_img.copy()
        
        # Apply inversion if selected
        if invert:
            display_img = 255 - display_img
        
        # Draw crop area
        crop_x1 = max(0, crop_x)
        crop_y1 = max(0, crop_y)
        crop_x2 = min(w, crop_x + total_crop_size)
        crop_y2 = min(h, crop_y + total_crop_size)
        
        cv2.rectangle(display_img, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 255, 0), 2)
        
        # Draw bin grid within crop area
        for x in range(crop_x1, crop_x2, bin_size):
            cv2.line(display_img, (x, crop_y1), (x, crop_y2), (0, 255, 0), 1)
        for y in range(crop_y1, crop_y2, bin_size):
            cv2.line(display_img, (crop_x1, y), (crop_x2, y), (0, 255, 0), 1)
        
        # Convert to BGR for display
        display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
        
        # Add text overlay
        invert_text = "INVERTED" if invert else "NORMAL"
        color = (0, 0, 255) if invert else (0, 255, 0)
        cv2.putText(display_img, f"Grayscale: {invert_text}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(display_img, f"Grid: {num_bins}x{num_bins} bins ({bin_size}px each)", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(display_img, "Press 'i' to toggle inversion", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Grayscale Inversion Preview', display_img)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            cv2.destroyAllWindows()
            return None
        elif key == ord('i'):
            invert = not invert
            print(f"Inversion: {'ON' if invert else 'OFF'}")
        elif key == ord('n'):
            break
    
    cv2.destroyAllWindows()
    print(f"Grayscale inversion: {'ON' if invert else 'OFF'}")
    return invert

def process_image_series(input_dir, output_dir, rotation, bin_size, num_bins, crop_x, crop_y, invert, file_pattern):
    """
    Process the entire image series with the selected parameters.
    
    Args:
        input_dir: Input directory
        output_dir: Output directory
        rotation: Rotation angle
        bin_size: Bin size in pixels
        num_bins: Number of bins in each dimension
        crop_x: Crop X position
        crop_y: Crop Y position
        invert: Whether to invert grayscale
        file_pattern: File pattern to match
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get list of image files
    image_files = glob.glob(os.path.join(input_dir, file_pattern))
    image_files.sort()
    
    if not image_files:
        print(f"No images found in {input_dir} with pattern {file_pattern}")
        return
    
    total_crop_size = num_bins * bin_size
    
    print(f"\n=== PROCESSING {len(image_files)} IMAGES ===")
    print(f"Parameters:")
    print(f"  Rotation: {rotation}°")
    print(f"  Grid: {num_bins}x{num_bins} bins ({bin_size}px each)")
    print(f"  Total crop size: {total_crop_size}x{total_crop_size}px")
    print(f"  Crop position: ({crop_x}, {crop_y})")
    print(f"  Invert grayscale: {'Yes' if invert else 'No'}")
    
    # Process each image
    for i, image_path in enumerate(image_files):
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not load {image_path}")
            continue
        
        # Convert to grayscale
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img.copy()
        
        # Apply rotation
        if rotation != 0:
            img_gray = rotate_image(img_gray, rotation)
        
        # Apply square crop
        h, w = img_gray.shape
        crop_x1 = max(0, crop_x)
        crop_y1 = max(0, crop_y)
        crop_x2 = min(w, crop_x + total_crop_size)
        crop_y2 = min(h, crop_y + total_crop_size)
        
        # Crop to square area
        img_cropped = img_gray[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Resize to exact size if needed (in case crop was smaller)
        if img_cropped.shape[0] != total_crop_size or img_cropped.shape[1] != total_crop_size:
            img_cropped = cv2.resize(img_cropped, (total_crop_size, total_crop_size))
        
        # Apply grayscale inversion
        if invert:
            img_cropped = 255 - img_cropped
        
        # Save processed image
        filename = f"frame{i:06d}.tif"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, img_cropped)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(image_files)} images")
    
    print(f"Processing complete! Processed images saved to {output_dir}")

def interactive_preprocessing_workflow(input_dir, output_dir, file_pattern):
    """
    Complete interactive preprocessing workflow.
    
    Args:
        input_dir: Input directory
        output_dir: Output directory
        file_pattern: File pattern to match
    """
    # Get sample image
    image_files = glob.glob(os.path.join(input_dir, file_pattern))
    if not image_files:
        print(f"No images found in {input_dir} with pattern {file_pattern}")
        return
    
    sample_image = image_files[0]
    print(f"Using {sample_image} for interactive setup...")
    
    # Step 1: Rotation selection
    rotation = step1_rotation_selection(sample_image)
    if rotation is None:
        print("Workflow cancelled.")
        return
    
    # Step 2: Bin size selection
    bin_size = step2_bin_size_selection(sample_image, rotation)
    if bin_size is None:
        print("Workflow cancelled.")
        return
    
    # Step 3: Grid size and crop positioning
    num_bins, crop_x, crop_y = step3_grid_and_crop_selection(sample_image, rotation, bin_size)
    if num_bins is None:
        print("Workflow cancelled.")
        return
    
    # Step 4: Grayscale inversion
    invert = step4_grayscale_inversion_preview(sample_image, rotation, bin_size, num_bins, crop_x, crop_y)
    if invert is None:
        print("Workflow cancelled.")
        return
    
    # Process all images
    process_image_series(input_dir, output_dir, rotation, bin_size, num_bins, crop_x, crop_y, invert, file_pattern)

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Interactive image series preprocessing for PTV analysis')
    parser.add_argument('--input', default=DEFAULT_IMAGE_DIR, help='Input directory')
    parser.add_argument('--output', default=DEFAULT_OUTPUT_DIR, help='Output directory')
    parser.add_argument('--pattern', default=DEFAULT_PATTERN, help='File pattern to match')
    parser.add_argument('--interactive', action='store_true', default=True, help='Interactive mode (default)')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_preprocessing_workflow(args.input, args.output, args.pattern)
    else:
        print("Non-interactive mode not implemented. Use --interactive flag.")

if __name__ == "__main__":
    main()