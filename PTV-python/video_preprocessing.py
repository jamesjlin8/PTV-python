#!/usr/bin/env python3
"""
Video preprocessing utilities for PTV analysis.
This module provides functions for extracting frames from videos and preprocessing them.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import os

# Default settings
DEFAULT_VIDEO = "./input_video.mp4"
DEFAULT_OUTPUT_DIR = "./extracted_frames"
DEFAULT_PREVIEW_FRAME = 100
DEFAULT_FRAME_RATE = 30
DEFAULT_QUALITY = 95

def select_roi(video_path: str = DEFAULT_VIDEO, frame_number: int = DEFAULT_PREVIEW_FRAME) -> Optional[tuple]:
    """
    Select region of interest (ROI) from a video frame interactively.
    
    Args:
        video_path: Path to the video file
        frame_number: Frame number to use for ROI selection
        
    Returns:
        Tuple of (x, y, width, height) for the selected ROI, or None if cancelled
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read frame
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame {frame_number}")
        cap.release()
        return None
    
    cap.release()
    
    # Global variables for ROI selection
    roi_selected = False
    roi_coords = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal roi_selected, roi_coords
        
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_coords = [x, y, 0, 0]
            roi_selected = False
            
        elif event == cv2.EVENT_MOUSEMOVE and roi_coords is not None:
            roi_coords[2] = x - roi_coords[0]
            roi_coords[3] = y - roi_coords[1]
            
        elif event == cv2.EVENT_LBUTTONUP:
            roi_selected = True
    
    # Create window and set mouse callback
    cv2.namedWindow('Select ROI', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Select ROI', mouse_callback)
    
    print("Instructions:")
    print("1. Click and drag to select ROI")
    print("2. Press 'q' to quit without selection")
    print("3. Press 'Enter' to confirm selection")
    
    while True:
        # Create display frame
        display_frame = frame.copy()
        
        # Draw current ROI
        if roi_coords is not None:
            x, y, w, h = roi_coords
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow('Select ROI', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            return None
        elif key == 13 and roi_selected:  # Enter key
            break
    
    cv2.destroyAllWindows()
    
    if roi_coords is not None:
        x, y, w, h = roi_coords
        # Ensure positive dimensions
        if w < 0:
            x += w
            w = -w
        if h < 0:
            y += h
            h = -h
        return (x, y, w, h)
    
    return None

def video_to_images(
    video_path: str = DEFAULT_VIDEO,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    roi: Optional[tuple] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    frame_skip: int = 1,
    image_format: str = "tif",
    quality: int = DEFAULT_QUALITY
) -> None:
    """
    Extract frames from video and save as images.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save extracted frames
        roi: Region of interest (x, y, width, height) to crop frames
        start_frame: Starting frame number
        end_frame: Ending frame number (None for all frames)
        frame_skip: Number of frames to skip between extractions
        image_format: Output image format ('tif', 'png', 'jpg')
        quality: JPEG quality (1-100, only for JPEG format)
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video properties:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")
    
    # Set end frame if not specified
    if end_frame is None:
        end_frame = total_frames
    
    # Set starting position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = start_frame
    saved_count = 0
    
    print(f"Extracting frames {start_frame} to {end_frame} (skip: {frame_skip})...")
    
    while frame_count < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply ROI if specified
        if roi is not None:
            x, y, w, h = roi
            frame = frame[y:y+h, x:x+w]
        
        # Save frame if it's time to save
        if (frame_count - start_frame) % frame_skip == 0:
            # Generate filename
            filename = f"frame{saved_count:06d}.{image_format}"
            output_path = os.path.join(output_dir, filename)
            
            # Save image
            if image_format.lower() == 'jpg':
                cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                cv2.imwrite(output_path, frame)
            
            saved_count += 1
            
            if saved_count % 100 == 0:
                print(f"Saved {saved_count} frames...")
        
        frame_count += 1
    
    cap.release()
    print(f"Extraction complete! Saved {saved_count} frames to {output_dir}")

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract frames from video for PTV analysis')
    parser.add_argument('--video', default=DEFAULT_VIDEO, help='Input video file')
    parser.add_argument('--output', default=DEFAULT_OUTPUT_DIR, help='Output directory')
    parser.add_argument('--roi', nargs=4, type=int, help='ROI coordinates (x y width height)')
    parser.add_argument('--start', type=int, default=0, help='Starting frame number')
    parser.add_argument('--end', type=int, help='Ending frame number')
    parser.add_argument('--skip', type=int, default=1, help='Frame skip interval')
    parser.add_argument('--format', default='tif', choices=['tif', 'png', 'jpg'], help='Output image format')
    parser.add_argument('--quality', type=int, default=DEFAULT_QUALITY, help='JPEG quality (1-100)')
    parser.add_argument('--select-roi', action='store_true', help='Interactively select ROI')
    
    args = parser.parse_args()
    
    # Select ROI if requested
    roi = None
    if args.select_roi:
        roi = select_roi(args.video)
        if roi is None:
            print("ROI selection cancelled")
            return
        print(f"Selected ROI: {roi}")
    elif args.roi:
        roi = tuple(args.roi)
    
    # Extract frames
    video_to_images(
        video_path=args.video,
        output_dir=args.output,
        roi=roi,
        start_frame=args.start,
        end_frame=args.end,
        frame_skip=args.skip,
        image_format=args.format,
        quality=args.quality
    )

if __name__ == "__main__":
    main()
