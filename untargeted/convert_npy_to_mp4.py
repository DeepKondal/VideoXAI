import numpy as np
import argparse
import cv2
import os

# Parse Arguments
parser = argparse.ArgumentParser(description="Convert .npy video to .mp4")
parser.add_argument('--npy', type=str, required=True, help='Path to the input .npy file')
parser.add_argument('--output', type=str, required=True, help='Path to save the output .mp4 file')
parser.add_argument('--fps', type=int, default=30, help='Frames per second for output video')
args = parser.parse_args()


def convert_npy_to_video(npy_path, output_path, fps=30):
    """
    Convert an .npy file containing video frames into an .mp4 file.
    
    Args:
        npy_path (str): Path to the .npy file
        output_path (str): Path to save the output .mp4 video
        fps (int): Frames per second for the output video
    """
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"❌ Error: .npy file not found: {npy_path}")

    print(f"📂 Loading .npy file: {npy_path}")
    video_array = np.load(npy_path, mmap_mode='r')  # Use memory mapping to reduce memory usage

    # **Step 1: Ensure Correct Shape (Frames, H, W, C)**
    if len(video_array.shape) == 4:
        if video_array.shape[1] == 3 and video_array.shape[-1] != 3:
            print("🔄 Fixing shape: (Frames, C, H, W) → (Frames, H, W, C)")
            video_array = np.transpose(video_array, (0, 2, 3, 1))  # Convert to (Frames, H, W, C)

    # **Step 2: Validate Shape**
    if len(video_array.shape) != 4 or video_array.shape[-1] != 3:
        raise ValueError(f"❌ Invalid .npy shape: {video_array.shape}. Expected (Frames, Height, Width, 3).")

    num_frames, height, width, _ = video_array.shape
    print(f"📏 Video Shape: {num_frames} frames, {height}x{width} resolution")

    # **Step 3: Initialize OpenCV Video Writer**
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # **Step 4: Convert Frames (Ensure uint8 Format)**
    for i, frame in enumerate(video_array):
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)  # Normalize & convert to uint8
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert RGB → BGR for OpenCV
        if i % 10 == 0:  # Log progress every 10 frames
            print(f"🎥 Processing frame {i+1}/{num_frames}")

    # **Step 5: Release Resources**
    out.release()
    print(f"✅ Video saved successfully: {output_path}")


# Run the conversion
convert_npy_to_video(args.npy, args.output, args.fps)
