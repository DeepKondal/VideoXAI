import numpy as np
import argparse
import cv2
import os

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, required=True, help='Path to the input video file (mp4)')
args = parser.parse_args()

# Convert Video to .npy
def convert_video_to_npy(video_path):
    save_name = os.path.basename(os.path.splitext(video_path)[0]) + ".npy"
    save_path = os.path.join("videos", save_name)
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(64):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    video_array = np.array(frames, dtype=np.float32) / 255.0  # Normalize
    np.save(save_path, video_array)
    print(f"Saved video to {save_path}")

convert_video_to_npy(args.video)
