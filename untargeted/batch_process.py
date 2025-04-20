import os
import argparse
import subprocess
import torch

# Argument Parsing
parser = argparse.ArgumentParser(description="Batch process videos using VBAD attack pipeline")
parser.add_argument("--video_list", type=str, required=True, help="Path to the video list file")
parser.add_argument("--sigma", type=float, default=1e-3, help="Sigma value for VBAD attack (default: 1e-3)")
parser.add_argument("--untargeted", action="store_true", help="Enable untargeted attack mode")
args = parser.parse_args()

# Directories
RAW_VIDEOS_DIR = "/Users/ravindersingh/Desktop/strangler/XAIport/dataprocess/raw_videos"
VIDEOS_DIR = "videos"
OUTPUT_TARGETED_DIR = "output_videos_targeted"
OUTPUT_UNTARGETED_DIR = "output_videos_untargeted"
FINAL_TARGETED_DIR = "final_perturbed_videos/targeted"
FINAL_UNTARGETED_DIR = "final_perturbed_videos/untargeted"

# Ensure necessary directories exist
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(OUTPUT_TARGETED_DIR, exist_ok=True)
os.makedirs(OUTPUT_UNTARGETED_DIR, exist_ok=True)
os.makedirs(FINAL_TARGETED_DIR, exist_ok=True)
os.makedirs(FINAL_UNTARGETED_DIR, exist_ok=True)

# Read video_list.txt
with open(args.video_list, "r") as f:
    video_entries = [line.strip().split() for line in f.readlines()]

# Step 1: Convert videos to .npy format
for video_name, label in video_entries:
    video_path = os.path.join(RAW_VIDEOS_DIR, video_name)
    npy_path = os.path.join(VIDEOS_DIR, f"{os.path.splitext(video_name)[0]}.npy")

    if not os.path.exists(npy_path):  # Avoid redundant conversion
        print(f"🎥 Converting {video_name} → {npy_path}")
        subprocess.run(["python", "untargeted/convert_mp4_to_npy.py", "--video", video_path])
        torch.cuda.empty_cache()
    else:
        print(f"✅ {npy_path} already exists, skipping conversion.")

# Step 2: Run VBAD attacks
for video_name, label in video_entries:
    npy_path = os.path.join(VIDEOS_DIR, f"{os.path.splitext(video_name)[0]}.npy")

    # Define separate output paths for targeted and untargeted attacks
    output_targeted_npy = os.path.join(OUTPUT_TARGETED_DIR, f"output_{label}.npy")
    output_untargeted_npy = os.path.join(OUTPUT_UNTARGETED_DIR, f"output_{label}.npy")

    print(f"📌 Processing {video_name} → {npy_path}")

    # Run Targeted Attack
    if not args.untargeted:
        if not os.path.exists(output_targeted_npy):  # Avoid redundant attacks
            print(f"⚡ Running **targeted attack** on {npy_path} → {output_targeted_npy}")
            attack_command = [
                "python", "untargeted/main.py",
                "--gpus", "0",
                "--video", npy_path,
                "--label", label,
                "--adv-save-path", output_targeted_npy,
                "--sigma", str(args.sigma),
            ]
            subprocess.run(attack_command, check=True)
            torch.cuda.empty_cache()
        else:
            print(f"✅ {output_targeted_npy} already exists, skipping attack.")

    # Run Untargeted Attack
    if args.untargeted:
        if not os.path.exists(output_untargeted_npy):  # Avoid redundant attacks
            print(f"⚡ Running **untargeted attack** on {npy_path} → {output_untargeted_npy}")
            attack_command = [
                "python", "untargeted/main.py",
                "--gpus", "0",
                "--video", npy_path,
                "--label", label,
                "--adv-save-path", output_untargeted_npy,
                "--sigma", str(args.sigma),
                "--untargeted"
            ]
            subprocess.run(attack_command, check=True)
            torch.cuda.empty_cache()
        else:
            print(f"✅ {output_untargeted_npy} already exists, skipping attack.")

# Step 3: Convert adversarial .npy videos to .mp4 format
for video_name, label in video_entries:
    output_targeted_npy = os.path.join(OUTPUT_TARGETED_DIR, f"output_{label}.npy")
    output_untargeted_npy = os.path.join(OUTPUT_UNTARGETED_DIR, f"output_{label}.npy")

    final_targeted_video = os.path.join(FINAL_TARGETED_DIR, f"adv_{label}.mp4")
    final_untargeted_video = os.path.join(FINAL_UNTARGETED_DIR, f"adv_{label}.mp4")

    # Convert Targeted Videos
    if not args.untargeted and os.path.exists(output_targeted_npy) and not os.path.exists(final_targeted_video):
        print(f"🎬 Converting {output_targeted_npy} → {final_targeted_video}")
        subprocess.run(["python", "convert_npy_to_mp4.py", "--npy", output_targeted_npy, "--output", final_targeted_video, "--fps", "30"])
        torch.cuda.empty_cache()

    # Convert Untargeted Videos
    if args.untargeted and os.path.exists(output_untargeted_npy) and not os.path.exists(final_untargeted_video):
        print(f"🎬 Converting {output_untargeted_npy} → {final_untargeted_video}")
        subprocess.run(["python", "convert_npy_to_mp4.py", "--npy", output_untargeted_npy, "--output", final_untargeted_video, "--fps", "30"])
        torch.cuda.empty_cache()

print("🎉 Batch processing completed!")
