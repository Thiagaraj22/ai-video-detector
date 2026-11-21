import cv2
import os
from tqdm import tqdm

def extract_frames(video_path, output_folder, fps=1):
    """Extract 1 frame per second from a video"""
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Could not open: {video_path}")
        return 0
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        cap.release()
        return 0
        
    interval = max(1, int(video_fps / fps))
    
    count = 0
    saved = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved += 1
        count += 1
    
    cap.release()
    return saved


def process_all_videos(input_folder, output_folder, label, split_ratio=0.8):
    """Process all videos and split into train/val"""
    videos = [f for f in os.listdir(input_folder) 
              if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if len(videos) == 0:
        print(f"No videos found in {input_folder}")
        print("Make sure your videos end in .mp4, .avi, .mov or .mkv")
        return
    
    split_point = int(len(videos) * split_ratio)
    train_videos = videos[:split_point]
    val_videos   = videos[split_point:]
    
    print(f"\nProcessing {label} videos...")
    print(f"  Train: {len(train_videos)} videos")
    print(f"  Val:   {len(val_videos)} videos")
    
    total_frames = 0
    
    # Process train videos
    for video_name in tqdm(train_videos, desc=f"Train {label}"):
        video_path   = os.path.join(input_folder, video_name)
        video_stem   = os.path.splitext(video_name)[0]
        out_folder   = os.path.join(output_folder, "train", label, video_stem)
        saved = extract_frames(video_path, out_folder, fps=1)
        total_frames += saved
    
    # Process val videos
    for video_name in tqdm(val_videos, desc=f"Val {label}"):
        video_path   = os.path.join(input_folder, video_name)
        video_stem   = os.path.splitext(video_name)[0]
        out_folder   = os.path.join(output_folder, "val", label, video_stem)
        saved = extract_frames(video_path, out_folder, fps=5)
        total_frames += saved
    
    print(f"Total frames saved for {label}: {total_frames}")


# ── Run it ────────────────────────────────────────────────────────────
process_all_videos("dataset/real", "dataset/frames", label="real")
process_all_videos("dataset/fake", "dataset/frames", label="fake")

print("\nDone! Your frames are ready in dataset/frames/")