# extract_dino_features.py
import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ── Settings ──────────────────────────────────────────────────────
DATA_DIR    = "dataset/frames"
SAVE_DIR    = "dataset/dino_features"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Load DINOv2 ───────────────────────────────────────────────────
print("Loading DINOv2...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
dino = dino.to(device)
dino.eval()
print(f"DINOv2 ready on {device}")

# ── Image transform (DINOv2 uses 224x224) ─────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ── Compute trajectory features for one video ─────────────────────
def compute_trajectory_features(frame_folder):
    """
    Given a folder of frames from one video,
    return a fixed-size feature vector describing
    how the video moves through DINOv2 feature space.
    """
    frame_files = sorted([
        f for f in os.listdir(frame_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if len(frame_files) < 3:
        return None  # not enough frames for trajectory

    # Get DINOv2 features for each frame
    features = []
    for fname in frame_files:
        img_path = os.path.join(frame_folder, fname)
        try:
            img    = Image.open(img_path).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = dino(tensor)         # (1, 384)
            features.append(feat.squeeze().cpu())
        except Exception:
            continue

    if len(features) < 3:
        return None

    features = torch.stack(features)        # (N, 384)

    # ── Step vectors (how much the video moves each step) ─────────
    diffs = features[1:] - features[:-1]    # (N-1, 384)

    # ── Curvature (angle between consecutive steps) ────────────────
    curvatures = []
    for i in range(len(diffs) - 1):
        a = diffs[i]
        b = diffs[i + 1]
        cos_sim = torch.nn.functional.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0)
        ).item()
        # curvature = 1 - cosine similarity (0 = straight, 2 = reversed)
        curvatures.append(1.0 - cos_sim)

    # ── Step distances (how far each step travels) ─────────────────
    distances = [torch.norm(diffs[i]).item() for i in range(len(diffs))]

    curvatures = np.array(curvatures)
    distances  = np.array(distances)

    # ── Aggregate into statistics ──────────────────────────────────
    feature_vector = np.array([
        # Curvature statistics
        curvatures.mean(),
        curvatures.std(),
        curvatures.max(),
        curvatures.min(),
        np.percentile(curvatures, 25),
        np.percentile(curvatures, 75),
        np.percentile(curvatures, 90),

        # Distance statistics
        distances.mean(),
        distances.std(),
        distances.max(),
        distances.min(),
        np.percentile(distances, 25),
        np.percentile(distances, 75),

        # Frame count (captures video length info)
        float(len(features)),
    ])

    return feature_vector       # shape: (14,)


# ── Process all splits and labels ─────────────────────────────────
def process_split(split_name):
    all_features = []
    all_labels   = []
    skipped      = 0

    for label_name in ['real', 'fake']:
        label     = 0 if label_name == 'real' else 1
        label_dir = os.path.join(DATA_DIR, split_name, label_name)

        if not os.path.exists(label_dir):
            print(f"Skipping {label_dir} (not found)")
            continue

        video_folders = [
            d for d in os.listdir(label_dir)
            if os.path.isdir(os.path.join(label_dir, d))
        ]

        print(f"\n{split_name} / {label_name}: {len(video_folders)} videos")

        for vid_folder in tqdm(video_folders, desc=f"{split_name}/{label_name}"):
            full_path = os.path.join(label_dir, vid_folder)
            feats     = compute_trajectory_features(full_path)

            if feats is not None:
                all_features.append(feats)
                all_labels.append(label)
            else:
                skipped += 1

    if skipped > 0:
        print(f"Skipped {skipped} videos (too few frames)")

    X = np.array(all_features)   # (N, 14)
    y = np.array(all_labels)     # (N,)

    # Save
    save_path = os.path.join(SAVE_DIR, f"{split_name}_features.npz")
    np.savez(save_path, X=X, y=y)
    print(f"\nSaved {split_name}: {X.shape} features, {y.sum()} fake / {(y==0).sum()} real")
    return X, y


print("\n=== Extracting TRAIN features ===")
X_train, y_train = process_split("train")

print("\n=== Extracting VAL features ===")
X_val, y_val = process_split("val")

print("\n=== All done! ===")
print(f"Train: {X_train.shape}, Val: {X_val.shape}")
print("Features saved to dataset/dino_features/")