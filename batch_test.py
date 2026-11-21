# batch_test.py
import os
import torch
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score
)

# ── Settings ──────────────────────────────────────────────────────
TEST_FOLDER   = "test_videos"
TRUE_LABEL    = "AI-Generated"   # all videos in this folder are AI-generated
SAVE_DIR      = "cross_generator_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Map filename keywords to generator names ───────────────────────
# Edit these to match your actual filenames
GENERATOR_MAP = {
    "sora"   : "Sora (OpenAI)",
    "veo"    : "Veo (Google)",
    "kling"  : "Kling",
    "hailuo" : "Hailuo (MiniMax)",
    "pika"   : "Pika",
    "runway" : "Runway",
}

def get_generator(filename):
    fname_lower = filename.lower()
    for keyword, name in GENERATOR_MAP.items():
        if keyword in fname_lower:
            return name
    return "Unknown"

# ── Load DINOv2 ───────────────────────────────────────────────────
print("Loading DINOv2...")
dino = torch.hub.load('facebookresearch/dinov2',
                      'dinov2_vits14', pretrained=True)
dino.eval()
print("DINOv2 ready")

# ── Load trained classifier ───────────────────────────────────────
with open("restrav_model.pkl", "rb") as f:
    payload = pickle.load(f)
classifier = payload["model"]
scaler     = payload["scaler"]
print(f"Classifier loaded: {payload.get('name', 'unknown')}\n")

# ── Transforms ────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Extract DINOv2 trajectory features from one video ─────────────
def extract_and_predict(video_path):
    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    step   = max(1, int(fps))
    frames = []
    count  = 0
    while cap.isOpened() and len(frames) < 60:
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        count += 1
    cap.release()

    if len(frames) < 3:
        return None, None, None

    # DINOv2 feature per frame
    feats = []
    for f in frames:
        t = transform(Image.fromarray(f)).unsqueeze(0)
        with torch.no_grad():
            feats.append(dino(t).squeeze().cpu())

    feat_tensor = torch.stack(feats)
    diffs       = feat_tensor[1:] - feat_tensor[:-1]

    # Curvature between consecutive step vectors
    curvatures = []
    for i in range(len(diffs) - 1):
        cos = torch.nn.functional.cosine_similarity(
            diffs[i].unsqueeze(0), diffs[i+1].unsqueeze(0)).item()
        curvatures.append(1.0 - cos)

    # Step distances
    distances = [torch.norm(diffs[i]).item() for i in range(len(diffs))]

    c = np.array(curvatures)
    d = np.array(distances)

    fv = np.array([[
        c.mean(), c.std(), c.max(), c.min(),
        np.percentile(c, 25), np.percentile(c, 75), np.percentile(c, 90),
        d.mean(), d.std(), d.max(), d.min(),
        np.percentile(d, 25), np.percentile(d, 75),
        float(len(frames))
    ]])

    fv_scaled = scaler.transform(fv)
    prob_fake = classifier.predict_proba(fv_scaled)[0][1]
    verdict   = "AI-Generated" if prob_fake > 0.5 else "Real"

    return verdict, prob_fake, fv[0]


# ══════════════════════════════════════════════════════════════════
# RUN BATCH TEST
# ══════════════════════════════════════════════════════════════════
if not os.path.exists(TEST_FOLDER):
    print(f"ERROR: '{TEST_FOLDER}' folder not found.")
    print("Create it and add your AI-generated test videos.")
    exit()

video_files = sorted([
    f for f in os.listdir(TEST_FOLDER)
    if f.lower().endswith(('.mp4', '.avi', '.mov'))
])

if len(video_files) == 0:
    print(f"No videos found in '{TEST_FOLDER}'")
    exit()

print(f"Found {len(video_files)} videos to test\n")
print("-" * 75)

results    = []
all_probs  = []
all_labels = []   # 1 = AI-Generated (all are true positives here)
skipped    = 0

for fname in video_files:
    path      = os.path.join(TEST_FOLDER, fname)
    generator = get_generator(fname)
    verdict, prob, feats = extract_and_predict(path)

    if verdict is None:
        print(f"SKIP  {fname[:55]} (too short)")
        skipped += 1
        continue

    is_correct = (verdict == TRUE_LABEL)
    status     = "✅" if is_correct else "❌"

    print(f"{status} [{generator:<20}] {fname[:40]:<40} → {verdict} ({prob*100:.1f}%)")

    results.append({
        "file"      : fname,
        "generator" : generator,
        "verdict"   : verdict,
        "prob_fake" : prob,
        "correct"   : is_correct,
    })
    all_probs.append(prob)
    all_labels.append(1)   # ground truth: all are AI-generated

print("-" * 75)

# ══════════════════════════════════════════════════════════════════
# OVERALL METRICS
# ══════════════════════════════════════════════════════════════════
total   = len(results)
correct = sum(r["correct"] for r in results)
preds   = [1 if r["prob_fake"] > 0.5 else 0 for r in results]

print(f"\n=== Overall Cross-Generator Results ===")
print(f"Videos tested : {total}  (skipped {skipped})")
print(f"Correct       : {correct}")
print(f"Accuracy      : {correct/total*100:.1f}%")

# AUC-ROC — only meaningful if we also have real videos for comparison
# Here we compute a simpler metric: mean confidence on AI videos
mean_conf = np.mean(all_probs) * 100
print(f"Mean AI confidence : {mean_conf:.1f}%")
print(f"(AUC-ROC requires both real and fake samples — "
      f"add real videos to test folder for full ROC curve)")

# ── Per-generator breakdown ────────────────────────────────────────
print(f"\n=== Per-Generator Breakdown ===")
generators = sorted(set(r["generator"] for r in results))
gen_data   = {}

for gen in generators:
    gen_results = [r for r in results if r["generator"] == gen]
    n           = len(gen_results)
    n_correct   = sum(r["correct"] for r in gen_results)
    mean_prob   = np.mean([r["prob_fake"] for r in gen_results]) * 100
    gen_data[gen] = {"n": n, "correct": n_correct, "mean_prob": mean_prob}
    print(f"  {gen:<22} {n_correct}/{n} correct  "
          f"({n_correct/n*100:.0f}%)  mean confidence: {mean_prob:.1f}%")

# ══════════════════════════════════════════════════════════════════
# FIGURE 1 — CONFUSION MATRIX (needs both real and fake)
# ══════════════════════════════════════════════════════════════════
# For cross-generator test (AI only), we show a detection summary instead
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: per-generator accuracy bar chart
gen_names  = list(gen_data.keys())
gen_accs   = [gen_data[g]["correct"] / gen_data[g]["n"] * 100
              for g in gen_names]
colors     = ["#3FA34D" if a >= 70 else "#F2A93B" if a >= 50 else "#E2484A"
              for a in gen_accs]

bars = axes[0].bar(gen_names, gen_accs, color=colors, edgecolor="white",
                   linewidth=1.5, zorder=2)
axes[0].axhline(50, color="grey",    linestyle="--", linewidth=1,
                label="50% (random)", zorder=1)
axes[0].axhline(70, color="#2E75B6", linestyle="--", linewidth=1,
                label="70% (target)", zorder=1)
axes[0].set_ylim(0, 110)
axes[0].set_ylabel("Detection Accuracy (%)", fontsize=12)
axes[0].set_title("Cross-Generator Detection Accuracy", fontsize=13, fontweight="bold")
axes[0].set_xticks(range(len(gen_names)))
axes[0].set_xticklabels(gen_names, rotation=20, ha="right", fontsize=10)
axes[0].legend(fontsize=9)
axes[0].grid(axis="y", alpha=0.3, zorder=0)

for bar, acc in zip(bars, gen_accs):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 2,
                 f"{acc:.0f}%", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")

# Right: per-video confidence scatter coloured by correct/wrong
indices    = range(len(results))
conf_vals  = [r["prob_fake"] * 100 for r in results]
gen_labels = [r["generator"] for r in results]
correct_arr= [r["correct"]   for r in results]

scatter_colors = ["#3FA34D" if c else "#E2484A" for c in correct_arr]
axes[1].scatter(indices, conf_vals, c=scatter_colors, s=60,
                alpha=0.8, edgecolors="white", linewidth=0.5, zorder=2)
axes[1].axhline(50, color="grey", linestyle="--", linewidth=1,
                label="Decision threshold (50%)", zorder=1)
axes[1].set_ylim(-5, 110)
axes[1].set_xlabel("Video index", fontsize=11)
axes[1].set_ylabel("AI-Generated Confidence (%)", fontsize=11)
axes[1].set_title("Per-Video Confidence Score", fontsize=13, fontweight="bold")
axes[1].legend(handles=[
    plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#3FA34D",
               markersize=8, label="Correct (AI-Generated detected)"),
    plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#E2484A",
               markersize=8, label="Wrong (classified as Real)"),
    plt.Line2D([0],[0], color="grey", linestyle="--",
               label="Decision threshold (50%)"),
], fontsize=9)
axes[1].grid(alpha=0.3, zorder=0)

plt.tight_layout()
out1 = os.path.join(SAVE_DIR, "cross_generator_accuracy.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nSaved: {out1}")

# ══════════════════════════════════════════════════════════════════
# FIGURE 2 — CONFUSION MATRIX + ROC CURVE
# (adds real video support if real videos are present in test folder)
# ══════════════════════════════════════════════════════════════════
real_folder = os.path.join(TEST_FOLDER, "real")
has_real    = os.path.exists(real_folder) and \
              len([f for f in os.listdir(real_folder)
                   if f.lower().endswith(('.mp4','.avi','.mov'))]) > 0

if has_real:
    print(f"\nFound real videos in {real_folder} — computing full ROC & confusion matrix")

    real_files  = [f for f in os.listdir(real_folder)
                   if f.lower().endswith(('.mp4','.avi','.mov'))]
    real_probs  = []
    real_labels = []

    for fname in sorted(real_files):
        path            = os.path.join(real_folder, fname)
        verdict, prob, _= extract_and_predict(path)
        if verdict is None:
            continue
        real_probs.append(prob)
        real_labels.append(0)
        flag = "✅" if verdict == "Real" else "❌"
        print(f"  {flag} [Real] {fname[:50]} → {verdict} ({prob*100:.1f}%)")

    # Combine fake + real
    combined_probs  = all_probs  + real_probs
    combined_labels = all_labels + real_labels
    combined_preds  = [1 if p > 0.5 else 0 for p in combined_probs]

    auc = roc_auc_score(combined_labels, combined_probs)
    print(f"\nAUC-ROC (with real videos): {auc:.4f}")
    print(classification_report(combined_labels, combined_preds,
                                target_names=["Real", "AI-Generated"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Confusion matrix
    cm = confusion_matrix(combined_labels, combined_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "AI-Generated"],
                yticklabels=["Real", "AI-Generated"],
                ax=axes[0], linewidths=0.5)
    axes[0].set_title("Confusion Matrix\n(Cross-Generator Test)",
                       fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Actual Label", fontsize=11)
    axes[0].set_xlabel("Predicted Label", fontsize=11)

    # ROC curve
    fpr, tpr, _ = roc_curve(combined_labels, combined_probs)
    axes[1].plot(fpr, tpr, color="#2E75B6", linewidth=2.5,
                 label=f"ROC Curve (AUC = {auc:.4f})")
    axes[1].fill_between(fpr, tpr, alpha=0.08, color="#2E75B6")
    axes[1].plot([0,1],[0,1], "k--", linewidth=1, label="Random baseline")
    axes[1].set_xlabel("False Positive Rate", fontsize=11)
    axes[1].set_ylabel("True Positive Rate", fontsize=11)
    axes[1].set_title("ROC Curve\n(Cross-Generator Test)",
                       fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out2 = os.path.join(SAVE_DIR, "cross_generator_roc_confusion.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out2}")

else:
    print(f"\nNOTE: To generate a full ROC curve and confusion matrix,")
    print(f"create a subfolder:  test_videos/real/")
    print(f"and add 10-20 real camera videos to it.")
    print(f"Then re-run this script.")

# ══════════════════════════════════════════════════════════════════
# FIGURE 3 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════
feature_names = [
    "curv_mean",  "curv_std",  "curv_max",  "curv_min",
    "curv_p25",   "curv_p75",  "curv_p90",
    "dist_mean",  "dist_std",  "dist_max",  "dist_min",
    "dist_p25",   "dist_p75",  "frame_count"
]

clf_name = payload.get("name", "")

if hasattr(classifier, "feature_importances_"):
    importance = classifier.feature_importances_
    idx        = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 5))
    bar_colors = ["#2E75B6" if i < 7 else "#F2A93B"
                  for i in idx]
    bars = plt.bar(range(len(feature_names)),
                   importance[idx],
                   color=bar_colors,
                   edgecolor="white", linewidth=1.2)
    plt.xticks(range(len(feature_names)),
               [feature_names[i] for i in idx],
               rotation=45, ha="right", fontsize=10)
    plt.ylabel("Feature Importance (Gini)", fontsize=12)
    plt.title("Feature Importance — ReStraV DINOv2 Trajectory Classifier\n"
              "(Blue = Curvature features, Orange = Distance features)",
              fontsize=12, fontweight="bold")
    plt.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    plt.legend(handles=[
        Patch(color="#2E75B6", label="Curvature features"),
        Patch(color="#F2A93B", label="Distance features"),
    ], fontsize=10)

    plt.tight_layout()
    out3 = os.path.join(SAVE_DIR, "feature_importance.png")
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out3}")

elif hasattr(classifier, "coef_"):
    # Logistic regression — show coefficients instead
    coef   = np.abs(classifier.coef_[0])
    idx    = np.argsort(coef)[::-1]

    plt.figure(figsize=(10, 5))
    bar_colors = ["#2E75B6" if i < 7 else "#F2A93B" for i in idx]
    plt.bar(range(len(feature_names)), coef[idx],
            color=bar_colors, edgecolor="white", linewidth=1.2)
    plt.xticks(range(len(feature_names)),
               [feature_names[i] for i in idx],
               rotation=45, ha="right", fontsize=10)
    plt.ylabel("|Coefficient| (Logistic Regression)", fontsize=12)
    plt.title("Feature Coefficients — ReStraV DINOv2 Trajectory Classifier\n"
              "(Blue = Curvature features, Orange = Distance features)",
              fontsize=12, fontweight="bold")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out3 = os.path.join(SAVE_DIR, "feature_importance.png")
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out3}")

print(f"\n=== Done! All charts saved to '{SAVE_DIR}/' ===")
print(f"  cross_generator_accuracy.png")
print(f"  cross_generator_roc_confusion.png  (if real/ subfolder exists)")
print(f"  feature_importance.png")