# app.py — AI Video Detector (ReStraV / DINOv2 only)
import streamlit as st
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import tempfile
import os
import pickle
from torchvision import transforms

st.set_page_config(
    page_title="AI Video Detector",
    page_icon="🎬",
    layout="centered"
)
st.title("🎬 AI Video Detector")
st.caption("Temporal trajectory analysis using DINOv2 — ReStraV method")

# ── Load DINOv2 + classifier (cached — only loads once) ───────────
@st.cache_resource
def load_models():
    with st.spinner("Loading DINOv2 model..."):
        dino = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vitb14',
            pretrained=True
        )
        dino.eval()

    with open("restrav_model.pkl", "rb") as f:
        payload = pickle.load(f)

    return dino, payload["model"], payload["scaler"]

dino, classifier, scaler = load_models()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Frame extraction ──────────────────────────────────────────────
def extract_frames(video_path, max_frames=60):
    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    step   = max(1, int(fps))   # 1 frame per second
    frames = []
    count  = 0
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        count += 1
    cap.release()
    return frames

# ── DINOv2 trajectory analysis ────────────────────────────────────
def analyse_trajectory(frames):
    """
    Extract DINOv2 features for each frame, then compute
    curvature and distance statistics across the trajectory.
    Returns the feature vector, curvature list, and distance list.
    """
    # Get DINOv2 feature vector for each frame
    feat_list = []
    for frame in frames:
        tensor = transform(frame).unsqueeze(0)
        with torch.no_grad():
            feat = dino(tensor)         # (1, 384)
        feat_list.append(feat.squeeze().cpu())

    feat_tensor = torch.stack(feat_list)   # (N, 384)

    # Step vectors between consecutive frames
    diffs = feat_tensor[1:] - feat_tensor[:-1]   # (N-1, 384)

    # Curvature: angle between consecutive step vectors
    curvatures = []
    for i in range(len(diffs) - 1):
        cos_sim = F.cosine_similarity(
            diffs[i].unsqueeze(0),
            diffs[i+1].unsqueeze(0)
        ).item()
        curvatures.append(1.0 - cos_sim)  # 0=straight, 2=reversed

    # Distance: how far each step travels in feature space
    distances = [torch.norm(diffs[i]).item() for i in range(len(diffs))]

    curvatures = np.array(curvatures)
    distances  = np.array(distances)

    # Build feature vector (14 statistics)
    feature_vec = np.array([[
        curvatures.mean(),
        curvatures.std(),
        curvatures.max(),
        curvatures.min(),
        np.percentile(curvatures, 25),
        np.percentile(curvatures, 75),
        np.percentile(curvatures, 90),
        distances.mean(),
        distances.std(),
        distances.max(),
        distances.min(),
        np.percentile(distances, 25),
        np.percentile(distances, 75),
    ]])

    return feature_vec, curvatures, distances

# ── Run classifier on feature vector ─────────────────────────────
def classify(feature_vec):
    scaled     = scaler.transform(feature_vec)
    prob_fake  = classifier.predict_proba(scaled)[0][1]
    prob_real  = 1.0 - prob_fake
    verdict    = "AI-Generated" if prob_fake > 0.5 else "Real"
    confidence = prob_fake if verdict == "AI-Generated" else prob_real
    return verdict, confidence, prob_fake, prob_real

# ── UI ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a video file",
    type=["mp4", "avi", "mov"],
    help="Maximum file size: 200MB — the system analyses motion patterns across frames"
)

if uploaded:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("Extracting frames..."):
        frames = extract_frames(tmp_path)

    os.unlink(tmp_path)

    if len(frames) < 3:
        st.error("Video too short — need at least 3 seconds for trajectory analysis.")
        st.stop()

    with st.spinner(f"Analysing trajectory across {len(frames)} frames with DINOv2..."):
        feature_vec, curvatures, distances = analyse_trajectory(frames)

    verdict, confidence, prob_fake, prob_real = classify(feature_vec)

    # ── Results display ───────────────────────────────────────────
    st.markdown("---")

    # Verdict badge
    if verdict == "AI-Generated":
        st.error(f"## 🔴 Verdict: {verdict}")
    else:
        st.success(f"## 🟢 Verdict: {verdict}")

    # Stats row
    col1, col2, col3 = st.columns(3)
    col1.metric("Frames Analysed", len(frames))
    col2.metric("Confidence", f"{confidence * 100:.1f}%")
    col3.metric("Method", "DINOv2 ReStraV")

    # Confidence bars
    st.markdown("### Confidence Score")
    st.markdown("**AI-Generated probability:**")
    st.progress(float(prob_fake), text=f"{prob_fake * 100:.1f}%")
    st.markdown("**Real probability:**")
    st.progress(float(prob_real), text=f"{prob_real * 100:.1f}%")

    # Trajectory charts
    st.markdown("### Trajectory Analysis")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Curvature over time**")
        st.caption("Higher = more unnatural motion (AI videos tend to be higher)")
        import pandas as pd
        st.line_chart(pd.DataFrame({
            "Curvature": curvatures
        }))

    with col_b:
        st.markdown("**Step distance over time**")
        st.caption("Distance each frame moves in DINOv2 feature space")
        st.line_chart(pd.DataFrame({
            "Distance": distances
        }))

    # Summary stats
    st.markdown("### Trajectory Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Curvature",  f"{curvatures.mean():.4f}")
    c2.metric("Curvature Std",   f"{curvatures.std():.4f}")
    c3.metric("Mean Distance",   f"{distances.mean():.4f}")
    c4.metric("Distance Std",    f"{distances.std():.4f}")

    st.info(
        "💡 **How this works:** Each frame is converted into a 384-dimensional "
        "feature vector by DINOv2. The system then measures how smoothly these "
        "vectors flow over time. Real videos follow straighter, more consistent "
        "trajectories. AI-generated videos show higher curvature — they 'wobble' "
        "through feature space in ways that violate natural video dynamics."
    )