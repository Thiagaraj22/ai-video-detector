    # train_restrav.py
import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    accuracy_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

SAVE_DIR = "dataset/dino_features"

class TrajectoryMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.net(x)

# ── Load features ─────────────────────────────────────────────────
print("Loading features...")
train = np.load(os.path.join(SAVE_DIR, "train_features.npz"))
val   = np.load(os.path.join(SAVE_DIR, "val_features.npz"))

X_train, y_train = train['X'], train['y']
X_val,   y_val   = val['X'],   val['y']

X_train = X_train[:, :-1]
X_val   = X_val[:, :-1]

print(f"Train: {X_train.shape}, Val: {X_val.shape}")
print(f"Train — real: {(y_train==0).sum()}, fake: {(y_train==1).sum()}")
print(f"Val   — real: {(y_val==0).sum()},   fake: {(y_val==1).sum()}")

# ── Normalise features ────────────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

# ── Train two classifiers and compare ────────────────────────────
print("\n--- Logistic Regression ---")
lr = LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0)
lr.fit(X_train, y_train)
y_pred_lr  = lr.predict(X_val)
y_prob_lr  = lr.predict_proba(X_val)[:, 1]

print(classification_report(y_val, y_pred_lr, target_names=['real', 'fake']))
print(f"AUC-ROC: {roc_auc_score(y_val, y_prob_lr):.4f}")

print("\n--- Gradient Boosting ---")
gb = GradientBoostingClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
)
gb.fit(X_train, y_train)
y_pred_gb  = gb.predict(X_val)
y_prob_gb  = gb.predict_proba(X_val)[:, 1]

print(classification_report(y_val, y_pred_gb, target_names=['real', 'fake']))
print(f"AUC-ROC: {roc_auc_score(y_val, y_prob_gb):.4f}")

# ── Pick the better model ─────────────────────────────────────────
auc_lr = roc_auc_score(y_val, y_prob_lr)
auc_gb = roc_auc_score(y_val, y_prob_gb)

if auc_gb >= auc_lr:
    best_model = gb
    best_probs = y_prob_gb
    best_preds = y_pred_gb
    best_name  = "GradientBoosting"
else:
    best_model = lr
    best_probs = y_prob_lr
    best_preds = y_pred_lr
    best_name  = "LogisticRegression"

print(f"\nBest model: {best_name} (AUC = {max(auc_lr, auc_gb):.4f})")

# ── Save model + scaler ───────────────────────────────────────────
with open("restrav_model.pkl", "wb") as f:
    pickle.dump({"model": best_model, "scaler": scaler, "name": best_name}, f)
print("Model saved to restrav_model.pkl")

# ── Feature importance ────────────────────────────────────────────
feature_names = [
    "curv_mean", "curv_std", "curv_max", "curv_min",
    "curv_p25", "curv_p75", "curv_p90",
    "dist_mean", "dist_std", "dist_max", "dist_min",
    "dist_p25", "dist_p75"
]

if best_name == "GradientBoosting":
    importance = best_model.feature_importances_
    plt.figure(figsize=(9, 5))
    idx = np.argsort(importance)[::-1]
    plt.bar(range(len(feature_names)), importance[idx])
    plt.xticks(range(len(feature_names)),
               [feature_names[i] for i in idx], rotation=45, ha='right')
    plt.title("Feature Importance — ReStraV Trajectory Classifier")
    plt.tight_layout()
    plt.savefig("restrav_feature_importance.png", dpi=150)
    print("Saved: restrav_feature_importance.png")

# ── Confusion matrix ──────────────────────────────────────────────
cm = confusion_matrix(y_val, best_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'])
plt.title(f'Confusion Matrix — {best_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig("restrav_confusion_matrix.png", dpi=150)
print("Saved: restrav_confusion_matrix.png")

# ── ROC curve ─────────────────────────────────────────────────────
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_val, best_probs)
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, color='#2E75B6', linewidth=2,
         label=f'AUC = {max(auc_lr, auc_gb):.4f}')
plt.plot([0,1],[0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — ReStraV Classifier')
plt.legend()
plt.tight_layout()
plt.savefig("restrav_roc_curve.png", dpi=150)
print("Saved: restrav_roc_curve.png")

print("\nDone!")