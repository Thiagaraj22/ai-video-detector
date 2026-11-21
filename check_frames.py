import os

def count_files(folder):
    total = 0
    for root, dirs, files in os.walk(folder):
        total += len([f for f in files if f.endswith('.jpg')])
    return total

train_real = count_files("dataset/frames/train/real")
train_fake = count_files("dataset/frames/train/fake")
val_real   = count_files("dataset/frames/val/real")
val_fake   = count_files("dataset/frames/val/fake")

print("=== Frame counts ===")
print(f"Train real:  {train_real}")
print(f"Train fake:  {train_fake}")
print(f"Val real:    {val_real}")
print(f"Val fake:    {val_fake}")
print(f"Total:       {train_real + train_fake + val_real + val_fake}")