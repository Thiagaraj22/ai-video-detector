import os

real_videos = os.listdir("dataset/real")
fake_videos = os.listdir("dataset/fake")

print(f"Real videos found: {len(real_videos)}")
print(f"Fake videos found: {len(fake_videos)}")
print()
print("First 5 real videos:", real_videos[:5])
print("First 5 fake videos:", fake_videos[:5])