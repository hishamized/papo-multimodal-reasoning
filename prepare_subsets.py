import os
import json
from datasets import load_dataset
from PIL import Image

data_dir = "/home/hishamized/.gemini/antigravity/scratch/papo_project/data"
os.makedirs(os.path.join(data_dir, "train_images"), exist_ok=True)
os.makedirs(os.path.join(data_dir, "val_images"), exist_ok=True)

print("Streaming and saving 50 training samples from PAPOGalaxy/PAPO_ViRL39K_train...")
train_stream = load_dataset("PAPOGalaxy/PAPO_ViRL39K_train", split="train", streaming=True)
train_records = []
for idx, item in enumerate(train_stream):
    if idx >= 50:
        break
    # Extract image
    imgs = item.get("images", [])
    img_filename = f"train_img_{idx:04d}.png"
    img_path = os.path.join(data_dir, "train_images", img_filename)
    if imgs and len(imgs) > 0:
        pil_img = imgs[0]
        pil_img.save(img_path)
    else:
        img_path = None
    
    train_records.append({
        "id": idx,
        "problem": item.get("problem", ""),
        "answer": item.get("answer", ""),
        "image_path": img_path
    })
    if (idx + 1) % 10 == 0:
        print(f"Processed {idx + 1}/50 train samples")

with open(os.path.join(data_dir, "train_subset.json"), "w") as f:
    json.dump(train_records, f, indent=2)
print("Saved train_subset.json with 50 samples!")

print("Streaming and saving 20 validation samples from PAPOGalaxy/PAPO_MMK12_test...")
val_stream = load_dataset("PAPOGalaxy/PAPO_MMK12_test", split="train", streaming=True)
val_records = []
for idx, item in enumerate(val_stream):
    if idx >= 20:
        break
    imgs = item.get("images", [])
    img_filename = f"val_img_{idx:04d}.png"
    img_path = os.path.join(data_dir, "val_images", img_filename)
    if imgs and len(imgs) > 0:
        pil_img = imgs[0]
        pil_img.save(img_path)
    else:
        img_path = None
    
    val_records.append({
        "id": idx,
        "problem": item.get("problem", ""),
        "answer": item.get("answer", ""),
        "image_path": img_path
    })

with open(os.path.join(data_dir, "val_subset.json"), "w") as f:
    json.dump(val_records, f, indent=2)
print("Saved val_subset.json with 20 samples!")
