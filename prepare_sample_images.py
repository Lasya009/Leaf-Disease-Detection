import shutil
from pathlib import Path
import zipfile

# Source dataset (your full validation set)
src_dir = Path("data/processed/val")

# Destination folder for lightweight sample set
dst_dir = Path("sample_images/val")
dst_dir.mkdir(parents=True, exist_ok=True)

# Loop through each class folder in val
for class_dir in src_dir.iterdir():
    if class_dir.is_dir():
        # Create corresponding folder in sample_images/val
        target_class_dir = dst_dir / class_dir.name
        target_class_dir.mkdir(parents=True, exist_ok=True)

        # Get all images in the class folder
        images = [p for p in class_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]

        if images:
            # Copy just the first image (or more if you want)
            shutil.copy(images[0], target_class_dir / images[0].name)
            print(f"Copied {images[0].name} → {target_class_dir}")

# Create ZIP file
zip_path = Path("sample_images.zip")
with zipfile.ZipFile(zip_path, "w") as zipf:
    for file in dst_dir.rglob("*"):
        zipf.write(file, file.relative_to(dst_dir.parent))

print(f"✅ Created {zip_path} with sample images.")
