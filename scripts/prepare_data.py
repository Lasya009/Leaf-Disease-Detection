"""
Prepare PlantVillage dataset into train/val/test folders.
Assumes `data_dir` contains many class subfolders, each containing images.
"""
import argparse
import os
import random
import shutil
from pathlib import Path


def is_image_file(p: Path):
    return p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}


def prepare(data_dir, out_dir, val_split=0.15, test_split=0.1, seed=42):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    random.seed(seed)

    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir {data_dir} does not exist")

    classes = [p for p in data_dir.iterdir() if p.is_dir()]
    if not classes:
        raise RuntimeError("No class subfolders found in data_dir. Ensure dataset extracted correctly.")

    (out_dir / 'train').mkdir(parents=True, exist_ok=True)
    (out_dir / 'val').mkdir(parents=True, exist_ok=True)
    (out_dir / 'test').mkdir(parents=True, exist_ok=True)

    mapping = []

    for cls in classes:
        images = [p for p in cls.iterdir() if p.is_file() and is_image_file(p)]
        if not images:
            continue
        random.shuffle(images)
        n = len(images)
        n_test = int(n * test_split)
        n_val = int(n * val_split)
        n_train = n - n_val - n_test

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        for subset_name, subset in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
            target_dir = out_dir / subset_name / cls.name
            target_dir.mkdir(parents=True, exist_ok=True)
            for p in subset:
                # copy rather than move so original dataset remains
                # if link mode is enabled, create a hard link to save space
                try:
                    if prepare._use_links:
                        # create hard link; if target exists, skip
                        tgt = target_dir / p.name
                        if not tgt.exists():
                            os.link(p, tgt)
                    else:
                        shutil.copy2(p, target_dir / p.name)
                except Exception:
                    # fallback to copy on any error (permissions, cross-device, etc.)
                    shutil.copy2(p, target_dir / p.name)

        mapping.append(cls.name)

    # write class mapping
    with open(out_dir / 'classes.txt', 'w', encoding='utf-8') as f:
        for c in sorted(mapping):
            f.write(f"{c}\n")

    print(f"Prepared data in {out_dir}. Classes: {len(mapping)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--val-split', type=float, default=0.15)
    parser.add_argument('--test-split', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--link', action='store_true', help='Create hard links instead of copying files (saves space on NTFS)')
    args = parser.parse_args()
    # store link preference on the function to avoid changing signature
    prepare._use_links = bool(args.link)
    prepare(args.data_dir, args.out_dir, args.val_split, args.test_split, args.seed)
