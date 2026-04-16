#!/usr/bin/env python3
"""Recreate train/val/test splits with improved garbage data."""

import random
import shutil
from pathlib import Path

random.seed(42)

DATA_DIR = Path("/Users/pyan/ecosort/data/processed/ontario")
SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}

def recreate_splits():
    print("=" * 70)
    print("RECREATING TRAIN/VAL/TEST SPLITS")
    print("=" * 70)
    
    categories = ["blue_bin", "green_bin", "garbage", "hazardous", "e_waste", "yard_waste"]
    
    # Collect all images and copy to temp location first
    all_images = {}
    temp_dir = Path("/tmp/ecosort_images")
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True)
    
    for cat in categories:
        cat_temp = temp_dir / cat
        cat_temp.mkdir(parents=True)
        
        # Copy from all existing split directories
        count = 0
        for split in ["train", "val", "test"]:
            split_dir = DATA_DIR / split / cat
            if split_dir.exists():
                for img in split_dir.glob("*.jpg"):
                    shutil.copy2(img, cat_temp / f"{img.stem}_{split}{img.suffix}")
                    count += 1
        
        all_images[cat] = list(cat_temp.glob("*.jpg"))
        print(f"  {cat}: {len(all_images[cat])} total images")
    
    # Clear and recreate split directories
    print("\nCreating new splits...")
    for split in SPLITS:
        for cat in categories:
            split_dir = DATA_DIR / split / cat
            shutil.rmtree(split_dir, ignore_errors=True)
            split_dir.mkdir(parents=True, exist_ok=True)
    
    # Shuffle and split
    for cat, images in all_images.items():
        random.shuffle(images)
        n = len(images)
        n_train = int(n * SPLITS["train"])
        n_val = int(n * SPLITS["val"])
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]
        
        for split, imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            split_dir = DATA_DIR / split / cat
            for img in imgs:
                # Rename to remove split suffix
                new_name = img.stem.replace("_train", "").replace("_val", "").replace("_test", "") + img.suffix
                shutil.copy2(img, split_dir / new_name)
        
        print(f"  {cat}: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")
    
    # Cleanup temp
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL DATASET SUMMARY")
    print("=" * 70)
    
    total = {"train": 0, "val": 0, "test": 0}
    for split in SPLITS:
        print(f"\n{split.upper()}:")
        for cat in categories:
            split_dir = DATA_DIR / split / cat
            n = len(list(split_dir.glob("*.jpg")))
            total[split] += n
            print(f"  {cat}: {n}")
    
    print(f"\nTotal: train={total['train']}, val={total['val']}, test={total['test']}")
    print(f"Grand total: {sum(total.values())}")


if __name__ == "__main__":
    recreate_splits()
