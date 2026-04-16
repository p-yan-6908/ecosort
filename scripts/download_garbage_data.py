#!/usr/bin/env python3
"""Download and integrate garbage/other weak category datasets."""

import shutil
import random
from pathlib import Path
from PIL import Image

# Dataset paths (already downloaded via kagglehub)
DATASET1_TRASH = Path("/Users/pyan/.cache/kagglehub/datasets/farzadnekouei/trash-type-image-dataset/versions/1/TrashType_Image_Dataset/trash")
DATASET2_TRASH = Path("/Users/pyan/.cache/kagglehub/datasets/asdasdasasdas/garbage-classification/versions/2/Garbage classification/Garbage classification/trash")
HAZARDOUS_GLOVES = Path("/Users/pyan/.cache/kagglehub/datasets/shreyanshjain10/hazardous-waste/versions/1/Dataset/gloves")
HOUSEHOLD_DATA = Path("/Users/pyan/.cache/kagglehub/datasets/alistairking/recyclable-and-household-waste-classification/versions/1/images/images")

# Output
OUTPUT_DIR = Path("/Users/pyan/ecosort/data/raw/garbage_fix")

# Category mappings from household waste dataset
# Map to Ontario categories
HOUSEHOLD_TO_ONTARIO = {
    # Garbage (black bin) - non-recyclable
    "plastic_trash_bags": "garbage",
    "plastic_straws": "garbage",
    "styrofoam_cups": "garbage",
    "styrofoam_food_containers": "garbage",
    "disposable_plastic_cutlery": "garbage",
    "paper_cups": "garbage",  # often have plastic lining
    "shoes": "garbage",
    "clothing": "garbage",  # textile waste goes to garbage
    
    # Blue bin (recyclables)
    "aluminum_food_cans": "blue_bin",
    "aluminum_soda_cans": "blue_bin",
    "steel_food_cans": "blue_bin",
    "glass_beverage_bottles": "blue_bin",
    "glass_food_jars": "blue_bin",
    "glass_cosmetic_containers": "blue_bin",
    "plastic_soda_bottles": "blue_bin",
    "plastic_water_bottles": "blue_bin",
    "plastic_detergent_bottles": "blue_bin",
    "plastic_food_containers": "blue_bin",
    "plastic_cup_lids": "blue_bin",
    "cardboard_boxes": "blue_bin",
    "cardboard_packaging": "blue_bin",
    "newspaper": "blue_bin",
    "magazines": "blue_bin",
    "office_paper": "blue_bin",
    
    # Green bin (organics)
    "food_waste": "green_bin",
    "coffee_grounds": "green_bin",
    "tea_bags": "green_bin",
    "eggshells": "green_bin",
    
    # Aerosol cans -> hazardous (pressurized)
    "aerosol_cans": "hazardous",
    
    # Plastic shopping bags -> garbage (most municipalities)
    "plastic_shopping_bags": "garbage",
}


def copy_images(src_dir: Path, dst_dir: Path, prefix: str = "", limit: int = None):
    """Copy images from src to dst directory."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    images = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png")) + list(src_dir.glob("*.jpeg"))
    if limit:
        random.shuffle(images)
        images = images[:limit]
    
    for i, img_path in enumerate(images):
        try:
            # Verify it's a valid image
            img = Image.open(img_path)
            img.verify()
            
            # Copy with new name
            ext = img_path.suffix
            new_name = f"{prefix}_{i:04d}{ext}" if prefix else img_path.name
            shutil.copy2(img_path, dst_dir / new_name)
        except Exception as e:
            print(f"  Skipping {img_path}: {e}")
    
    return len(list(dst_dir.glob("*")))


def main():
    print("=" * 70)
    print("DOWNLOADING ADDITIONAL GARBAGE/HOUSEHOLD WASTE DATA")
    print("=" * 70)
    
    # Create output directories
    for category in ["garbage", "hazardous", "e_waste", "yard_waste"]:
        (OUTPUT_DIR / category).mkdir(parents=True, exist_ok=True)
    
    counts = {}
    
    # 1. Copy trash images from dataset 1 & 2 (real garbage)
    print("\n1. Copying trash/garbage images...")
    n1 = copy_images(DATASET1_TRASH, OUTPUT_DIR / "garbage", "trashtype")
    n2 = copy_images(DATASET2_TRASH, OUTPUT_DIR / "garbage", "garbage_class")
    counts["garbage"] = n1 + n2
    print(f"   Garbage: {counts['garbage']} images")
    
    # 2. Copy hazardous waste images (gloves, masks)
    print("\n2. Copying hazardous waste images...")
    n = copy_images(HAZARDOUS_GLOVES, OUTPUT_DIR / "hazardous", "hazard_gloves")
    counts["hazardous"] = n
    print(f"   Hazardous: {counts['hazardous']} images")
    
    # 3. Copy household waste images by category
    print("\n3. Copying household waste images by category...")
    for household_cat, ontario_cat in HOUSEHOLD_TO_ONTARIO.items():
        src = HOUSEHOLD_DATA / household_cat / "real_world"
        if src.exists():
            # Only copy garbage and hazardous to avoid overwhelming blue_bin/green_bin
            if ontario_cat in ["garbage", "hazardous", "yard_waste"]:
                prefix = household_cat
                n = copy_images(src, OUTPUT_DIR / ontario_cat, prefix, limit=200)
                counts[ontario_cat] = counts.get(ontario_cat, 0) + n
                print(f"   {household_cat} -> {ontario_cat}: {n} images")
    
    # 4. Add some food_waste to yard_waste (similar organic nature)
    print("\n4. Adding organic waste to yard_waste...")
    for cat in ["food_waste", "coffee_grounds", "tea_bags"]:
        src = HOUSEHOLD_DATA / cat / "real_world"
        if src.exists():
            n = copy_images(src, OUTPUT_DIR / "yard_waste", cat[:4], limit=50)
            counts["yard_waste"] = counts.get("yard_waste", 0) + n
            print(f"   {cat} -> yard_waste: {n} images")
    
    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    for cat, count in sorted(counts.items()):
        print(f"   {cat}: {count} images")
    
    print(f"\nTotal new images: {sum(counts.values())}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Now integrate into processed dataset
    print("\n" + "=" * 70)
    print("INTEGRATING INTO PROCESSED DATASET")
    print("=" * 70)
    
    PROCESSED_DIR = Path("/Users/pyan/ecosort/data/processed/ontario")
    
    for category in ["garbage", "hazardous", "yard_waste"]:
        src_dir = OUTPUT_DIR / category
        dst_dir = PROCESSED_DIR / "train" / category
        
        if dst_dir.exists():
            # Remove old placeholder images
            for f in dst_dir.glob("garbage_*.jpg"):
                f.unlink()
            
            # Copy new images
            new_count = 0
            for img in src_dir.glob("*"):
                if img.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                    shutil.copy2(img, dst_dir / img.name)
                    new_count += 1
            
            total = len(list(dst_dir.glob("*.jpg")))
            print(f"   {category}: added {new_count} new images, total now: {total}")
    
    print("\nDone! Run prepare_extended_dataset.py to recreate train/val/test splits.")


if __name__ == "__main__":
    random.seed(42)
    main()
