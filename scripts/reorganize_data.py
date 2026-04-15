#!/usr/bin/env python3
"""
Reorganize data directory structure.

This script:
1. Consolidates all scattered data into clean structure
2. Moves external data to appropriate locations  
3. Creates a clean, organized directory structure
"""

import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("/Users/pyan/ecosort/data")

def reorganize():
    """Reorganize the data directory."""
    logger.info("=" * 60)
    logger.info("REORGANIZING DATA DIRECTORY")
    logger.info("=" * 60)
    
    # Step 1: Move external data to raw/downloads
    logger.info("\n1. Organizing external datasets...")
    
    external_dir = DATA_DIR / "external"
    raw_downloads = DATA_DIR / "raw" / "downloads"
    raw_downloads.mkdir(parents=True, exist_ok=True)
    
    # Move green_bin_new to raw/downloads/green_bin_images
    green_bin_new = external_dir / "green_bin_new"
    if green_bin_new.exists():
        dest = raw_downloads / "green_bin_organic"
        if not dest.exists():
            shutil.move(str(green_bin_new), str(dest))
            logger.info(f"  Moved green_bin_new -> {dest}")
    
    # Move TACO to raw/downloads/TACO
    taco_dir = external_dir / "TACO"
    if taco_dir.exists():
        dest = raw_downloads / "TACO"
        if not dest.exists():
            shutil.move(str(taco_dir), str(dest))
            logger.info(f"  Moved TACO -> {dest}")
    
    # Clean up empty directories in external
    for empty_dir in ["green_bin", "e_waste", "hazardous", "yard_waste", "taco_garbage"]:
        dir_path = external_dir / empty_dir
        if dir_path.exists() and dir_path.is_dir():
            try:
                dir_path.rmdir()  # Only removes if empty
                logger.info(f"  Removed empty directory: {empty_dir}")
            except:
                pass
    
    # Step 2: Ensure processed structure exists
    logger.info("\n2. Ensuring processed directory structure...")
    
    processed_dir = DATA_DIR / "processed"
    for split in ["train", "val", "test"]:
        for category in ["blue_bin", "green_bin", "garbage", "hazardous", "e_waste", "yard_waste"]:
            (processed_dir / "ontario" / split / category).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"  Created structure in {processed_dir / 'ontario'}")
    
    # Step 3: Create category mappings file
    logger.info("\n3. Creating dataset manifest...")
    
    manifest = {
        "trashnet": {
            "path": "raw/trashnet/dataset-resized",
            "categories": ["glass", "paper", "cardboard", "plastic", "metal", "trash"],
            "count": 2527
        },
        "realwaste": {
            "path": "raw/realwaste", 
            "categories": ["Cardboard", "Food Organics", "Glass", "Metal", "Miscellaneous Trash", 
                          "Paper", "Plastic", "Textile Trash", "Vegetation"],
            "count": 4752
        },
        "green_bin_organic": {
            "path": "raw/downloads/green_bin_organic",
            "categories": ["food_waste"],
            "count": 750
        },
        "TACO": {
            "path": "raw/downloads/TACO/data",
            "categories": ["various"],
            "count": 230
        }
    }
    
    import json
    manifest_path = DATA_DIR / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"  Created manifest: {manifest_path}")
    
    # Step 4: Print final structure
    logger.info("\n4. Final directory structure:")
    
    for root, dirs, files in DATA_DIR.walk():
        if ".git" in str(root) or "_backup" in str(root):
            continue
        level = len(root.relative_to(DATA_DIR).parts)
        indent = "  " * level
        logger.info(f"{indent}{root.name}/")
        if level < 3:
            for d in sorted(dirs):
                if not d.startswith('.'):
                    logger.info(f"{indent}  {d}/")
    
    logger.info("\n" + "=" * 60)
    logger.info("REORGANIZATION COMPLETE")
    logger.info("=" * 60)
    
    # Print summary
    logger.info("\nSummary:")
    logger.info(f"  Raw datasets: raw/trashnet, raw/realwaste, raw/downloads/")
    logger.info(f"  Processed: processed/ontario/{{train,val,test}}/{{categories}}/")
    logger.info(f"  Backup: _backup/ontario/ (previous processed data)")
    logger.info("\nNext steps:")
    logger.info("  1. Run: python scripts/prepare_extended_dataset.py")
    logger.info("  2. This will merge all datasets and create final training data")

if __name__ == "__main__":
    reorganize()
