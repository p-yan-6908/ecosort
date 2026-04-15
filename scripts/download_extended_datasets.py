#!/usr/bin/env python3
"""
Download and prepare extended waste classification datasets.

This script downloads multiple publicly available waste datasets
and organizes them for training the EcoSort classifier on all
Ontario waste categories.

Datasets:
1. TrashNet - 2,527 images (glass, paper, cardboard, plastic, metal, trash)
2. RealWaste - 4,752 images (9 categories including vegetation, food organics)
3. Garbage Classification (Kaggle) - 10,000+ images
4. E-Waste Dataset - Electronic waste images
5. CompostNet - Organic waste images
"""

import argparse
import zipfile
import tarfile
import urllib.request
import shutil
from pathlib import Path
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dataset URLs and info
DATASETS = {
    "trashnet": {
        "url": "https://huggingface.co/datasets/garythung/trashnet/resolve/main/data/dataset-resized.zip",
        "filename": "trashnet.zip",
        "categories": {
            "glass": "blue_bin",
            "paper": "blue_bin",
            "cardboard": "blue_bin",
            "plastic": "blue_bin",
            "metal": "blue_bin",
            "trash": "garbage",
        },
        "size": "~400MB",
    },
    "realwaste": {
        "url": "https://archive.ics.uci.edu/static/public/908/realwaste.zip",
        "filename": "realwaste.zip",
        "categories": {
            "Cardboard": "blue_bin",
            "Food Organics": "green_bin",
            "Glass": "blue_bin",
            "Metal": "blue_bin",
            "Miscellaneous Trash": "garbage",
            "Paper": "blue_bin",
            "Plastic": "blue_bin",
            "Textile Trash": "garbage",
            "Vegetation": "yard_waste",
        },
        "size": "~650MB",
    },
}

# Additional dataset sources (require manual download)
MANUAL_DATASETS = {
    "kaggle_garbage": {
        "kaggle_id": "asdasdasasdas/garbage-classification",
        "categories": {
            "glass": "blue_bin",
            "paper": "blue_bin",
            "cardboard": "blue_bin",
            "plastic": "blue_bin",
            "metal": "blue_bin",
            "trash": "garbage",
        },
        "instructions": "Download from: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification",
    },
    "kaggle_recyclable": {
        "kaggle_id": "alistairking/recyclable-and-household-waste-classification",
        "instructions": "Download from: https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification",
    },
    "ewaste_roboflow": {
        "url": "https://universe.roboflow.com/electronic-waste-detection/e-waste-dataset-r0ojc",
        "instructions": "Export from Roboflow Universe (requires free account)",
        "target": "e_waste",
    },
    "taco": {
        "url": "http://tacodataset.org/",
        "instructions": "Download from TACO dataset website (requires registration)",
        "notes": "Large-scale trash detection dataset with 1500+ images",
    },
}


def download_file(url: str, dest: Path, desc: str = None) -> bool:
    """Download a file with progress bar."""
    try:
        logger.info(f"Downloading {desc or dest.name}...")
        urllib.request.urlretrieve(url, dest)
        logger.info(f"Downloaded to {dest}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def extract_zip(zip_path: Path, dest_dir: Path) -> bool:
    """Extract a zip file."""
    try:
        logger.info(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dest_dir)
        logger.info(f"Extracted to {dest_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to extract {zip_path}: {e}")
        return False


def download_trashnet(data_dir: Path) -> bool:
    """Download TrashNet dataset."""
    logger.info("=" * 60)
    logger.info("DOWNLOADING TRASHNET DATASET")
    logger.info("=" * 60)
    
    raw_dir = data_dir / "raw" / "trashnet"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = raw_dir / "dataset-resized.zip"
    
    if zip_path.exists():
        logger.info(f"TrashNet already downloaded: {zip_path}")
        return True
    
    # Download from HuggingFace
    url = DATASETS["trashnet"]["url"]
    if download_file(url, zip_path, "TrashNet"):
        return extract_zip(zip_path, raw_dir)
    return False


def download_realwaste(data_dir: Path) -> bool:
    """Download RealWaste dataset from UCI."""
    logger.info("=" * 60)
    logger.info("DOWNLOADING REALWASTE DATASET")
    logger.info("=" * 60)
    
    raw_dir = data_dir / "raw" / "realwaste"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = raw_dir / "realwaste.zip"
    
    if zip_path.exists():
        logger.info(f"RealWaste already downloaded: {zip_path}")
        return True
    
    url = DATASETS["realwaste"]["url"]
    if download_file(url, zip_path, "RealWaste"):
        return extract_zip(zip_path, raw_dir)
    return False


def print_manual_download_instructions():
    """Print instructions for manual downloads."""
    logger.info("=" * 60)
    logger.info("MANUAL DOWNLOAD REQUIRED FOR ADDITIONAL DATASETS")
    logger.info("=" * 60)
    
    for name, info in MANUAL_DATASETS.items():
        logger.info(f"\n{name.upper()}:")
        if "url" in info:
            logger.info(f"  URL: {info['url']}")
        if "instructions" in info:
            logger.info(f"  Instructions: {info['instructions']}")
        if "categories" in info:
            logger.info(f"  Categories: {list(info['categories'].keys())}")
        if "notes" in info:
            logger.info(f"  Notes: {info['notes']}")


def main():
    parser = argparse.ArgumentParser(description="Download waste classification datasets")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--dataset", choices=["all", "trashnet", "realwaste"], default="all", help="Dataset to download")
    parser.add_argument("--manual", action="store_true", help="Show manual download instructions")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if args.manual:
        print_manual_download_instructions()
        return
    
    if args.dataset in ["all", "trashnet"]:
        download_trashnet(data_dir)
    
    if args.dataset in ["all", "realwaste"]:
        download_realwaste(data_dir)
    
    # Always show manual instructions at the end
    print_manual_download_instructions()
    
    logger.info("=" * 60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("1. Download additional datasets manually (see instructions above)")
    logger.info("2. Run: python scripts/prepare_extended_dataset.py")


if __name__ == "__main__":
    main()
