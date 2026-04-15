# EcoSort Data Directory

This directory contains all training and test data for the EcoSort waste classifier.

## Directory Structure

```
data/
├── raw/                    # Raw downloaded datasets (unchanged from source)
│   ├── trashnet/          # TrashNet dataset (2,527 images)
│   ├── realwaste/         # RealWaste dataset (4,752 images)
│   └── ...                # Other raw datasets
│
├── processed/              # Processed and split datasets
│   └── ontario/           # Ready-to-use Ontario 6-category dataset
│       ├── train/         # Training set (70%)
│       ├── val/           # Validation set (15%)
│       └── test/          # Test set (15%)
│
└── README.md              # This file
```

## Ontario Categories

All processed data is organized into 6 categories:

| Category | Description |
|----------|-------------|
| `blue_bin` | Recyclables (cardboard, paper, plastic, metal, glass) |
| `green_bin` | Organics (food scraps, soiled paper) |
| `garbage` | Non-recyclables, compostable plastics |
| `hazardous` | Batteries, paint, chemicals |
| `e_waste` | Electronics, computers, phones |
| `yard_waste` | Leaves, grass, branches |

## Downloading Datasets

```bash
# Download all available datasets
python scripts/download_extended_datasets.py --dataset all

# Download specific dataset
python scripts/download_extended_datasets.py --dataset realwaste
```

## Preparing the Dataset

```bash
# Process and merge all datasets
python scripts/prepare_extended_dataset.py --data-dir data

# Output: data/processed/ontario/
```

## Current Dataset Status

Run to see current status:
```bash
python scripts/prepare_extended_dataset.py --data-dir data --status
```

## Adding Custom Data

1. Place raw images in `data/raw/custom/<category>/`
2. Run preparation script
3. Images will be automatically split and added

## Notes

- All processed images are resized to max 1024x1024
- Converted to JPEG format
- Split ratio: 70% train, 15% val, 15% test
- Random seed: 42 (reproducible)
