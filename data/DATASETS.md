# Waste Classification Datasets for EcoSort

This document lists recommended datasets for training the EcoSort classifier on all 6 Ontario waste categories.

## Ontario Waste Categories

1. **Blue Bin (Recyclables)** - cardboard, paper, plastic, metal, glass
2. **Green Bin (Organics)** - food scraps, soiled paper, coffee grounds
3. **Garbage** - non-recyclables, compostable plastics, ceramics
4. **Household Hazardous** - batteries, paint, chemicals, propane
5. **Electronic Waste** - computers, phones, cables, appliances
6. **Yard Waste** - leaves, grass, branches

---

## Primary Datasets (Auto-Download)

### 1. TrashNet
- **Images**: 2,527
- **Categories**: glass, paper, cardboard, plastic, metal, trash
- **Source**: Stanford CS229 Project
- **License**: MIT
- **Download**: `python scripts/download_extended_datasets.py --dataset trashnet`
- **Coverage**: Blue Bin (recyclables), Garbage

### 2. RealWaste (UCI)
- **Images**: 4,752
- **Categories**: Cardboard, Food Organics, Glass, Metal, Miscellaneous Trash, Paper, Plastic, Textile Trash, Vegetation
- **Source**: UCI Machine Learning Repository
- **License**: CC BY 4.0
- **Download**: `python scripts/download_extended_datasets.py --dataset realwaste`
- **Coverage**: Blue Bin, Green Bin, Garbage, Yard Waste

---

## Additional Datasets (Manual Download)

### 3. Garbage Classification (Kaggle)
- **Images**: 10,000+
- **Categories**: battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass
- **URL**: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification
- **Coverage**: Blue Bin, Green Bin, Garbage, Hazardous

### 4. Recyclable and Household Waste Classification
- **Images**: 30,000+
- **Categories**: recyclables, household waste
- **URL**: https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification
- **Coverage**: Blue Bin, Garbage

### 5. E-Waste Dataset (Roboflow)
- **Images**: 2,000+
- **Categories**: laptops, phones, monitors, keyboards, cables
- **URL**: https://universe.roboflow.com/electronic-waste-detection/e-waste-dataset-r0ojc
- **Coverage**: E-Waste (strong coverage)

### 6. RecyBat24 (Battery Detection)
- **Images**: 5,000+
- **Categories**: lithium-ion batteries in e-waste
- **URL**: https://www.nature.com/articles/s41597-025-05211-5
- **Coverage**: Hazardous (batteries)

### 7. CompostNet
- **Images**: 1,000+
- **Categories**: food waste for compost
- **URL**: https://github.com/sarahmfrost/compostnet
- **Coverage**: Green Bin (organics)

### 8. TACO (Trash Annotations in Context)
- **Images**: 1,500+
- **Categories**: 60+ waste categories
- **URL**: http://tacodataset.org/
- **Coverage**: All categories (requires registration)

---

## Category Coverage Matrix

| Dataset | Blue Bin | Green Bin | Garbage | Hazardous | E-Waste | Yard Waste |
|---------|----------|-----------|---------|-----------|---------|------------|
| TrashNet | ✅ 2,391 | ❌ | ✅ 137 | ❌ | ❌ | ❌ |
| RealWaste | ✅ 2,591 | ✅ 411 | ✅ 495 | ❌ | ❌ | ✅ 436 |
| Kaggle Garbage | ✅ 5,000+ | ✅ 1,000+ | ✅ 2,000+ | ✅ 1,000+ | ❌ | ❌ |
| E-Waste | ❌ | ❌ | ❌ | ✅ batteries | ✅ 2,000+ | ❌ |
| CompostNet | ❌ | ✅ 1,000+ | ❌ | ❌ | ❌ | ❌ |

---

## Quick Start

```bash
# 1. Download primary datasets
python scripts/download_extended_datasets.py --dataset all

# 2. Download additional datasets manually (see URLs above)

# 3. Prepare merged dataset
python scripts/prepare_extended_dataset.py --data-dir data

# 4. Train model
python scripts/train.py --data-dir data/processed/ontario_extended
```

---

## Recommended Minimum Images per Category

For good training performance:
- **Training set**: 500+ images per category
- **Validation set**: 100+ images per category
- **Test set**: 100+ images per category

---

## Data Augmentation

The training pipeline includes augmentation:
- Random horizontal/vertical flips
- Random rotation (±30°)
- Color jitter
- Random crops
- Blur and noise

---

## Adding Custom Data

To add your own images:

1. Create a directory structure:
   ```
   data/raw/custom/
   ├── blue_bin/
   ├── green_bin/
   ├── garbage/
   ├── hazardous/
   ├── e_waste/
   └── yard_waste/
   ```

2. Add images to appropriate folders

3. Run the preparation script:
   ```bash
   python scripts/prepare_extended_dataset.py
   ```

---

## Notes

- All images are resized to max 1024x1024
- Converted to JPEG format
- Split 70/15/15 for train/val/test
- Balanced across categories

## License

Different datasets have different licenses. Check individual dataset pages for details.
