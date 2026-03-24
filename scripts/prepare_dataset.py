#!/usr/bin/env python3
from pathlib import Path
from ecosort.data.trashnet_mapping import map_trashnet_to_ontario


def main():
    trashnet_dir = Path("data/raw/trashnet/dataset-resized")
    output_dir = Path("data/processed/ontario")

    if not trashnet_dir.exists():
        print(f"TrashNet not found at {trashnet_dir}")
        print("Run 'python scripts/download_trashnet.py' first")
        return

    print("Mapping TrashNet to Ontario categories...")
    map_trashnet_to_ontario(trashnet_dir, output_dir)
    print(f"Dataset prepared at {output_dir}")
    print("\nNext steps:")
    print("1. Add additional images for green_bin, hazardous, e_waste, yard_waste")
    print("2. Run 'python scripts/train.py' to train the model")


if __name__ == "__main__":
    main()
