#!/usr/bin/env python3
from pathlib import Path
from ecosort.data.download import download_trashnet


def main():
    data_dir = Path("data")
    download_trashnet(data_dir)
    print("TrashNet downloaded successfully!")


if __name__ == "__main__":
    main()
