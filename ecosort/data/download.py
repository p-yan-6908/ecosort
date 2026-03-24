"""Dataset Download Utilities"""

from pathlib import Path
import zipfile
import urllib.request

TRASHNET_URL = (
    "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
)


def download_file(url: str, dest_path: Path, desc: str = "Downloading"):
    """Download a file with progress."""
    print(f"{desc}: {url}")
    urllib.request.urlretrieve(url, dest_path)
    print(f"Saved to: {dest_path}")


def download_trashnet(data_dir: Path) -> Path:
    """Download TrashNet dataset."""
    import shutil

    zip_path = data_dir / "trashnet.zip"
    extract_dir = data_dir / "raw" / "trashnet"
    dataset_dir = extract_dir / "dataset-resized"

    if dataset_dir.exists():
        print(f"TrashNet already exists at {dataset_dir}")
        return dataset_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    download_file(TRASHNET_URL, zip_path, "Downloading TrashNet")

    print("Extracting...")
    temp_dir = data_dir / "raw"
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    # Move dataset-resized to trashnet folder
    temp_dataset = temp_dir / "dataset-resized"
    if temp_dataset.exists() and not dataset_dir.exists():
        shutil.move(str(temp_dataset), str(dataset_dir))

    # Clean up zip and __MACOSX
    zip_path.unlink()
    macosx_dir = temp_dir / "__MACOSX"
    if macosx_dir.exists():
        shutil.rmtree(macosx_dir)

    print(f"TrashNet ready at {dataset_dir}")
    return dataset_dir
