"""Tests for WasteDataset class."""

import pytest
from pathlib import Path
from PIL import Image
import numpy as np

from ecosort.data.dataset import WasteDataset, create_dataloaders


class TestWasteDataset:
    """Tests for WasteDataset class."""

    @pytest.fixture
    def sample_transform(self):
        """Simple transform for testing."""
        def transform(image):
            return {"image": np.array(image)}
        return transform

    def test_dataset_creation(self, temp_data_dir):
        """Test dataset can be created with valid directory."""
        dataset = WasteDataset(temp_data_dir, split="train")
        assert dataset is not None
        assert len(dataset.classes) > 0

    def test_dataset_len(self, temp_data_dir):
        """Test dataset length matches number of images."""
        dataset = WasteDataset(temp_data_dir, split="train")
        assert len(dataset) > 0

    def test_dataset_getitem(self, temp_data_dir, sample_transform):
        """Test dataset __getitem__ returns correct types."""
        dataset = WasteDataset(
            temp_data_dir,
            split="train",
            transform=sample_transform
        )
        image, label = dataset[0]
        
        assert image is not None
        assert isinstance(label, int)

    def test_dataset_class_distribution(self, temp_data_dir):
        """Test class distribution calculation."""
        dataset = WasteDataset(temp_data_dir, split="train")
        distribution = dataset.get_class_distribution()
        
        assert isinstance(distribution, dict)
        assert all(isinstance(v, int) for v in distribution.values())

    def test_dataset_raises_on_missing_split(self, temp_data_dir):
        """Test dataset raises on non-existent split."""
        with pytest.raises(ValueError):
            WasteDataset(temp_data_dir, split="nonexistent")

    def test_dataset_uses_custom_class_mapping(self, temp_data_dir):
        """Test dataset can use custom class-to-index mapping."""
        custom_mapping = {"blue_bin": 0, "green_bin": 1, "garbage": 2}
        dataset = WasteDataset(
            temp_data_dir,
            split="train",
            class_to_idx=custom_mapping
        )
        
        assert dataset.class_to_idx == custom_mapping


class TestDataLoaders:
    """Tests for create_dataloaders function."""

    def test_creates_loaders(self, temp_data_dir):
        """Test that dataloaders are created."""
        def dummy_transform(image):
            return {"image": np.array(image)}
        
        train_loader, val_loader, test_loader = create_dataloaders(
            temp_data_dir,
            dummy_transform,
            dummy_transform,
            batch_size=2,
            num_workers=0,
        )
        
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

    def test_loaders_produce_batches(self, temp_data_dir):
        """Test that loaders produce batches of correct size."""
        def dummy_transform(image):
            return {"image": np.array(image)}
        
        train_loader, _, _ = create_dataloaders(
            temp_data_dir,
            dummy_transform,
            dummy_transform,
            batch_size=2,
            num_workers=0,
        )
        
        batch = next(iter(train_loader))
        images, labels = batch
        
        assert images is not None
        assert labels is not None
