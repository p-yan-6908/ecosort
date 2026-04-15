"""Tests for constants module."""

import pytest

from ecosort.constants import (
    ONTARIO_CATEGORIES,
    CATEGORY_NAME_TO_ID,
    CATEGORY_ID_TO_NAME,
    TRASHNET_TO_ONTARIO,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


class TestOntarioCategories:
    """Tests for Ontario categories constants."""

    def test_categories_count(self):
        """Test we have exactly 6 categories."""
        assert len(ONTARIO_CATEGORIES) == 6

    def test_categories_have_required_fields(self):
        """Test each category has all required fields."""
        for cat in ONTARIO_CATEGORIES:
            assert hasattr(cat, 'id')
            assert hasattr(cat, 'name')
            assert hasattr(cat, 'display')
            assert hasattr(cat, 'color')
            assert hasattr(cat, 'icon')
            assert hasattr(cat, 'description')

    def test_category_ids_are_unique(self):
        """Test all category IDs are unique."""
        ids = [cat.id for cat in ONTARIO_CATEGORIES]
        assert len(ids) == len(set(ids))

    def test_category_names_are_unique(self):
        """Test all category names are unique."""
        names = [cat.name for cat in ONTARIO_CATEGORIES]
        assert len(names) == len(set(names))

    def test_sorting_tips_exist(self):
        """Test each category has sorting tips."""
        for cat in ONTARIO_CATEGORIES:
            assert hasattr(cat, 'sorting_tips')
            assert len(cat.sorting_tips) > 0


class TestCategoryMappings:
    """Tests for category mapping dictionaries."""

    def test_name_to_id_mapping_complete(self):
        """Test name to ID mapping includes all categories."""
        for cat in ONTARIO_CATEGORIES:
            assert cat.name in CATEGORY_NAME_TO_ID
            assert CATEGORY_NAME_TO_ID[cat.name] == cat.id

    def test_id_to_name_mapping_complete(self):
        """Test ID to name mapping includes all categories."""
        for cat in ONTARIO_CATEGORIES:
            assert cat.id in CATEGORY_ID_TO_NAME
            assert CATEGORY_ID_TO_NAME[cat.id] == cat.name

    def test_mappings_are_consistent(self):
        """Test forward and reverse mappings are consistent."""
        for name, id_ in CATEGORY_NAME_TO_ID.items():
            assert CATEGORY_ID_TO_NAME[id_] == name


class TestTrashNetMapping:
    """Tests for TrashNet to Ontario mapping."""

    def test_trashnet_mapping_has_expected_classes(self):
        """Test TrashNet mapping includes expected original classes."""
        expected = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        for cls in expected:
            assert cls in TRASHNET_TO_ONTARIO

    def test_trashnet_maps_to_valid_ontario_categories(self):
        """Test TrashNet classes map to valid Ontario categories."""
        for trashnet_cls, ontario_cls in TRASHNET_TO_ONTARIO.items():
            assert ontario_cls in CATEGORY_NAME_TO_ID


class TestImageNetConstants:
    """Tests for ImageNet normalization constants."""

    def test_imagenet_mean_length(self):
        """Test ImageNet mean has 3 values."""
        assert len(IMAGENET_MEAN) == 3

    def test_imagenet_std_length(self):
        """Test ImageNet std has 3 values."""
        assert len(IMAGENET_STD) == 3

    def test_imagenet_mean_range(self):
        """Test ImageNet mean values are in valid range."""
        for val in IMAGENET_MEAN:
            assert 0.0 <= val <= 1.0

    def test_imagenet_std_positive(self):
        """Test ImageNet std values are positive."""
        for val in IMAGENET_STD:
            assert val > 0
