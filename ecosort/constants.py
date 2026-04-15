"""Ontario 2026 Waste Categories Constants.

This module defines the 6 Ontario waste categories and provides
mappings from various dataset categories to the Ontario standard.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class WasteCategory:
    """Represents an Ontario waste category.
    
    Attributes:
        id: Unique numeric identifier.
        name: Internal category name (e.g., 'blue_bin').
        display: Human-readable display name.
        color: Hex color code for UI.
        icon: Emoji icon for the category.
        description: Brief description of items in category.
        sorting_tips: Tuple of sorting instructions.
    """
    id: int
    name: str
    display: str
    color: str
    icon: str
    description: str
    sorting_tips: Tuple[str, ...] = ()


# Ontario's 6 main waste categories
ONTARIO_CATEGORIES: List[WasteCategory] = [
    WasteCategory(
        0,
        "blue_bin",
        "Blue Bin (Recyclables)",
        "#2563EB",
        "♻️",
        "Cardboard, paper, plastic, metal, glass",
        (
            "Rinse containers before recycling",
            "Flatten cardboard boxes",
            "Remove caps from bottles",
            "No black plastic or Styrofoam in most municipalities",
            "Check local guidelines for accepted items",
        ),
    ),
    WasteCategory(
        1,
        "green_bin",
        "Green Bin (Organics)",
        "#16A34A",
        "🌿",
        "Food scraps, soiled paper, coffee grounds, pet waste",
        (
            "Use certified compostable bags",
            "No plastic bags in green bin",
            "Include meat, bones, dairy",
            "Soiled paper towels and napkins go here",
            "Coffee grounds and filters are compostable",
        ),
    ),
    WasteCategory(
        2,
        "garbage",
        "Garbage (Black Bin)",
        "#1F2937",
        "🗑️",
        "Non-recyclables, compostable plastics, ceramics",
        (
            "Last resort — check other bins first",
            "Compostable plastics go to garbage, not green bin",
            "Broken ceramics and mirrors go here",
            "Bag all garbage securely",
            "Diapers and pet waste go here",
        ),
    ),
    WasteCategory(
        3,
        "hazardous",
        "Household Hazardous Waste",
        "#DC2626",
        "⚠️",
        "Batteries, paint, chemicals, propane tanks",
        (
            "Take to designated drop-off depots",
            "Never put in regular garbage or recycling",
            "Keep in original containers",
            "Check municipality for depot hours",
            "Do not mix different hazardous materials",
        ),
    ),
    WasteCategory(
        4,
        "e_waste",
        "Electronic Waste",
        "#7C3AED",
        "💻",
        "Computers, phones, cables, small appliances",
        (
            "Drop off at e-waste depots or retail stores",
            "Wipe personal data before disposal",
            "Cables and chargers count as e-waste",
            "Many retailers accept old electronics",
            "Batteries should be removed separately",
        ),
    ),
    WasteCategory(
        5,
        "yard_waste",
        "Yard Waste",
        "#65A30D",
        "🍂",
        "Leaves, grass clippings, branches, plants",
        (
            "Use paper yard waste bags or open containers",
            "No plastic bags",
            "Branches must be under 10cm diameter",
            "Check seasonal curbside collection dates",
            "Plants with soil should have soil removed",
        ),
    ),
]

# Quick lookup dictionaries
CATEGORY_NAME_TO_ID: Dict[str, int] = {c.name: c.id for c in ONTARIO_CATEGORIES}
CATEGORY_ID_TO_NAME: Dict[int, str] = {c.id: c.name for c in ONTARIO_CATEGORIES}


# ============================================================================
# Dataset Category Mappings
# ============================================================================

# TrashNet dataset mapping
TRASHNET_TO_ONTARIO: Dict[str, str] = {
    "cardboard": "blue_bin",
    "glass": "blue_bin",
    "metal": "blue_bin",
    "paper": "blue_bin",
    "plastic": "blue_bin",
    "trash": "garbage",
}

# RealWaste (UCI) dataset mapping
REALWASTE_TO_ONTARIO: Dict[str, str] = {
    "Cardboard": "blue_bin",
    "Food Organics": "green_bin",
    "Glass": "blue_bin",
    "Metal": "blue_bin",
    "Miscellaneous Trash": "garbage",
    "Paper": "blue_bin",
    "Plastic": "blue_bin",
    "Textile Trash": "garbage",
    "Vegetation": "yard_waste",
}

# Kaggle Garbage Classification mapping
KAGGLE_GARBAGE_TO_ONTARIO: Dict[str, str] = {
    "battery": "hazardous",
    "biological": "green_bin",
    "brown-glass": "blue_bin",
    "cardboard": "blue_bin",
    "clothes": "garbage",
    "green-glass": "blue_bin",
    "metal": "blue_bin",
    "paper": "blue_bin",
    "plastic": "blue_bin",
    "shoes": "garbage",
    "trash": "garbage",
    "white-glass": "blue_bin",
}

# TACO dataset mapping (subset of common categories)
TACO_TO_ONTARIO: Dict[str, str] = {
    "Plastic bag": "garbage",
    "Plastic bottle": "blue_bin",
    "Plastic container": "blue_bin",
    "Glass bottle": "blue_bin",
    "Glass jar": "blue_bin",
    "Aluminum can": "blue_bin",
    "Steel can": "blue_bin",
    "Paper": "blue_bin",
    "Cardboard": "blue_bin",
    "Food waste": "green_bin",
    "Cigarette": "garbage",
    "Styrofoam": "garbage",
    "Electronics": "e_waste",
    "Battery": "hazardous",
}

# Combined mapping for all datasets
ALL_DATASET_MAPPINGS: Dict[str, Dict[str, str]] = {
    "trashnet": TRASHNET_TO_ONTARIO,
    "realwaste": REALWASTE_TO_ONTARIO,
    "kaggle_garbage": KAGGLE_GARBAGE_TO_ONTARIO,
    "taco": TACO_TO_ONTARIO,
}


def map_category(source_category: str, dataset: str = None) -> str:
    """Map a source category to Ontario category.
    
    Args:
        source_category: Category name from source dataset.
        dataset: Optional dataset name for specific mapping.
    
    Returns:
        Ontario category name.
    
    Example:
        >>> map_category("glass", "trashnet")
        'blue_bin'
        >>> map_category("Food Organics")
        'green_bin'
    """
    # Try dataset-specific mapping first
    if dataset and dataset in ALL_DATASET_MAPPINGS:
        mapping = ALL_DATASET_MAPPINGS[dataset]
        if source_category in mapping:
            return mapping[source_category]
    
    # Try all mappings
    source_lower = source_category.lower()
    for mapping in ALL_DATASET_MAPPINGS.values():
        if source_category in mapping:
            return mapping[source_category]
        if source_lower in {k.lower(): v for k, v in mapping.items()}:
            return mapping[source_category]
    
    # Default to garbage for unknown categories
    return "garbage"


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
