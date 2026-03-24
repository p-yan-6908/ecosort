"""Ontario 2026 Waste Categories Constants"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class WasteCategory:
    id: int
    name: str
    display: str
    color: str
    icon: str
    description: str


ONTARIO_CATEGORIES: List[WasteCategory] = [
    WasteCategory(
        0,
        "blue_bin",
        "Blue Bin (Recyclables)",
        "#2563EB",
        "♻️",
        "Cardboard, paper, plastic, metal, glass, black plastic, Styrofoam",
    ),
    WasteCategory(
        1,
        "green_bin",
        "Green Bin (Organics)",
        "#16A34A",
        "🌿",
        "Food scraps, soiled paper, coffee grounds, pet waste",
    ),
    WasteCategory(
        2,
        "garbage",
        "Garbage (Black Bin)",
        "#1F2937",
        "🗑️",
        "Non-recyclables, compostable plastics, ceramics",
    ),
    WasteCategory(
        3,
        "hazardous",
        "Household Hazardous Waste",
        "#DC2626",
        "⚠️",
        "Batteries, paint, chemicals, propane",
    ),
    WasteCategory(
        4,
        "e_waste",
        "Electronic Waste",
        "#7C3AED",
        "💻",
        "Computers, phones, cables, small appliances",
    ),
    WasteCategory(
        5,
        "yard_waste",
        "Yard Waste",
        "#65A30D",
        "🍂",
        "Leaves, grass clippings, branches",
    ),
]

CATEGORY_NAME_TO_ID: Dict[str, int] = {c.name: c.id for c in ONTARIO_CATEGORIES}
CATEGORY_ID_TO_NAME: Dict[int, str] = {c.id: c.name for c in ONTARIO_CATEGORIES}

TRASHNET_TO_ONTARIO: Dict[str, str] = {
    "cardboard": "blue_bin",
    "glass": "blue_bin",
    "metal": "blue_bin",
    "paper": "blue_bin",
    "plastic": "blue_bin",
    "trash": "garbage",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
