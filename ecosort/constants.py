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
    sorting_tips: tuple = ()


ONTARIO_CATEGORIES: List[WasteCategory] = [
    WasteCategory(
        0,
        "blue_bin",
        "Blue Bin (Recyclables)",
        "#2563EB",
        "♻️",
        "Cardboard, paper, plastic, metal, glass, black plastic, Styrofoam",
        (
            "Rinse containers before recycling",
            "Flatten cardboard boxes",
            "Remove caps from bottles",
            "No black plastic or Styrofoam in most municipalities",
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
        ),
    ),
    WasteCategory(
        3,
        "hazardous",
        "Household Hazardous Waste",
        "#DC2626",
        "⚠️",
        "Batteries, paint, chemicals, propane",
        (
            "Take to designated drop-off depots",
            "Never put in regular garbage or recycling",
            "Keep in original containers",
            "Check municipality for depot hours",
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
        ),
    ),
    WasteCategory(
        5,
        "yard_waste",
        "Yard Waste",
        "#65A30D",
        "🍂",
        "Leaves, grass clippings, branches",
        (
            "Use paper yard waste bags or open containers",
            "No plastic bags",
            "Branches must be under 10cm diameter",
            "Check seasonal curbside collection dates",
        ),
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
