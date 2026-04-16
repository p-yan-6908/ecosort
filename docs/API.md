# EcoSort API Documentation

## Overview

The EcoSort API provides waste classification for Ontario, Canada (2026 Standards). Upload an image and receive classification results with confidence scores.

**Base URL:** `http://localhost:8000`

## Authentication

No authentication required for local development. For production, add API key authentication.

---

## Endpoints

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0"
}
```

---

### List Classes

```http
GET /classes
```

**Response:**
```json
{
  "categories": [
    {
      "id": 0,
      "name": "blue_bin",
      "display_name": "Blue Bin (Recyclables)",
      "icon": "♻️",
      "color": "#2563EB",
      "description": "Cardboard, paper, plastic, metal, glass"
    },
    ...
  ]
}
```

---

### Predict Single Image

```http
POST /predict
Content-Type: multipart/form-data

file: <image_file>
```

**Response:**
```json
{
  "class_id": 0,
  "class_name": "blue_bin",
  "display_name": "Blue Bin (Recyclables)",
  "confidence": 0.95,
  "icon": "♻️",
  "color": "#2563EB",
  "description": "Cardboard, paper, plastic, metal, glass",
  "all_probabilities": {
    "blue_bin": 0.95,
    "green_bin": 0.03,
    "garbage": 0.01,
    "hazardous": 0.005,
    "e_waste": 0.003,
    "yard_waste": 0.002
  }
}
```

**Example (cURL):**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg"
```

**Example (Python):**
```python
import requests

with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )
result = response.json()
print(f"Class: {result['display_name']}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

### Predict Top-K Classes

```http
POST /predict/top-k?k=3
Content-Type: multipart/form-data

file: <image_file>
```

**Response:**
```json
{
  "predictions": [
    {"class_name": "blue_bin", "display_name": "Blue Bin", "confidence": 0.95, "icon": "♻️"},
    {"class_name": "green_bin", "display_name": "Green Bin", "confidence": 0.03, "icon": "🌿"},
    {"class_name": "garbage", "display_name": "Garbage", "confidence": 0.01, "icon": "🗑️"}
  ]
}
```

---

### Batch Prediction

```http
POST /predict/batch
Content-Type: multipart/form-data

files: <multiple image files>
```

**Response:**
```json
{
  "results": [
    {"filename": "image1.jpg", "prediction": {...}},
    {"filename": "image2.jpg", "prediction": {...}}
  ],
  "total": 2
}
```

---

### Model Metrics

```http
GET /metrics
```

**Response:**
```json
{
  "model": {
    "architecture": "mobilenet_v3_small",
    "head_type": "eca",
    "num_classes": 6,
    "num_parameters": 1096073
  },
  "performance": {
    "validation_accuracy": 0.9728,
    "test_accuracy": 0.9732
  }
}
```

---

## Error Handling

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 400 | Invalid request (wrong content type, corrupted image) |
| 500 | Server error (model not loaded, internal error) |

**Error Response:**
```json
{
  "detail": "Error message describing the issue"
}
```

---

## Interactive Documentation

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## Categories Reference

| ID | Name | Display Name | Icon | Description |
|----|------|--------------|------|-------------|
| 0 | blue_bin | Blue Bin (Recyclables) | ♻️ | Cardboard, paper, plastic, metal, glass |
| 1 | green_bin | Green Bin (Organics) | 🌿 | Food scraps, soiled paper, coffee grounds |
| 2 | garbage | Garbage (Black Bin) | 🗑️ | Non-recyclables, compostable plastics |
| 3 | hazardous | Household Hazardous | ⚠️ | Batteries, paint, chemicals |
| 4 | e_waste | Electronic Waste | 💻 | Computers, phones, cables |
| 5 | yard_waste | Yard Waste | 🍂 | Leaves, grass, branches |
