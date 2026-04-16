# ♻️ EcoSort - Ontario Waste Classification AI

AI-powered waste classification system for Ontario, Canada (2026 Standards).

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **97.32%** |
| **Validation Accuracy** | 97.28% |
| **Parameters** | 1.10M |
| **Model Size** | ~4 MB |
| **Inference Time** | ~50ms (CPU) |

## ✨ Features

- 🖼️ **Image-based waste classification** - Upload or capture photos
- 🍁 **Ontario 2026 waste sorting standards** - Up-to-date category mappings
- 🚀 **Fast inference** - MobileNetV3-Small + ECA Attention
- 🌐 **Beautiful web interface** - Organic, forest-inspired design
- 📱 **Mobile-responsive** - Works on phones, tablets, and desktops
- 📊 **Batch prediction** - Process multiple images at once
- 📜 **Prediction history** - Track recent classifications
- 🔧 **Fine-tunable** - Train on custom data

## 🗂️ Categories

| Category | Icon | Items |
|----------|------|-------|
| Blue Bin (Recyclables) | ♻️ | Cardboard, paper, plastic, metal, glass |
| Green Bin (Organics) | 🌿 | Food scraps, soiled paper, coffee grounds |
| Garbage (Black Bin) | 🗑️ | Non-recyclables, compostable plastics |
| Household Hazardous | ⚠️ | Batteries, paint, chemicals |
| Electronic Waste | 💻 | Computers, phones, cables |
| Yard Waste | 🍂 | Leaves, grass, branches |

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- 8GB RAM minimum
- macOS, Linux, or Windows

### Installation

```bash
# Clone repository
git clone https://github.com/p-yan-6908/ecosort.git
cd ecosort

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Server

```bash
# Development
python run_server.py

# Production
uvicorn ecosort.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Open http://localhost:8000/static/index.html in your browser.

### Docker

```bash
# Build and run
docker-compose up -d

# Or manually
docker build -t ecosort .
docker run -p 8000:8000 ecosort
```

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [API Documentation](docs/API.md) | REST API endpoints and usage |
| [Model Documentation](docs/MODEL.md) | Model architecture and training |
| [Deployment Guide](docs/DEPLOYMENT.md) | Production deployment options |
| [Contributing Guide](CONTRIBUTING.md) | Development guidelines |
| [Autoresearch Results](autoresearch_results.md) | Model optimization experiments |

## 📡 API Endpoints

### Single Image Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

**Response:**
```json
{
  "class_name": "blue_bin",
  "display_name": "Blue Bin (Recyclables)",
  "confidence": 0.95,
  "icon": "♻️",
  "description": "Cardboard, paper, plastic, metal, glass"
}
```

### Other Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Single image prediction |
| `/predict/batch` | POST | Batch image prediction |
| `/predict/top-k` | POST | Top-K predictions |
| `/classes` | GET | List all categories |
| `/health` | GET | Health check |
| `/metrics` | GET | Model metrics |

Full API documentation at http://localhost:8000/docs

## 📁 Project Structure

```
ecosort/
├── ecosort/           # Main package
│   ├── api/           # FastAPI endpoints
│   │   └── routes/    # API route handlers
│   ├── data/          # Dataset and transforms
│   ├── models/        # Model architecture
│   ├── training/      # Training pipeline
│   └── inference/     # Prediction logic
├── tests/             # Test suite (46 tests)
├── scripts/           # Training/utility scripts
├── web/               # Web interface
├── docs/              # Documentation
├── models/            # Model checkpoints
│   └── checkpoints/   # Trained models
└── data/              # Dataset storage
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
model:
  architecture: mobilenet_v3_small
  head_type: eca  # Best performing
  num_classes: 6
  dropout: 0.2
  pretrained: true

training:
  label_smoothing: 0.1  # Critical for best performance
  phase1:
    epochs: 10
    learning_rate: 0.01
    freeze_backbone: true
  phase2:
    epochs: 20
    learning_rate: 0.0001
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=ecosort --cov-report=html
```

## 🎨 Web Interface Features

- **Drag & Drop** - Drop images directly onto the upload area
- **Camera Capture** - Take photos with your device camera
- **Confidence Display** - See how confident the model is
- **Sorting Tips** - Get specific instructions for each category
- **Prediction History** - View recent predictions (stored locally)
- **Responsive Design** - Works on all screen sizes

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License

## 🙏 Acknowledgments

- TrashNet dataset for initial training data
- RealWaste dataset for additional training images
- Ontario Government for waste sorting standards
- FastAPI and PyTorch communities

---

Made with 💚 for a cleaner Ontario
