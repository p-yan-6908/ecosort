# ♻️ EcoSort - Ontario Waste Classification AI

AI-powered waste classification system for Ontario, Canada (2026 Standards).

## ✨ Features

- 🖼️ **Image-based waste classification** - Upload or capture photos
- 🍁 **Ontario 2026 waste sorting standards** - Up-to-date category mappings
- 🚀 **Fast inference** - MobileNetV3-Small for quick predictions (~50ms)
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

### Download Dataset

```bash
python scripts/download_trashnet.py
python scripts/prepare_dataset.py
```

### Train Model

```bash
# Full training (both phases)
python scripts/train.py

# Or train specific phase
python scripts/train.py --phase 1
python scripts/train.py --phase 2
```

### Run Web Interface

```bash
uvicorn ecosort.api.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/static/index.html in your browser.

## 📡 API Endpoints

### Single Image Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

### Top-K Predictions

```bash
curl -X POST "http://localhost:8000/predict/top-k?k=3" \
  -F "file=@image.jpg"
```

### Other Endpoints

- `GET /classes` - List all categories
- `GET /health` - Health check
- `GET /metrics` - Model performance metrics

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
├── tests/             # Test suite
├── scripts/           # Training/utility scripts
├── web/               # Web interface
│   ├── js/            # JavaScript
│   ├── css/           # Styles
│   └── index.html     # Main page
└── docs/              # Documentation
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
model:
  architecture: mobilenet_v3_small
  num_classes: 3
  dropout: 0.2
  pretrained: true

training:
  batch_size: 8
  phase1:
    epochs: 10
    learning_rate: 0.01
  phase2:
    epochs: 20
    learning_rate: 0.0001
```

## 🧪 Testing

Run all tests:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=ecosort --cov-report=html
```

## 🎨 Web Interface Features

- **Drag & Drop** - Drop images directly onto the upload area
- **Camera Capture** - Take photos with your device camera
- **Confidence Display** - See how confident the model is
- **Sorting Tips** - Get specific instructions for each category
- **Prediction History** - View recent predictions (stored locally)
- **Responsive Design** - Works on all screen sizes

## 📊 Performance

- Model size: ~4MB (quantized)
- Inference time: ~50ms on CPU
- RAM usage: ~200MB during inference
- Target accuracy: 85%+ on Ontario categories

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

MIT License

## 🙏 Acknowledgments

- TrashNet dataset for initial training data
- Ontario Government for waste sorting standards
- FastAPI and PyTorch communities

---

Made with 💚 for a cleaner Ontario
