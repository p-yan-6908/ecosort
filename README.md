# ♻️ EcoSort - Ontario Waste Classification AI

AI-powered waste classification system for Ontario, Canada (2026 Standards).

## Features

- 🖼️ Image-based waste classification
- 🍁 Ontario 2026 waste sorting standards
- 🚀 Fast inference with MobileNetV3-Small
- 🌐 Beautiful web interface
- 📱 Mobile-responsive design
- 🔧 Fine-tunable on custom data

## Categories

| Category | Icon | Items |
|----------|------|-------|
| Blue Bin (Recyclables) | ♻️ | Cardboard, paper, plastic, metal, glass |
| Green Bin (Organics) | 🌿 | Food scraps, soiled paper, coffee grounds |
| Garbage (Black Bin) | 🗑️ | Non-recyclables, compostable plastics |
| Household Hazardous | ⚠️ | Batteries, paint, chemicals |
| Electronic Waste | 💻 | Computers, phones, cables |
| Yard Waste | 🍂 | Leaves, grass, branches |

## Quick Start

### Prerequisites

- Python 3.9+
- 8GB RAM minimum
- macOS, Linux, or Windows

### Installation

```bash
# Clone repository
git clone <repository-url>
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

## Project Structure

```
ecosort/
├── ecosort/           # Main package
│   ├── data/          # Dataset and transforms
│   ├── models/        # Model architecture
│   ├── training/      # Training pipeline
│   ├── inference/     # Prediction logic
│   └── api/           # FastAPI backend
├── scripts/           # Training and utility scripts
├── web/               # Web interface
├── data/              # Datasets
└── models/            # Model checkpoints
```

## Configuration

Edit `config.yaml` to customize:
- Model architecture
- Training hyperparameters
- Data augmentation
- API settings

## API Endpoints

- `POST /predict` - Classify single image
- `POST /predict/top-k` - Get top-k predictions
- `GET /classes` - List all categories
- `GET /health` - Health check

## Training on Custom Data

1. Organize images in `data/processed/ontario/`:
   ```
   ontario/
   ├── train/
   │   ├── blue_bin/
   │   ├── green_bin/
   │   └── ...
   ├── val/
   └── test/
   ```

2. Run training:
   ```bash
   python scripts/train.py
   ```

## Performance

- Model size: ~4MB (quantized)
- Inference time: ~50ms on CPU
- RAM usage: ~200MB during inference
- Target accuracy: 85%+ on Ontario categories

## License

MIT License