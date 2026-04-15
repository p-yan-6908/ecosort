# Contributing to EcoSort

Thank you for your interest in contributing to EcoSort! This document provides guidelines and instructions for contributing.

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Git
- 8GB RAM minimum (for training)

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/ecosort.git
   cd ecosort
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"  # Installs development dependencies
   ```

4. **Download and prepare the dataset**
   ```bash
   python scripts/download_trashnet.py
   python scripts/prepare_dataset.py
   ```

5. **Run tests**
   ```bash
   pytest tests/ -v
   ```

## 🧪 Running Tests

We use pytest for testing. Run all tests with:

```bash
pytest tests/ -v
```

Run specific test files:

```bash
pytest tests/test_models.py -v
pytest tests/test_api.py -v
```

Run with coverage:

```bash
pytest tests/ --cov=ecosort --cov-report=html
```

## 📝 Code Style

### Python

- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Write docstrings for all public modules, classes, and functions
- Keep functions focused and under 50 lines when possible

### Type Hints

```python
# Good
def predict(self, image: Image.Image) -> Dict[str, Any]:
    """Predict waste category from image."""
    pass

# Bad
def predict(self, image):
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def classify_image(image_path: str) -> Dict[str, float]:
    """Classify a waste image and return probabilities.
    
    Args:
        image_path: Path to the image file.
    
    Returns:
        Dictionary mapping category names to probabilities.
    
    Raises:
        FileNotFoundError: If the image file doesn't exist.
        ValueError: If the file is not a valid image.
    
    Example:
        >>> probs = classify_image("trash.jpg")
        >>> print(probs["blue_bin"])
        0.95
    """
    pass
```

## 🏗️ Project Structure

```
ecosort/
├── ecosort/           # Main package
│   ├── api/           # FastAPI endpoints
│   ├── data/          # Dataset and transforms
│   ├── models/        # Model architecture
│   ├── training/      # Training pipeline
│   └── inference/     # Prediction logic
├── tests/             # Test suite
├── scripts/           # Training/utility scripts
├── web/               # Web interface
└── docs/              # Documentation
```

## 🔧 Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write clean, documented code
- Add tests for new functionality
- Ensure all tests pass

### 3. Commit Your Changes

Write clear commit messages:

```bash
git commit -m "Add batch prediction endpoint

- Add /predict/batch endpoint for multiple images
- Update API documentation
- Add tests for batch prediction"
```

### 4. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## 🧪 Pull Request Guidelines

### Before Submitting

- [ ] All tests pass
- [ ] Code is formatted and linted
- [ ] New code has tests
- [ ] Documentation is updated

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe how you tested these changes

## Screenshots (if applicable)
Add screenshots for UI changes
```

## 🐛 Reporting Bugs

Use GitHub Issues to report bugs. Include:

1. Description of the bug
2. Steps to reproduce
3. Expected behavior
4. Actual behavior
5. Environment (Python version, OS)

## 💡 Requesting Features

Open a GitHub Issue with:

1. Clear description of the feature
2. Use case and motivation
3. Possible implementation approach (optional)

## 📚 Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Ontario Waste Standards](https://www.ontario.ca/page/waste-management)

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! 🌿♻️
