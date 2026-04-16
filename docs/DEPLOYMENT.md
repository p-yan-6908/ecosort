# EcoSort Deployment Guide

## Prerequisites

- Docker (optional)
- Python 3.9+
- 4GB RAM minimum
- GPU (optional, for faster inference)

---

## Option 1: Docker Deployment

### Build Image

```bash
docker build -t ecosort:latest .
```

### Run Container

```bash
docker run -d \
  --name ecosort \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  ecosort:latest
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  ecosort:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - WORKERS=4
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```bash
docker-compose up -d
```

---

## Option 2: Direct Python Deployment

### Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### Run Server

```bash
# Development
python run_server.py

# Production with Uvicorn
uvicorn ecosort.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Option 3: Cloud Deployment

### AWS EC2

1. Launch EC2 instance (t3.medium or larger)
2. Install Docker
3. Run Docker container as above
4. Configure security group to allow port 8000

### Google Cloud Run

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/ecosort

# Deploy to Cloud Run
gcloud run deploy ecosort \
  --image gcr.io/PROJECT_ID/ecosort \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Heroku

1. Create `Procfile`:
   ```
   web: uvicorn ecosort.api.main:app --host 0.0.0.0 --port $PORT
   ```

2. Deploy:
   ```bash
   heroku create ecosort-app
   heroku container:push web
   heroku container:release web
   ```

---

## Production Checklist

- [ ] Use HTTPS (TLS/SSL)
- [ ] Add authentication (API keys)
- [ ] Configure rate limiting
- [ ] Set up logging and monitoring
- [ ] Configure health checks
- [ ] Set up CI/CD pipeline
- [ ] Configure backup for models

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| HOST | 0.0.0.0 | Server host |
| PORT | 8000 | Server port |
| WORKERS | 4 | Number of workers |
| LOG_LEVEL | info | Logging level |

---

## Monitoring

### Health Check Endpoint

```bash
curl http://localhost:8000/health
```

### Metrics Endpoint

```bash
curl http://localhost:8000/metrics
```

---

## Scaling

### Horizontal Scaling

Deploy multiple instances behind a load balancer:

```
                    ┌──────────────┐
                    │ Load Balancer│
                    └──────┬───────┘
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │  EcoSort #1  │ │  EcoSort #2  │ │  EcoSort #3  │
    └──────────────┘ └──────────────┘ └──────────────┘
```

### Vertical Scaling

For GPU support, use instances with NVIDIA GPUs:

```bash
docker run --gpus all -p 8000:8000 ecosort:latest
```
