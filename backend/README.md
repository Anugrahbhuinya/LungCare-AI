# LungCare AI — Backend Foundation

Enterprise FastAPI backend for LungCare AI, an extensible AI-assisted chest X-ray research and medical information assistance platform.

---

## 1. Runtime & Compatibility Guide

### Python & TensorFlow Compatibility Matrix

The backend foundation is lightweight and compatible with **Python 3.11, 3.12, and 3.13**. However, for upcoming milestones (Phase 2.2 model loading and inference with DenseNet121), **Python 3.11 is strongly recommended as the primary production runtime**.

| Component / Layer | Python 3.11 (Recommended) | Python 3.12 | Python 3.13 (Host Current) | Notes |
|:---|:---:|:---:|:---:|:---|
| **Phase 2.1 Backend Core** (FastAPI, Pydantic, HTTPX, Pytest) | ✅ Supported | ✅ Supported | ✅ Supported | Foundation executes with 100% test pass rate across all versions. |
| **Phase 2.2 TensorFlow / Keras 3** (DenseNet121 Inference) | 🌟 **Golden Standard** (TF 2.15–2.18) | ⚠️ Partial / CPU only on Windows | ❌ Not Recommended | Native Windows pre-built wheels for TensorFlow are not yet stabilized for 3.13. |
| **Phase 2.3 RAG & Embeddings** (ChromaDB / LangChain) | ✅ Supported | ✅ Supported | ⚠️ Emerging | Vector store native C++ bindings are best supported on 3.11/3.12. |

---

## 2. Environment Setup Strategies

### Option A: Native Windows with Python 3.11 (Recommended for Local Dev)

Ensure Python 3.11 is installed on the system (via python.org or Microsoft Store).

```powershell
# Navigate to the backend directory
cd backend

# Create a dedicated Python 3.11 virtual environment
py -3.11 -m venv .venv

# Activate the virtual environment in PowerShell
.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install pinned foundation dependencies
python -m pip install -r requirements.txt
```

> **Windows Native TensorFlow Note:** Starting with TensorFlow 2.11, native Windows GPU support was deprecated by the TensorFlow team. For native Windows, TensorFlow operates in CPU mode. If local GPU acceleration (NVIDIA CUDA) is needed on Windows, use **WSL2 (Option B)**.

---

### Option B: WSL2 (Windows Subsystem for Linux — Recommended for GPU Acceleration)

WSL2 provides full native Linux kernel compatibility and direct access to NVIDIA GPUs via the NVIDIA Container Toolkit and CUDA on WSL.

```bash
# Inside Ubuntu WSL2 terminal:
cd /mnt/c/Users/ASUS/OneDrive/Desktop/LungCare-AI/backend

# Create Python 3.11 environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Option C: Docker Containerization (Recommended for Production Deployment)

Containerizing with a standard Linux base image ensures exact reproducibility and simplifies GPU passthrough.

```dockerfile
# Target production Dockerfile base:
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (libgl1 for OpenCV / image processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 3. Configuration & Startup

### Environment Configuration
```powershell
# Copy the example configuration template
Copy-Item .env.example .env
```

Key environment variables in `.env`:
- `APP_NAME`: Application title (default: `"LungCare AI API"`)
- `ENVIRONMENT`: Runtime environment (`development` / `production`)
- `DEBUG`: Boolean flag (`true` enables Swagger UI and `/openapi.json`)
- `CORS_ORIGINS`: Explicit JSON list of allowed origins (wildcards prohibited with credentials)
- `LOG_LEVEL`: Severity threshold (`DEBUG`, `INFO`, `WARNING`, `ERROR`)

### Starting the Server
```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 4. API Endpoints Reference

| Route | Method | Description | Response Schema |
|:---|:---:|:---|:---|
| `/` | `GET` | Application discovery & probe links | `RootMetadataResponse` |
| `/health` | `GET` | Liveness probe (Kubernetes / ALB) | `HealthResponse` |
| `/ready` | `GET` | Readiness probe (dependency checks) | `ReadinessResponse` |
| `/docs` | `GET` | Interactive Swagger UI (Dev mode) | HTML |
| `/redoc` | `GET` | Interactive ReDoc documentation | HTML |
| `/openapi.json` | `GET` | Root OpenAPI schema alias | JSON (OpenAPI 3.1) |
| `/api/v1/` | `GET` | Version 1 discovery metadata | `APIMetadataResponse` |
| `/api/v1/health` | `GET` | Versioned liveness probe | `HealthResponse` |
| `/api/v1/ready` | `GET` | Versioned readiness probe | `ReadinessResponse` |
| `/api/v1/openapi.json` | `GET` | Primary versioned OpenAPI schema | JSON (OpenAPI 3.1) |

---

## 5. Automated Testing

Tests use modern asynchronous testing via `httpx.AsyncClient` and `ASGITransport`:

```powershell
python -m pytest -v
```

