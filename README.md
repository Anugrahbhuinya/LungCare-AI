# LungCare AI 🫁

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](<https://img.shields.io/badge/Python-3.11%20(Recommended)%20%7C%203.12-3776AB?style=flat&logo=python>)](https://www.python.org)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-E92063?style=flat&logo=pydantic)](https://docs.pydantic.dev/)
[![Tests](https://img.shields.io/badge/Tests-Pytest-0A9EDC?style=flat&logo=pytest)](https://docs.pytest.org/)

LungCare AI is an enterprise-grade AI-powered chest X-ray research and medical information assistance platform.

> ⚠️ **Research Disclaimer**: LungCare AI is an experimental prototype designed for clinical research, academic demonstration, and workflow exploration. It is **not** a certified medical diagnostic device.

---

## Project Structure

```
LungCare-AI/
│
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI application factory, lifespan & root alias
│   │   │
│   │   ├── api/
│   │   │   └── v1/
│   │   │       ├── router.py           # v1 centralized route aggregator & metadata
│   │   │       └── endpoints/
│   │   │           └── health.py       # Health and readiness probe endpoints
│   │   │
│   │   ├── core/
│   │   │   ├── config.py               # Pydantic v2 Settings & environment parsing
│   │   │   └── logging.py              # Centralized logging & millisecond precision
│   │   │
│   │   ├── schemas/
│   │   │   ├── health.py               # Health & Readiness Pydantic schemas
│   │   │   ├── metadata.py             # Discovery metadata schemas
│   │   │   └── error.py                # Standardized error response envelope
│   │   │
│   │   ├── services/                   # Business logic & domain service boundaries
│   │   └── utils/                      # Cross-cutting utilities & helpers
│   │
│   ├── tests/
│   │   └── test_health.py              # Async HTTPX test suite for probes, docs, & CORS
│   │
│   ├── requirements.txt                # Pinned backend dependencies
│   ├── .env.example                    # Template environment variables
│   ├── .gitignore                      # Git ignore rules for backend
│   └── README.md                       # Backend guide & runtime compatibility matrix
│
├── models/
│   └── LungCare_AI_Model/
│       ├── lungcare_densenet121.h5     # Trained DenseNet121 weights (224x224x3)
│       ├── lungcare_densenet121.keras  # Keras model format
│       ├── lungcare_config.json        # Class mapping & preprocessing config
│       └── lungcare_evaluation.json    # Validation metrics (97.68% accuracy)
│
├── frontend/                           # Client interface (Phase 3)
├── docs/                               # System documentation & architectural designs
└── README.md
```

---

## Getting Started

### Prerequisites & Python Compatibility

- **Recommended:** **Python 3.11** (Ensures full compatibility with TensorFlow 2.15–2.18 for Phase 2.2 DenseNet121 model loading).
- **Supported for Phase 2.1 Backend Core:** Python 3.11, 3.12, 3.13.
- Windows PowerShell, macOS terminal, or Linux (WSL2/Ubuntu recommended for CUDA GPU acceleration).

### 1. Environment Setup (Windows PowerShell)

```powershell
# Navigate to the backend directory
cd backend

# Create a virtual environment (Python 3.11 recommended)
py -3.11 -m venv .venv
# Alternatively, if python 3.11 is the default:
# python -m venv .venv

# Activate the virtual environment
.venv\Scripts\Activate.ps1

# Install pinned dependencies
python -m pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and configure environment variables as needed:

```powershell
Copy-Item .env.example .env
```

Key configuration flags:
- `DEBUG` (default: `false`): Production-safe default preventing internal exception stack trace leakage. Set to `true` only for local developer debugging.
- `ENABLE_DOCS` (default: `true` in development, `false` in production): Explicitly controls exposure of Swagger (`/docs`), ReDoc (`/redoc`), and OpenAPI schema (`/openapi.json`).
- `MAX_IMAGE_UPLOAD_BYTES` (default: `10485760` / 10MB): Maximum permitted raw upload payload size.
- `MAX_IMAGE_PIXELS` (default: `25000000` / 25MP): Pillow decompression bomb safety ceiling.

### 3. Run Development Server

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

When documentation endpoints are enabled (`ENABLE_DOCS=true` or in development mode):
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- Primary OpenAPI JSON: `http://localhost:8000/api/v1/openapi.json`
- Root OpenAPI JSON Alias: `http://localhost:8000/openapi.json`

In production (`ENVIRONMENT=production` and `ENABLE_DOCS=false`), documentation endpoints are strictly disabled to prevent schema reconnaissance.

### 4. Run Automated Tests

```powershell
python -m pytest -v
```

---

## API Endpoints Reference

| Endpoint                    | Method | Purpose                               | Response Sample / Highlights                                           |
| :-------------------------- | :----: | :------------------------------------ | :--------------------------------------------------------------------- |
| `/`                         | `GET`  | Root discovery metadata               | `{"app_name": "LungCare AI API", "version": "1.0.0", ...}`             |
| `/health`                   | `GET`  | Service liveness probe                | `{"status": "healthy", "service": "lungcare-api", "version": "1.0.0"}` |
| `/ready`                    | `GET`  | Readiness probe (model state)         | `{"status": "ready", "model_loaded": false, ...}`                      |
| `/openapi.json`             | `GET`  | Root OpenAPI schema alias             | JSON schema (OpenAPI 3.1)                                              |
| `/api/v1/`                  | `GET`  | Version 1 metadata & routes           | `{"app_name": "LungCare AI API", "api_version": "v1", ...}`            |
| `/api/v1/health`            | `GET`  | Versioned liveness probe              | `{"status": "healthy", "service": "lungcare-api", "version": "1.0.0"}` |
| `/api/v1/ready`             | `GET`  | Versioned readiness probe             | `{"status": "ready", "model_loaded": false, ...}`                      |
| `/api/v1/openapi.json`      | `GET`  | Versioned OpenAPI schema              | JSON schema (OpenAPI 3.1)                                              |
| `/api/v1/predictions`        | `POST` | DenseNet121 four-class inference      | `{"predicted_class": "PNEUMONIA", "confidence": 0.9854, ...}`         |
| `/api/v1/predictions/explain`| `POST` | Prediction + Grad-CAM Visual Explain  | `{"predicted_class": "PNEUMONIA", "overlay_base64": "data:...", ...}`  |

### Prediction & Explainability Usage

#### 1. POST `/api/v1/predictions`
Upload a chest radiograph for four-class DenseNet121 classification (COVID-19, Normal, Pneumonia, Tuberculosis).
```bash
curl -X POST "http://localhost:8000/api/v1/predictions" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@chest_xray.png;type=image/png"
```

#### 2. POST `/api/v1/predictions/explain`
Upload a chest radiograph to receive both the classification predictions and class-specific Grad-CAM visual heatmaps/overlays.
```bash
curl -X POST "http://localhost:8000/api/v1/predictions/explain" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@chest_xray.png;type=image/png" \
  -F "target_class=PNEUMONIA" \
  -F "alpha=0.4"
```

- **Request Parameters (multipart/form-data):**
  - `image` *(required)*: Binary image file (JPEG, PNG, WEBP, max 10MB, max 25M pixels).
  - `target_class` *(optional)*: Class to explain by index (0-3) or name (`COVID19`, `NORMAL`, `PNEUMONIA`, `TURBERCULOSIS`). Defaults to the model's predicted class.
  - `alpha` *(optional)*: Float in `[0.0, 1.0]` specifying overlay blending ratio. Defaults to `0.4`.

- **Base64 Decoding Expectations:**
  - `heatmap_base64` and `overlay_base64` are encoded as RFC 2397 Data URIs (`data:image/png;base64,<payload>`).
  - Directly bindable in frontend `<img src={data.overlay_base64} />` or decodable with standard Base64 libraries after stripping the prefix.

- **Error Codes:**
  - `400 Bad Request`: Empty image, corrupted image, invalid target class index, invalid alpha.
  - `413 Payload Too Large`: Upload > 10MB or exceeds 25,000,000 pixels (decompression bomb protection).
  - `415 Unsupported Media Type`: Unsupported format (e.g. BMP, TIFF, GIF, SVG).
  - `422 Unprocessable Entity`: Missing `image` field.
  - `502 Bad Gateway`: Model output validation failure, or Grad-CAM gradient divergence (NaN/Inf).
  - `503 Service Unavailable`: DenseNet121 model is not loaded in memory.
  - `500 Internal Server Error`: Unhandled runtime or rendering failure.
