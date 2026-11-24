# Phase 2.3.4.3 — Error Schema Unification & Domain Error Code Preservation Implementation Report

**Project:** LungCare AI  
**Milestone:** Phase 2.3.4.3 — Error Schema Unification & Domain Error Code Preservation  
**Engineer Role:** Principal Backend Engineer — FastAPI, Python Exception Architecture, and Production Error Handling  
**Date:** November 2026  
**Status:** Implementation Complete, Ready for Independent Verification  

---

## 1. Executive Summary

Phase 2.3.4.3 addresses **AUDIT-API-02** from the Phase 2.3.4.1 architecture audit by establishing an enterprise machine-readable error architecture across the LungCare AI FastAPI backend.

Prior to this milestone, domain exceptions (`EMPTY_IMAGE`, `UNSUPPORTED_FORMAT`, `MODEL_UNAVAILABLE`, `INVALID_MODEL_OUTPUT`, `GRADCAM_GRADIENT_ERROR`, etc.) were caught inside individual route handlers and re-raised as generic Starlette `HTTPException` instances. As a consequence, FastAPI's `http_exception_handler` serialized `error_type="HTTPException"`, completely erasing the underlying machine-readable failure identity and forcing client consumers to rely on fragile substring matching in error messages.

The implementation introduces:
1. A centralized error response factory (`create_error_response`) guaranteeing consistent Pydantic v2 `ErrorResponse` serialization and `X-Request-ID` header synchronization across all failure scenarios.
2. Centralized FastAPI domain exception handlers that preserve domain error codes while mapping them to standard HTTP status codes.
3. Elimination of redundant try/catch boilerplate in `predict_chest_xray` and `explain_chest_xray`, allowing domain exceptions to bubble directly to centralized handlers.
4. Strict production security sanitization preventing internal memory addresses, stack traces, or secrets from escaping when `DEBUG=False`.
5. Backward compatibility for standard `StarletteHTTPException` (preserving `error_type="HTTPException"` for 404s) and `RequestValidationError` (`error_type="ValidationError"` for 422s).

---

## 2. Existing Exception Hierarchy & Domain Taxonomy

All domain exceptions in LungCare AI inherit from clean domain-specific root exception classes:

```
Exception
 ├── StarletteHTTPException / HTTPException
 ├── RequestValidationError
 ├── ImagePreprocessingError (error_code: IMAGE_PREPROCESSING_ERROR)
 │    ├── EmptyImageError (error_code: EMPTY_IMAGE)
 │    ├── UnsupportedFormatError (error_code: UNSUPPORTED_FORMAT)
 │    ├── ImageTooLargeError (error_code: IMAGE_TOO_LARGE)
 │    ├── DecompressionBombError (error_code: DECOMPRESSION_BOMB)
 │    ├── CorruptImageError (error_code: CORRUPT_IMAGE)
 │    ├── InvalidImageModeError (error_code: INVALID_IMAGE_MODE)
 │    └── TensorValidationError (error_code: TENSOR_VALIDATION_ERROR)
 ├── InferenceError (error_code: INFERENCE_ERROR)
 │    ├── ModelUnavailableError (error_code: MODEL_UNAVAILABLE)
 │    ├── InvalidModelOutputError (error_code: INVALID_MODEL_OUTPUT)
 │    └── InferenceExecutionError (error_code: INFERENCE_EXECUTION_ERROR)
 ├── GradCAMError (error_code: GRADCAM_ERROR)
 │    ├── InvalidGradCAMInputError (error_code: INVALID_GRADCAM_INPUT)
 │    ├── GradCAMGradientError (error_code: GRADCAM_GRADIENT_ERROR)
 │    ├── GradCAMLayerNotFoundError (error_code: GRADCAM_LAYER_NOT_FOUND)
 │    └── GradCAMGenerationError (error_code: GRADCAM_GENERATION_ERROR)
 ├── VisualizerError (error_code: VISUALIZER_ERROR)
 │    └── InvalidVisualizerInputError (error_code: INVALID_VISUALIZER_INPUT)
 ├── MLCoordinatorError (error_code: ML_COORDINATOR_ERROR)
 │    └── MLNotReadyError (error_code: MODEL_NOT_READY)
 ├── ModelNotLoadedError (error_code: MODEL_UNAVAILABLE)
 └── ModelContractError (error_code: MODEL_CONTRACT_ERROR)
```

---

## 3. Machine-Readable Error Code & HTTP Status Mapping

The following table documents the exact mapping implemented in [`backend/app/core/errors.py`](file:///c:/Users/ASUS/OneDrive/Desktop/LungCare-AI/backend/app/core/errors.py):

| Exception Class | Domain Category | HTTP Status | Machine-Readable `error_type` | User-Facing Message (Production) |
| :--- | :--- | :---: | :--- | :--- |
| `EmptyImageError` | Image Preprocessing | **400** | `EMPTY_IMAGE` | `"Uploaded image byte sequence is empty."` |
| `CorruptImageError` | Image Preprocessing | **400** | `CORRUPT_IMAGE` | `"Cannot decode or verify image bytes. The image file appears corrupted or invalid."` |
| `InvalidImageModeError` | Image Preprocessing | **400** | `INVALID_IMAGE_MODE` | `"Cannot convert image mode to RGB."` |
| `ImageTooLargeError` | Image Preprocessing | **413** | `IMAGE_TOO_LARGE` | `"Uploaded image size (X bytes) exceeds the maximum allowed limit..."` |
| `DecompressionBombError` | Image Preprocessing | **413** | `DECOMPRESSION_BOMB` | `"Decompression bomb mitigation triggered..."` |
| `UnsupportedFormatError` | Image Preprocessing | **415** | `UNSUPPORTED_FORMAT` | `"Image format 'X' is not supported. Allowed formats: ('JPEG', 'PNG', 'WEBP')."` |
| `TensorValidationError` | Image Preprocessing | **500** | `TENSOR_VALIDATION_ERROR` | `"Image tensor validation failed."` |
| `ImagePreprocessingError` | Image Preprocessing | **400** | `IMAGE_PREPROCESSING_ERROR` | Dynamic validation message. |
| `InvalidGradCAMInputError` | Grad-CAM Input | **400** | `INVALID_GRADCAM_INPUT` | Dynamic validation message (e.g. index out of bounds, unrecognized class, invalid alpha). |
| `InvalidVisualizerInputError` | Visualizer Input | **400** | `INVALID_VISUALIZER_INPUT` | Dynamic input contract violation message. |
| `InvalidModelOutputError` | Inference Output | **502** | `INVALID_MODEL_OUTPUT` | `"Model output validation failed. Output violated integrity contracts."` |
| `GradCAMGradientError` | Grad-CAM Backprop | **502** | `GRADCAM_GRADIENT_ERROR` | `"Grad-CAM gradient backpropagation failed: ..."` |
| `ModelContractError` | Model Loader | **502** | `MODEL_CONTRACT_ERROR` | `"Model architecture contract validation failed."` |
| `ModelUnavailableError` | Inference Lifecycle | **503** | `MODEL_UNAVAILABLE` | `"DenseNet121 model is not loaded or temporarily unavailable. Please retry later."` |
| `ModelNotLoadedError` | Model Loader | **503** | `MODEL_UNAVAILABLE` | `"DenseNet121 model is not loaded or temporarily unavailable. Please retry later."` |
| `MLNotReadyError` | ML Coordinator | **503** | `MODEL_NOT_READY` | `"DenseNet121 model is not ready for execution. Please retry later."` |
| `MLCoordinatorError` | ML Coordinator | **503** | `ML_COORDINATOR_ERROR` | `"DenseNet121 model coordinator error. Please retry later."` |
| `InferenceExecutionError` | Inference Execution | **500** | `INFERENCE_EXECUTION_ERROR` | `"An unexpected error occurred during model inference."` |
| `InferenceError` | Inference Domain | **500** | `INFERENCE_ERROR` | `"An unexpected error occurred during model inference."` |
| `GradCAMLayerNotFoundError` | Grad-CAM Execution | **500** | `GRADCAM_LAYER_NOT_FOUND` | `"An unexpected error occurred during Grad-CAM explainability generation."` |
| `GradCAMGenerationError` | Grad-CAM Execution | **500** | `GRADCAM_GENERATION_ERROR` | `"An unexpected error occurred during Grad-CAM explainability generation."` |
| `GradCAMError` | Grad-CAM Domain | **500** | `GRADCAM_ERROR` | `"An unexpected error occurred during Grad-CAM explainability generation."` |
| `VisualizerError` | Overlay Generation | **500** | `VISUALIZER_ERROR` | `"An error occurred while rendering the visual explainability overlay."` |
| `StarletteHTTPException` | Framework Routing | *(Dynamic)* | `HTTPException` *(or custom code if dict detail)* | `str(exc.detail)` (e.g. `"Not Found"` for 404). |
| `RequestValidationError` | Pydantic Schema | **422** | `ValidationError` | `"Request validation failed."` *(sanitized field details)* |
| `Exception` | Generic Uncaught | **500** | `InternalServerError` | `"An unexpected server error occurred. Please contact system support."` |

---

## 4. Centralized Exception-Handling Architecture

The unified error architecture is implemented across three core modules:

```
[ Incoming Request ] ───► [ Endpoint ] ───► [ Service Raises Domain Exception ]
                                                          │
                                                          ▼
                                            [ FastAPI Exception Dispatcher ]
                                                          │
                                ┌─────────────────────────┴─────────────────────────┐
                                ▼                                                   ▼
                [ Specific Domain Handler ]                             [ Framework Exception Handler ]
                - image_preprocessing_exception_handler                  - http_exception_handler
                - inference_exception_handler                           - validation_exception_handler
                - gradcam_exception_handler                             - unhandled_exception_handler
                - visualizer_exception_handler
                - coordinator_exception_handler
                - model_loader_exception_handler
                                │                                                   │
                                └─────────────────────────┬─────────────────────────┘
                                                          ▼
                                            [ create_error_response() ]
                                            • get_request_id(request)
                                            • ErrorResponse(status='error', ...)
                                            • headers={'X-Request-ID': req_id}
                                                          │
                                                          ▼
                                                [ Outgoing JSONResponse ]
```

### Key Architectural Primitives

1. **Shared Error Response Factory (`create_error_response`):**
   ```python
   def create_error_response(
       request: Request,
       status_code: int,
       message: str,
       error_type: str,
       details: Optional[Any] = None,
       headers: Optional[Dict[str, str]] = None,
   ) -> JSONResponse:
       req_id = get_request_id(request)
       error_response = ErrorResponse(
           status="error",
           error=ErrorDetail(message=message, error_type=error_type, details=details),
           request_id=req_id,
       )
       response_headers = {"X-Request-ID": req_id}
       if headers:
           response_headers.update(headers)
       return JSONResponse(
           status_code=status_code,
           content=error_response.model_dump(mode="json"),
           headers=response_headers,
       )
   ```
2. **Subclass-First Resolution:**
   Resolvers evaluate specific subclasses before their base classes (e.g. `DecompressionBombError` before `ImageTooLargeError`, and both before `ImagePreprocessingError`).
3. **Clean Endpoint Elimination of Boilerplate:**
   `predict_chest_xray` and `explain_chest_xray` in `backend/app/api/v1/endpoints/predictions.py` no longer intercept domain exceptions to re-throw them as `HTTPException`. They allow domain exceptions to bubble naturally to the application-level handlers.
4. **Endpoint Input Normalization:**
   - Stream reading failure raises `CorruptImageError("Failed to read uploaded file payload.")` (status 400).
   - Empty upload raises `EmptyImageError("Uploaded image byte sequence is empty.")` (status 400).
   - Oversized upload raises `ImageTooLargeError(...)` (status 413).
   - Invalid target class or alpha raises `InvalidGradCAMInputError(...)` (status 400).

---

## 5. Security and Production Sanitization Verification

All error responses strictly comply with enterprise security constraints:
- **DEBUG=False Production Mode:**
  - Internal server errors (500) emit sanitized, generic messages:
    - `"An unexpected server error occurred. Please contact system support."`
    - `"An unexpected error occurred during model inference."`
    - `"An unexpected error occurred during Grad-CAM explainability generation."`
  - Zero Python exception class names, raw tracebacks, file paths, or memory addresses are included in API response payloads (`details=None`).
- **DEBUG=True Development Mode:**
  - Emits descriptive diagnostic messages (`str(exc)`) and populates `details={"exception_type": exc.__class__.__name__}`.
- **Log Privacy:**
  - Raw image bytes, Base64 strings, and payload buffers are never dumped to server logs.
  - Full exception tracebacks are logged exclusively on the server side via `logger.exception` along with correlation tracking `[%(request_id)s]`.

---

## 6. Files Created and Modified

| File Path | Action | Description |
| :--- | :---: | :--- |
| [`backend/app/core/errors.py`](file:///c:/Users/ASUS/OneDrive/Desktop/LungCare-AI/backend/app/core/errors.py) | **Created** | Centralized error response factory, domain exception resolvers, and FastAPI exception handlers. |
| [`backend/app/main.py`](file:///c:/Users/ASUS/OneDrive/Desktop/LungCare-AI/backend/app/main.py) | **Modified** | Replaced inline exception handlers with `register_exception_handlers(app)`. |
| [`backend/app/api/v1/endpoints/predictions.py`](file:///c:/Users/ASUS/OneDrive/Desktop/LungCare-AI/backend/app/api/v1/endpoints/predictions.py) | **Modified** | Replaced manual `try...except ... raise HTTPException` blocks with direct domain exception bubbling and input validation raises. |
| [`backend/tests/test_error_handling.py`](file:///c:/Users/ASUS/OneDrive/Desktop/LungCare-AI/backend/tests/test_error_handling.py) | **Created** | Comprehensive 32-test suite verifying domain error codes, HTTP statuses, request IDs, and security sanitization. |

---

## 7. Test Execution & Regression Results

### 7.1 Focused Error Handling Suite
**Command:** `pytest backend/tests/test_error_handling.py -v`  
- **Test Count:** 32 tests.
- **Result:** **32 / 32 passed in 0.74s (100%)**.

### 7.2 Complete Host Backend Test Suite
**Command:**
```powershell
& "C:\Users\ASUS\anaconda3\python.exe" -m pytest \
    backend/tests/test_concurrency_coordinator.py \
    backend/tests/test_configuration_and_security.py \
    backend/tests/test_explain_endpoint.py \
    backend/tests/test_health.py \
    backend/tests/test_prediction_endpoint.py \
    backend/tests/test_request_correlation.py \
    backend/tests/test_error_handling.py -v
```

**Breakdown by Module:**
- `test_concurrency_coordinator.py`: 14 passed
- `test_configuration_and_security.py`: 14 passed
- `test_explain_endpoint.py`: 25 passed
- `test_health.py`: 11 passed
- `test_prediction_endpoint.py`: 22 passed
- `test_request_correlation.py`: 12 passed
- `test_error_handling.py`: 32 passed
- **Result:** **130 unique host tests / 130 passed in 15.52s (100%)**.

### 7.3 Complete ML Backend Test Suite
**Command:**
```powershell
& "backend\.venv-ml\python.exe" -m pytest \
    backend/tests/test_model_loader.py \
    backend/tests/test_image_preprocessor.py \
    backend/tests/test_inference_service.py \
    backend/tests/test_gradcam_service.py \
    backend/tests/test_gradcam_visualizer.py -v
```

**Breakdown by Module:**
- `test_model_loader.py`: 12 passed
- `test_image_preprocessor.py`: 31 passed
- `test_inference_service.py`: 25 passed
- `test_gradcam_service.py`: 31 passed
- `test_gradcam_visualizer.py`: 19 passed
- **Result:** **118 unique ML tests / 118 passed in 30.15s (100%)**.

### 7.4 Grand Total Test Summary
- **Total Unique Tests:** **248 / 248 passed across environments (100%)**.
- **Regressions / Failures:** **0**.

### 7.5 Dependency & Environment Integrity
- Host Python 3.13: `python -m pip check` $\to$ **`No broken requirements found.`**
- ML Python 3.11: `backend\.venv-ml\python.exe -m pip check` $\to$ **`No broken requirements found.`**

---

## 8. Model Artifact Cryptographic Integrity (SHA-256)

All production model artifacts in `models/LungCare_AI_Model/` were cryptographically validated against the Phase 2.2.2 approved baseline:

| Artifact File Name | Phase 2.2.2 Baseline Hash | Current Workspace SHA-256 | Verification |
| :--- | :--- | :--- | :---: |
| `lungcare_config.json` | `6F2D60A977E0440DDED5B924FC7AFDD44B366364E060A2D4A7E16399BBF40456` | `6F2D60A977E0440DDED5B924FC7AFDD44B366364E060A2D4A7E16399BBF40456` | **EXACT MATCH** |
| `lungcare_densenet121.h5` | `AC2CA7D44A61D3F1AB43C9D53484CF55E06C966D2D0C724C519983F844A9671B` | `AC2CA7D44A61D3F1AB43C9D53484CF55E06C966D2D0C724C519983F844A9671B` | **EXACT MATCH** |
| `lungcare_densenet121.keras` | `B583CD87A5D551CE6920BA55985492176D6236AD9895082DBB6A688C0F574249` | `B583CD87A5D551CE6920BA55985492176D6236AD9895082DBB6A688C0F574249` | **EXACT MATCH** |
| `lungcare_evaluation.json` | `5239BAC95C1F1374A3D348573D0965836F89FC7FD8BC90E6F02F2BA912012392` | `5239BAC95C1F1374A3D348573D0965836F89FC7FD8BC90E6F02F2BA912012392` | **EXACT MATCH** |

Zero model files were touched, renamed, retrained, or regenerated.

---

## 9. Known Limitations & Remaining Roadmap

1. **OpenAPI Schema Harmonization (AUDIT-API-03):** Explicit route-level responses documentation in FastAPI (annotating 400, 413, 415, 422, 500, 502, 503 with their `ErrorResponse` schemas in Swagger UI) is scheduled for **Phase 2.3.4.4**.
2. **Duplicate Query/Form Parameter Cleanup (AUDIT-API-04):** The dual declaration of form parameters and query aliases (`target_class` / `alpha`) in `explain_chest_xray` remains functional for backward compatibility and is scheduled for cleanup in **Phase 2.3.4.5**.

---

## 10. Milestone Status

```
================================================================================
PHASE 2.3.4.3 — IMPLEMENTATION COMPLETE, READY FOR INDEPENDENT VERIFICATION
================================================================================
```
