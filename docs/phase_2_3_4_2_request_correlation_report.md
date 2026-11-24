# Phase 2.3.4.2 — Request Correlation & Contextual Logging Implementation Report

**Project:** LungCare AI  
**Milestone:** Phase 2.3.4.2 — Request Correlation & Contextual Logging  
**Engineer Role:** Principal Python Backend Engineer (ASGI Middleware & Observability Specialist)  
**Date:** November 2026  
**Status:** Implementation Complete, Ready for Independent Verification  

---

## 1. Executive Summary

Phase 2.3.4.2 addresses **AUDIT-API-01** from the Phase 2.3.4.1 architecture audit by implementing enterprise request correlation across the LungCare AI backend.

Prior to this milestone, log records produced during concurrent API requests were interleaved with no mechanism to associate log messages with specific HTTP transactions. Furthermore, clients received responses with no tracking headers, preventing distributed tracing and client-side bug reporting.

The implementation introduces:
1. A thread-safe, async-safe request context manager utilizing standard Python `contextvars.ContextVar`.
2. A pure ASGI `RequestCorrelationMiddleware` that extracts, sanitizes, or generates unique correlation IDs and guarantees `X-Request-ID` propagation in response headers across all paths.
3. Centralized logging integration hooking into `logging.setLogRecordFactory` and a custom `RequestContextFilter`, formatting all log records with `[%(request_id)s]`.
4. Schema and exception handling integration adding `request_id` to the enterprise `ErrorResponse` envelope.

---

## 2. Architecture & Files Changed

```
                                [ Incoming Request ]
                                         │
                                         ▼
                        [ RequestCorrelationMiddleware ]
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   │ Extract & Validate X-Request-ID Header    │
                   │ Regex: ^[a-zA-Z0-9_-]{1,64}$              │
                   │ (Generate UUID4 if missing/invalid/long)   │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                        [ contextvars.set(request_id) ]
                                         │
                  ┌──────────────────────┴──────────────────────┐
                  ▼                                             ▼
        [ Contextual Logging ]                        [ Request Handlers ]
     - LogRecordFactory sets                          - Ingress & threadpool
       record.request_id                                (run_in_threadpool)
     - LOG_FORMAT: [%(request_id)s]                   - Exception Handlers
     - Fallback: '[-]                                   (404, 422, 500)
                  │                                             │
                  └──────────────────────┬──────────────────────┘
                                         ▼
                       [ Outgoing Response Interceptor ]
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   │ Inject X-Request-ID into HTTP Headers     │
                   │ Populate request_id in ErrorResponse JSON │
                   │ Reset ContextVar via Token in finally     │
                   └───────────────────────────────────────────┘
```

### 1. `backend/app/core/request_context.py` (New Module)
- Centralizes request-scoped context tracking with `contextvars.ContextVar[str]`.
- Implements `is_valid_request_id(request_id: Optional[str]) -> bool` using regex `^[a-zA-Z0-9_-]{1,64}$`.
- Implements `generate_request_id() -> str` generating UUID4 strings.
- Implements `get_request_id(request: Optional[Any] = None) -> str` returning the active ID, reading from `contextvars`, falling back to `request.scope['state']['request_id']`, or returning `"-"`.
- Implements `set_request_id(request_id: str) -> Token[str]` and `reset_request_id(token: Token[str]) -> None`.

### 2. `backend/app/middleware/request_correlation.py` (New Module)
- Pure ASGI middleware (`RequestCorrelationMiddleware`).
- Bypasses non-HTTP scopes (e.g. `lifespan`).
- Extracts incoming `X-Request-ID` from ASGI scope raw headers.
- Validates candidate string; if missing, empty, containing disallowed characters, or exceeding 64 characters, replaces it with a generated UUID4.
- Binds effective ID to `scope["state"]["request_id"]` and `set_request_id(effective_request_id)`.
- Intercepts `http.response.start` messages to append `(b"x-request-id", effective_request_id.encode("latin1"))`.
- Restores previous `ContextVar` state in a `finally:` block using the token.

### 3. `backend/app/core/logging.py` (Modified)
- Installs a global LogRecord factory hook via `logging.setLogRecordFactory`. Every LogRecord created anywhere in Python automatically has `record.request_id = get_request_id()` assigned at instantiation.
- Updates `LOG_FORMAT`:
  `LOG_FORMAT = "%(asctime)s.%(msecs)03d | %(levelname)-8s | [%(request_id)s] | %(name)s:%(funcName)s:%(lineno)d - %(message)s"`
- Adds `RequestContextFilter` to root logger and stream handler as defense in depth.

### 4. `backend/app/schemas/error.py` (Modified)
- Adds optional `request_id: Optional[str] = Field(default=None, description="...")` to `ErrorResponse`.
- Preserves all existing fields (`status`, `error`, `timestamp`).

### 5. `backend/app/main.py` (Modified)
- Registers `app.add_middleware(RequestCorrelationMiddleware)` immediately after `CORSMiddleware`.
- Updates `http_exception_handler`, `validation_exception_handler`, and `unhandled_exception_handler` to pass `request_id=get_request_id(request)` to `ErrorResponse` and include `headers={"X-Request-ID": req_id}` on the returned `JSONResponse`.
- In `unhandled_exception_handler`, re-binds `set_request_id(req_id)` inside a `try...finally` block so that `logger.exception` emitted during unhandled 500 errors logs with the request ID.

### 6. `backend/tests/test_request_correlation.py` (New Test Suite)
- 12 comprehensive unit and integration tests verifying all validation, header injection, logging, isolation, reset, and exception scenarios.

---

## 3. Request ID Lifecycle & Validation Behavior

1. **Format Validation:**
   - Evaluated against regex `^[a-zA-Z0-9_-]{1,64}$`.
   - Rejects special characters (`<`, `>`, `\s`, `$`, `;`, `/`, etc.) preventing log injection and HTTP response header splitting.
   - Rejects empty strings and strings exceeding 64 characters.
2. **Deterministic Fallback:**
   - If an incoming header is absent or invalid, a standard UUID4 (36 characters) is generated.
   - For logs produced outside an HTTP request context (e.g. startup, model pre-warming, shutdown, or background tasks), `get_request_id()` returns `"-"`.
3. **Context Isolation:**
   - Python `contextvars` natively provides copy-on-write isolation across concurrent asyncio tasks.
   - Because `run_in_threadpool` in Starlette/AnyIO automatically propagates `contextvars.copy_context()`, inference and explainability operations offloaded to worker threads inherit the active `request_id` in logs.
4. **Guaranteed Reset:**
   - The middleware wraps request execution in a `try...finally:` block, calling `reset_request_id(token)`.
   - Even if route execution raises an unhandled exception or is cancelled by the client, the ContextVar is restored to its prior state.

---

## 4. Middleware Ordering Analysis

In FastAPI / Starlette, middleware added via `app.add_middleware` wraps the application in an onion architecture:

```
[ Ingress ]
    │
    ▼
[ RequestCorrelationMiddleware ]  <-- Extracts/generates ID, sets ContextVar
    │
    ▼
[ CORSMiddleware ]                <-- Evaluates origin whitelist & CORS preflight
    │
    ▼
[ Exception Handlers & Routers ]  <-- Executes endpoint logic with [%(request_id)s]
    │
    ▼
[ CORSMiddleware ]                <-- Injects CORS headers
    │
    ▼
[ RequestCorrelationMiddleware ]  <-- Injects X-Request-ID header & resets ContextVar
    │
    ▼
[ Egress ]
```

When an exception occurs:
- If handled by `http_exception_handler` or `validation_exception_handler`, the handler explicitly sets `headers={"X-Request-ID": req_id}` and `error_response.request_id = req_id`.
- If an unhandled exception bubbles up to Starlette's `ServerErrorMiddleware`, `unhandled_exception_handler` extracts `req_id` from `request.scope["state"]["request_id"]`, attaches it to `ErrorResponse` and response headers, and re-binds context for `logger.exception`.

---

## 5. Regression & Verification Results

### 1. Focused Request Correlation Suite (Python 3.13)
Executed: `pytest backend/tests/test_request_correlation.py -v`
- `test_missing_request_id_generates_valid_uuid4`: PASSED
- `test_valid_incoming_request_id_is_preserved`: PASSED
- `test_invalid_characters_request_id_is_replaced`: PASSED
- `test_empty_request_id_is_replaced`: PASSED
- `test_excessively_long_request_id_is_replaced`: PASSED
- `test_response_header_contains_effective_request_id`: PASSED
- `test_application_logs_contain_correct_request_id`: PASSED
- `test_concurrent_requests_do_not_leak_ids_into_one_another`: PASSED
- `test_contextvar_is_reset_after_request_completion`: PASSED
- `test_exception_response_preserves_request_id`: PASSED
- `test_validation_error_response_preserves_request_id`: PASSED
- `test_outside_http_context_uses_fallback_id`: PASSED
- **Result:** **12 / 12 passed in 0.54s (100%)**

### 2. Complete Host Backend Test Suite (Python 3.13)
Executed: `pytest backend/tests/test_concurrency_coordinator.py backend/tests/test_configuration_and_security.py backend/tests/test_explain_endpoint.py backend/tests/test_health.py backend/tests/test_prediction_endpoint.py backend/tests/test_request_correlation.py -v`
- `test_concurrency_coordinator.py`: 14 passed
- `test_configuration_and_security.py`: 14 passed
- `test_explain_endpoint.py`: 25 passed
- `test_health.py`: 11 passed
- `test_prediction_endpoint.py`: 22 passed
- `test_request_correlation.py`: 12 passed
- **Result:** **98 / 98 passed in 15.40s (100%)**

### 3. Complete ML Backend Test Suite (Python 3.11 / TensorFlow 2.17.1)
Executed: `backend/.venv-ml/python.exe -m pytest backend/tests/test_model_loader.py backend/tests/test_image_preprocessor.py backend/tests/test_inference_service.py backend/tests/test_gradcam_service.py backend/tests/test_gradcam_visualizer.py -v`
- `test_model_loader.py`: 12 passed
- `test_image_preprocessor.py`: 31 passed
- `test_inference_service.py`: 25 passed
- `test_gradcam_service.py`: 31 passed
- `test_gradcam_visualizer.py`: 19 passed
- **Result:** **118 / 118 passed in 26.69s (100%)**

### 4. Grand Total Regression
- **Total Executed Tests:** **216 / 216 passed (100%)**
- **Failures / Errors:** **0**

### 5. Dependency & Environment Integrity
- Host (`python -m pip check`): `No broken requirements found.`
- ML environment (`backend/.venv-ml/python.exe -m pip check`): `No broken requirements found.`

### 6. Model Artifact Cryptographic Integrity (SHA-256)
- `lungcare_config.json`: `6F2D60A977E0440DDED5B924FC7AFDD44B366364E060A2D4A7E16399BBF40456` (**MATCH**)
- `lungcare_densenet121.h5`: `AC2CA7D44A61D3F1AB43C9D53484CF55E06C966D2D0C724C519983F844A9671B` (**MATCH**)
- `lungcare_densenet121.keras`: `B583CD87A5D551CE6920BA55985492176D6236AD9895082DBB6A688C0F574249` (**MATCH**)
- `lungcare_evaluation.json`: `5239BAC95C1F1374A3D348573D0965836F89FC7FD8BC90E6F02F2BA912012392` (**MATCH**)

---

## 6. Remaining Limitations

- Request correlation is now fully operational across headers, logs, and error responses.
- As scheduled in the audit roadmap, domain error code harmonization (mapping service `error_code` into `error_type` instead of `"HTTPException"`) will be executed in Phase 2.3.4.3.
- OpenAPI parameter normalization (resolving query/form parameter duplication in `/explain`) will be addressed in Phase 2.3.4.4.

---

## 7. Status

**PHASE 2.3.4.2 — IMPLEMENTATION COMPLETE, READY FOR INDEPENDENT VERIFICATION**
