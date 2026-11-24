# Phase 2.3.3.2 — Unified ML Execution Coordinator Implementation Report

**Project:** LungCare AI  
**Milestone:** Phase 2.3.3.2 — Unified ML Execution Coordinator Implementation  
**Auditor/Engineer Role:** Principal Python Backend Engineer (Concurrency & ML Platform)  
**Date:** November 2026  
**Status:** COMPLETE & VERIFIED  

---

## 1. Executive Summary

Phase 2.3.3.2 implements the unified concurrency and lifecycle architecture approved in `docs/phase_2_3_3_1_concurrency_audit.md`.

Prior to this milestone, `InferenceService` and `GradCAMService` operated with independent `threading.Lock` instances while executing against the same underlying Keras DenseNet121 model and layer objects. Furthermore, model unloading during application shutdown had no synchronization mechanism to wait for in-flight requests.

This milestone resolves these concurrency vulnerabilities by introducing the centralized `MLExecutionCoordinator` (`backend/app/services/ml_coordinator.py`). All TensorFlow/Keras forward and backward executions are now synchronized under a unified process-wide execution lock, while active operations are atomically tracked and model shutdown draining prevents premature memory deallocation.

All existing API schemas, error mappings, model artifact baselines, and test suites have been verified with a 100% pass rate.

**Milestone Verdict:** **PHASE 2.3.3.2 — IMPLEMENTATION COMPLETE, READY FOR AUDIT**

---

## 2. Implemented Architecture & Design

### A. Core Coordinator (`backend/app/services/ml_coordinator.py`)
The `MLExecutionCoordinator` provides centralized concurrency synchronization and lifecycle state management:
1. **Model Lifecycle States (`ModelLifecycleState`):**
   - `UNLOADED`: Model artifact is not loaded in memory.
   - `LOADING`: Model weights deserialization and contract validation in progress.
   - `READY`: Model validated and available for synchronized inference and explainability.
   - `DRAINING`: Server shutdown initiated; new ML executions rejected, active requests allowed to drain.
   - `ERROR`: Model loading, weight verification, or contract checks failed.
2. **Dual-Lock Architecture:**
   - `_state_lock` & `_state_condition` (`threading.Condition`): Guards state transitions and atomically increments/decrements `_active_operations`. **Never held during TensorFlow/Keras execution**, preventing lock-inversion deadlocks.
   - `_execution_lock` (`threading.Lock`): Shared across `InferenceService` and `GradCAMService` to serialize CPU forward passes and gradient backpropagation.
3. **Execution Context Manager (`acquire_execution`):**
   - Validates that state is `READY` under `_state_lock`. If not ready (e.g. `DRAINING` or `LOADING`), raises `MLNotReadyError`.
   - Atomically increments `_active_operations`.
   - Releases `_state_lock` before acquiring `_execution_lock`.
   - Yields control inside `with _execution_lock:`.
   - In `finally`: releases `_execution_lock`, acquires `_state_condition`, decrements `_active_operations`, and calls `notify_all()` when the counter reaches 0.

### B. Service Integrations

1. **`ModelLoader` (`backend/app/services/model_loader.py`):**
   - Coordinates state transitions: invokes `start_loading()` at start, `complete_loading()` upon validation success, and `record_loading_error()` on failure.
   - Model loading executes under its loader lock without unnecessarily holding the ML execution lock.
   - `unload()` verifies `active_operations == 0` (or waits up to `drain_timeout`) before deallocating model references and transitioning to `UNLOADED`.
2. **`InferenceService` (`backend/app/services/inference_service.py`):**
   - Removed independent `self._inference_lock`.
   - Uses `with self._coordinator.acquire_execution():` exclusively for `model(tensor, training=False)` and NumPy conversion.
   - Image preprocessing and output validation remain completely outside the lock.
   - Maps `MLNotReadyError` to `ModelUnavailableError`, returning HTTP 503 `status.HTTP_503_SERVICE_UNAVAILABLE`.
3. **`GradCAMService` (`backend/app/services/gradcam_service.py`):**
   - Removed independent `self._lock`.
   - Uses `with self._coordinator.acquire_execution():` for gradient model retrieval, forward pass, and `tf.GradientTape` backpropagation.
   - Feature map validation, GAP pooling, ReLU, upsampling, and overlay rendering remain outside the lock.
   - Maps `MLNotReadyError` to `ModelUnavailableError` (HTTP 503).
4. **Dependency Injection (`backend/app/api/deps.py`):**
   - Exports `get_ml_coordinator()` and `get_ml_coordinator_dep()`.
   - Injects the identical singleton coordinator into `ModelLoader`, `InferenceService`, and `GradCAMService`.
5. **Lifespan Shutdown (`backend/app/main.py`):**
   - Initiates shutdown by calling `coordinator.start_draining()`.
   - Offloads bounded wait (`wait_for_active_operations(timeout=5.0)`) via `run_in_threadpool` to prevent blocking the asyncio event loop.
   - If timeout expires while requests remain active, deallocation is skipped to prevent native segmentation faults.

---

## 3. Concurrency Invariants & Scoping Table

| Component | Synchronization Primitive | Protected Scope | Unlocked Operations |
| :--- | :--- | :--- | :--- |
| **`MLExecutionCoordinator` (State)** | `_state_condition` (`threading.Condition`) | State transitions, `_active_operations` counter. | Entire TensorFlow/Keras execution. |
| **`MLExecutionCoordinator` (ML Exec)** | `_execution_lock` (`threading.Lock`) | `model(tensor)` forward pass, `GradientTape` backpropagation. | Preprocessing, contract validation, overlay rendering. |
| **`ModelLoader`** | `self._lock` (`threading.Lock`) | Double-checked load, artifact read, contract verification. | Subsequent model inference calls. |
| **`ImagePreprocessor`** | *None* | Pure stateless / reentrant operations. | 100% unlocked (concurrent across worker threads). |
| **`GradCAMVisualizer`** | *None* | Pure stateless / reentrant operations. | 100% unlocked (concurrent across worker threads). |

---

## 4. Multi-Worker & Performance Considerations

1. **Intra-Process vs. Multi-Worker Scope:**  
   Python `threading.Lock` coordinates threads within a single OS process. When running with Uvicorn multi-worker mode (`--workers N`), each worker process manages its own independent coordinator, memory space, and model instance. Intra-process thread safety is guaranteed by this coordinator; cross-process isolation is provided by the OS.
2. **Throughput Trade-Off:**  
   TensorFlow Keras operations on CPU utilize multi-threaded BLAS/OpenMP backends. Serializing model execution under a single lock eliminates internal Python layer state races and thread contention with negligible throughput impact on CPU hardware (~35–65 ms inference, ~80–120 ms Grad-CAM). Concurrent image preprocessing (10–25 ms) is preserved outside the lock, allowing pipeline overlapping.

---

## 5. Verification & Test Results

### A. Concurrency Coordinator Test Suite (`test_concurrency_coordinator.py`)
12 focused deterministic tests using `threading.Event` and `threading.Barrier` without flaky sleep polling:

| # | Test Name | Objective | Result |
| :-: | :--- | :--- | :---: |
| 1 | `test_initial_state_and_lifecycle_transitions` | Verify states through LOADING, READY, ERROR, UNLOADED. | **PASS** |
| 2 | `test_operation_rejected_when_loading` | Verify operations are rejected during LOADING. | **PASS** |
| 3 | `test_operation_rejected_when_draining` | Verify operations are rejected during DRAINING. | **PASS** |
| 4 | `test_two_simultaneous_predictions_serialize` | Verify two concurrent prediction calls serialize under lock. | **PASS** |
| 5 | `test_prediction_and_gradcam_share_same_execution_lock` | Verify both services share identical lock instance. | **PASS** |
| 6 | `test_multiple_gradcam_operations_serialize` | Verify multiple Grad-CAM operations serialize cleanly. | **PASS** |
| 7 | `test_exception_inside_execution_releases_lock_and_decrements_counter` | Verify lock is released and counter restored on error. | **PASS** |
| 8 | `test_subsequent_request_succeeds_after_operation_fails` | Verify system recovers immediately after failed request. | **PASS** |
| 9 | `test_concurrent_attempts_cannot_corrupt_counter` | Verify counter integrity under 20 concurrent threads. | **PASS** |
| 10 | `test_shutdown_waits_for_active_operation` | Verify shutdown blocks until active operation finishes. | **PASS** |
| 11 | `test_model_unloading_happens_only_after_operations_reach_zero` | Verify model unload is refused while operations active. | **PASS** |
| 12 | `test_inference_service_maps_coordinator_rejection_to_model_unavailable`| Verify coordinator rejection maps to ModelUnavailableError. | **PASS** |

### B. ML Environment Test Suite (Python 3.11 / TensorFlow 2.17.1)
```
Command: backend/.venv-ml/python.exe -m pytest backend/tests/test_model_loader.py backend/tests/test_image_preprocessor.py backend/tests/test_inference_service.py backend/tests/test_gradcam_service.py backend/tests/test_gradcam_visualizer.py -v
Result: 118 passed in 27.69s (100% pass rate)
```

### C. Host Backend Test Suite (Python 3.13)
```
Command: python -m pytest backend/tests/test_concurrency_coordinator.py backend/tests/test_configuration_and_security.py backend/tests/test_explain_endpoint.py backend/tests/test_health.py backend/tests/test_prediction_endpoint.py -v
Result: 84 passed in 14.64s (100% pass rate)
```

### D. Package Integrity & Dependency Audit
- `pip check` (Host): `No broken requirements found.`
- `pip check` (ML venv): `No broken requirements found.`

### E. Model Artifact SHA-256 Hashes
All four model artifacts in `models/LungCare_AI_Model/` match approved baseline hashes bit-for-bit:
- `lungcare_config.json`: `6F2D60A977E0440DDED5B924FC7AFDD44B366364E060A2D4A7E16399BBF40456` (**MATCH**)
- `lungcare_densenet121.h5`: `AC2CA7D44A61D3F1AB43C9D53484CF55E06C966D2D0C724C519983F844A9671B` (**MATCH**)
- `lungcare_densenet121.keras`: `B583CD87A5D551CE6920BA55985492176D6236AD9895082DBB6A688C0F574249` (**MATCH**)
- `lungcare_evaluation.json`: `5239BAC95C1F1374A3D348573D0965836F89FC7FD8BC90E6F02F2BA912012392` (**MATCH**)

### F. Lazy Loading Verification
Verified that importing `app.api.deps` or endpoint router modules does not initialize TensorFlow or load model weights (`ModelLoader.is_loaded is False`, `coordinator.state == UNLOADED`).

---

## 6. Deviations from Approved Architecture

**Zero Deviations.** The implementation follows the exact design specified in `docs/phase_2_3_3_1_concurrency_audit.md`.

---

## 7. Milestone Verdict

**PHASE 2.3.3.2 — IMPLEMENTATION COMPLETE, READY FOR AUDIT**
