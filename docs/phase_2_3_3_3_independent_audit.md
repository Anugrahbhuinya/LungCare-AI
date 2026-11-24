# Phase 2.3.3.3 — Independent Concurrency Coordinator Security & Architecture Audit

**Project:** LungCare AI  
**Milestone:** Phase 2.3.3.3 — Independent Concurrency Coordinator Security & Architecture Audit  
**Auditor Role:** Independent Principal Python Concurrency Engineer & TensorFlow/Keras Infrastructure Auditor  
**Date:** November 2026  
**Investigation Mode:** Strictly Read-Only (Zero code, test, configuration, model, or dependency modifications)  
**Evaluated Documents:**
- Architecture Approved: `docs/phase_2_3_3_1_concurrency_audit.md`
- Implementation Report: `docs/phase_2_3_3_2_implementation_report.md`

---

## 1. Executive Summary & Audit Matrix

An independent source-code-level concurrency, security, and lifecycle verification was performed on the Phase 2.3.3.2 implementation of the `MLExecutionCoordinator` and its integration with `ModelLoader`, `InferenceService`, `GradCAMService`, and FastAPI `lifespan`.

The core architecture succeeds in unifying TensorFlow execution under a single process-wide lock, eliminating the critical lock fragmentation vulnerability from earlier milestones, tracking in-flight operations, and cleanly separating stateless preprocessing from serialized neural network execution.

However, a race condition was identified in **Scenario E (Loading and Shutdown Race)** where `complete_loading()` in `backend/app/services/ml_coordinator.py#L124-L129` sets `self._state = ModelLifecycleState.READY` unconditionally. If application shutdown is triggered while a model load is in progress, the coordinator is placed in `DRAINING`. When loading finishes, `complete_loading()` overwrites `DRAINING` back to `READY`, violating the lifecycle invariant that a draining subsystem must not return to ready.

| Audit Item / Scenario | Severity | Status | Exact Source File & Lines | Summary / Evidence |
| :--- | :---: | :---: | :--- | :--- |
| **1. Singleton Coordinator Identity** | High | **PASS** | `ml_coordinator.py#L240-L251`, `deps.py#L38-L41` | Exactly one coordinator singleton shared across `ModelLoader`, `InferenceService`, and `GradCAMService`. |
| **1. Unified Process Execution Lock** | Critical | **PASS** | `ml_coordinator.py#L72, L219` | Exactly one `threading.Lock` per process synchronizes both inference and Grad-CAM. |
| **1. Removal of Fragmented Locks** | Critical | **PASS** | `inference_service.py#L190-L205`, `gradcam_service.py#L148-L163` | Obsolete `self._inference_lock` and `self._lock` removed completely. |
| **1. State Lock Scoping** | Critical | **PASS** | `ml_coordinator.py#L203-L227` | `_state_lock` released before acquiring `_execution_lock`. Condition never held during TensorFlow runs. |
| **1. Exception-Safe Lock Release** | Critical | **PASS** | `ml_coordinator.py#L217-L227` | `finally` block guarantees `_execution_lock` release and atomic `_active_operations` decrement. |
| **2. Scenario A: Queued Request Lock Wait** | High | **PASS** | `ml_coordinator.py#L215-L220` | Request B increments counter before waiting on execution lock; shutdown counts both A and B. |
| **2. Scenario B: Shutdown Admission Race** | High | **PASS** | `ml_coordinator.py#L138-L146, L203-L215` | Serialized under `_state_lock`. Request is either admitted and counted or cleanly rejected with 503. |
| **2. Scenario C: Exception During Execution** | High | **PASS** | `ml_coordinator.py#L221-L227`, `test_concurrency_coordinator.py#L255-L272` | Execution lock released and active operations decremented exactly once upon exception. |
| **2. Scenario D: Concurrent Unload Safety** | Medium | **PASS** | `model_loader.py#L173-L203`, `ml_coordinator.py#L169-L184` | `ModelLoader.unload()` protected by `self._lock`; refuses deallocation if operations active. |
| **2. Scenario E: Loading and Shutdown Race** | **Medium** | **FAIL** | `ml_coordinator.py#L124-L129` | **Defect:** `complete_loading()` overwrites `DRAINING` to `READY` if shutdown triggers while loading. |
| **2. Scenario F: Bounded Timeout Safety** | High | **PASS** | `main.py#L74-L85`, `model_loader.py#L182-L193` | Bounded shutdown timeout skips model deallocation if active operations remain. |
| **3. Lock Ordering & Deadlock Freedom** | Critical | **PASS** | System-wide lock hierarchy | Acyclic lock hierarchy: `ModelLoader._lock` -> `_state_lock`; `_execution_lock` disjoint from `_state_lock`. |
| **4. Concurrency Test Quality** | Medium | **PASS** | `test_concurrency_coordinator.py` | 12 deterministic tests using `threading.Event`/`Barrier`; zero hanging threads. |
| **5. Model Artifact & API Integrity** | Critical | **PASS** | `models/LungCare_AI_Model/`, `predictions.py` | All 4 SHA-256 hashes match baseline bit-for-bit; API contracts and Grad-CAM layers untouched. |
| **6. Full Regression Suites** | High | **PASS** | Host suite (84/84), ML suite (118/118) | **202 / 202 tests passing (100%)**; `pip check` clean in both environments. |

---

## 2. Detailed Source Audit Findings

### 1. Coordinator Singleton & Unified Lock Ownership
- **File:** `backend/app/services/ml_coordinator.py#L240-L251`
- **File:** `backend/app/api/deps.py#L38-L41`
- **Verification:**
  `get_ml_coordinator()` implements double-checked locking over `_singleton_lock` returning a persistent `MLExecutionCoordinator` instance.
  Inspecting service constructors:
  - `ModelLoader.__init__` (`model_loader.py#L65-L67`): defaults to `get_ml_coordinator()`.
  - `InferenceService.__init__` (`inference_service.py#L206-L208`): defaults to `get_ml_coordinator()`.
  - `GradCAMService.__init__` (`gradcam_service.py#L160-L162`): defaults to `get_ml_coordinator()`.
  Confirmed: All components share the exact same coordinator singleton (`assert loader.coordinator is inference.coordinator is gradcam.coordinator`).
- **Verdict:** **PASS**

### 2. Lock Fragmentation Elimination
- **File:** `backend/app/services/inference_service.py#L274-L295`
- **File:** `backend/app/services/gradcam_service.py#L327-L352`
- **Verification:**
  `InferenceService._inference_lock` has been completely eliminated. `predict_tensor()` executes forward pass inside `with self._coordinator.acquire_execution():`.
  `GradCAMService._lock` has been completely eliminated. `generate_heatmap()` executes gradient graph building, forward pass, and `GradientTape` backpropagation inside `with self._coordinator.acquire_execution():`.
  No private locks exist on either service. Both services acquire `coordinator.execution_lock`.
- **Verdict:** **PASS**

### 3. State Lock & Condition Scoping (No Lock Inversion)
- **File:** `backend/app/services/ml_coordinator.py#L202-L227`
- **Verification:**
  ```python
  # Step 1: Validate state and increment counter under state lock
  with self._state_lock:
      if self._state != ModelLifecycleState.READY:
          raise MLNotReadyError(...)
      self._active_operations += 1

  try:
      # Step 2: Acquire execution lock (OUTSIDE state lock)
      with self._execution_lock:
          yield
  finally:
      # Step 3: Decrement counter under state condition
      with self._state_condition:
          self._active_operations -= 1
          if self._active_operations == 0:
              self._state_condition.notify_all()
  ```
  `_state_lock` is explicitly released before `_execution_lock` is acquired.
  During the execution of `model(tensor)` and `tape.gradient()`, only `_execution_lock` is held. The condition lock is never held during TensorFlow computations.
- **Verdict:** **PASS**

---

## 3. Critical Race-Condition Tracing

### Scenario A — Request Waiting for Execution Lock
- **Trace:** Request A enters `acquire_execution()`, increments `_active_operations = 1`, acquires `_execution_lock`, runs forward pass. Request B enters `acquire_execution()`, increments `_active_operations = 2`, attempts to acquire `_execution_lock`, and blocks. Shutdown begins and calls `coordinator.start_draining()`, followed by `wait_for_active_operations()`.
- **Analysis:** `wait_for_active_operations()` inspects `self._active_operations`. Because Request B incremented the counter *before* waiting on `_execution_lock`, `_active_operations` equals 2. Shutdown condition `_active_operations == 0` is false; shutdown waits on `_state_condition`.
- **Conclusion:** Both active and queued requests are counted. The model cannot be unloaded while a queued request is waiting for the execution lock.
- **Verdict:** **PASS**

### Scenario B — Shutdown Admission Race
- **Trace:** A request invokes `acquire_execution()` concurrently with `start_draining()`.
- **Analysis:** Both methods acquire `self._state_lock`.
  - If request enters `_state_lock` first: State is `READY`, counter increments to 1, exits lock. `start_draining()` acquires `_state_lock`, sets `DRAINING`. Request is safely counted and permitted to finish.
  - If `start_draining()` enters first: State is set to `DRAINING`, exits lock. Request acquires `_state_lock`, checks `self._state != READY`, raises `MLNotReadyError(DRAINING)`, counter is not incremented.
- **Conclusion:** Admission is strictly atomic. A request cannot be admitted without being counted.
- **Verdict:** **PASS**

### Scenario C — Exception During Execution
- **Trace:** A hardware fault or invalid tensor causes `model(tensor)` or `tape.gradient()` to raise `InferenceExecutionError` or `GradCAMGenerationError` inside `yield`.
- **Analysis:** Python's context manager machinery executes `__exit__` on `_execution_lock`, releasing the lock immediately. The `finally:` block executes reacquiring `_state_condition`, decrements `_active_operations` by 1, and notifies waiting threads if 0.
- **Conclusion:** Counter decrement and lock release are guaranteed. Verified by `test_exception_inside_execution_releases_lock_and_decrements_counter` and `test_subsequent_request_succeeds_after_operation_fails`.
- **Verdict:** **PASS**

### Scenario D — Concurrent Unload
- **Trace:** Concurrent invocations of `ModelLoader.unload()`.
- **Analysis:** `ModelLoader.unload()` is serialized under `self._lock`. If `active_operations > 0`, it calls `wait_for_active_operations()`. If operations fail to drain, deallocation is aborted. `complete_unloading()` under `_state_lock` validates `active_operations == 0` before setting `UNLOADED`.
- **Conclusion:** Idempotent and thread-safe; model references are never cleared while operations execute.
- **Verdict:** **PASS**

### Scenario E — Loading and Shutdown Race (CRITICAL FINDING)
- **Trace:** Model loading is triggered on startup or reload. `ModelLoader.load_model()` invokes `coordinator.start_loading()`, transitioning state to `LOADING`. While Keras deserializes model weights from disk, the server receives a termination signal and executes lifespan shutdown, invoking `coordinator.start_draining()`, setting state to `DRAINING`. Once weight loading completes, `ModelLoader.load_model()` invokes `coordinator.complete_loading()`.
- **Analysis:** Inspecting `backend/app/services/ml_coordinator.py#L124-L129`:
  ```python
  def complete_loading(self) -> None:
      """Signal that model loading and contract validation completed successfully."""
      with self._state_lock:
          self._state = ModelLifecycleState.READY
          self._last_error = None
          logger.info("ML lifecycle transition -> READY")
  ```
  `complete_loading()` does **not** check whether `self._state == ModelLifecycleState.DRAINING`.
  It unconditionally overwrites `DRAINING` back to `READY`.
- **Consequence:** An application undergoing shutdown would be erroneously re-opened to accepting new ML inference requests because the finished background load overwrote `DRAINING` with `READY`.
- **Severity:** **MEDIUM**
- **Verdict:** **FAIL — REMEDIATION REQUIRED**

### Scenario F — Bounded Timeout
- **Trace:** Shutdown initiates with active operations. Draining wait timeout (5.0s) expires while operations remain active.
- **Analysis:** In `backend/app/main.py#L74-L83`:
  ```python
  drained = await run_in_threadpool(coordinator.wait_for_active_operations, timeout=drain_timeout)
  if not drained:
      logger.error(
          "Shutdown drain timed out after %.1f seconds with %d active operations remaining. "
          "Retaining model memory reference to prevent segmentation faults during in-flight execution.",
          drain_timeout,
          coordinator.active_operations,
      )
      return
  ```
  And in `ModelLoader.unload()` (`model_loader.py#L187-L192`):
  If `not drained`, it logs an error and executes `return` without clearing `self._model = None`.
- **Conclusion:** Timeout safely preserves model references in memory, preventing dangling pointer segmentation faults.
- **Verdict:** **PASS**

---

## 4. Lock Ordering & Deadlock Analysis

### Subsystem Lock Registry
1. `MLExecutionCoordinator._state_lock` (Internal condition lock for state and counter)
2. `MLExecutionCoordinator._execution_lock` (Process-wide ML execution mutual exclusion lock)
3. `ModelLoader._lock` (Model loader re-entrancy lock)

### Acquisition Hierarchy
```
Level 1: ModelLoader._lock
   │
   └──> Level 2: MLExecutionCoordinator._state_lock
           │
           └──> (Explicitly released before Level 3)
                   │
                   └──> Level 3: MLExecutionCoordinator._execution_lock
```

- **Invariant 1:** `_state_lock` and `_execution_lock` are **never held simultaneously**.
- **Invariant 2:** `ModelLoader._lock` is held only during disk loading or unloading; it never acquires `_execution_lock`.
- **Invariant 3:** No execution path acquires `ModelLoader._lock` while holding `_state_lock` or `_execution_lock`.
- **Invariant 4:** FastAPI async event loop is never blocked synchronously; blocking waits use `run_in_threadpool`.
- **Verdict:** **PASS (Deadlock Free)**

---

## 5. Test Quality & Suite Audit

Inspecting `backend/tests/test_concurrency_coordinator.py`:
- **Deterministic Synchronization:** Tests 1–12 use `threading.Event`, `threading.Barrier`, and bounded timeouts. Zero unbounded while-loops.
- **Lock Contention:** Test 4 (`test_two_simultaneous_predictions_serialize`) and Test 6 (`test_multiple_gradcam_operations_serialize`) prove serial execution and mutual exclusion under multi-threading.
- **Integration Coverage:** Test 5 verifies that `InferenceService` and `GradCAMService` share the identical coordinator execution lock.
- **Test Gap Identified:** There is currently no unit test asserting that `complete_loading()` respects `DRAINING` state during a concurrent load/shutdown race.

---

## 6. Regression & Integrity Verification

1. **Host Backend Regression Suite (Python 3.13):**
   - `test_concurrency_coordinator.py` (12 tests)
   - `test_configuration_and_security.py` (14 tests)
   - `test_explain_endpoint.py` (25 tests)
   - `test_health.py` (11 tests)
   - `test_prediction_endpoint.py` (22 tests)
   - **Result:** **84 / 84 passed in 14.64s (100%)**
2. **ML Regression Suite (Python 3.11 / TensorFlow 2.17.1):**
   - `test_model_loader.py` (12 tests)
   - `test_image_preprocessor.py` (31 tests)
   - `test_inference_service.py` (25 tests)
   - `test_gradcam_service.py` (31 tests)
   - `test_gradcam_visualizer.py` (19 tests)
   - **Result:** **118 / 118 passed in 27.69s (100%)**
3. **Grand Total Regression:** **202 / 202 passed (100%)**. Failures: 0.
4. **Environment Package Integrity:**
   - Host `pip check`: Clean.
   - ML venv `pip check`: Clean.
5. **Model Artifact Hashes (SHA-256):**
   - `lungcare_config.json`: `6F2D60A977E0440DDED5B924FC7AFDD44B366364E060A2D4A7E16399BBF40456` (**MATCH**)
   - `lungcare_densenet121.h5`: `AC2CA7D44A61D3F1AB43C9D53484CF55E06C966D2D0C724C519983F844A9671B` (**MATCH**)
   - `lungcare_densenet121.keras`: `B583CD87A5D551CE6920BA55985492176D6236AD9895082DBB6A688C0F574249` (**MATCH**)
   - `lungcare_evaluation.json`: `5239BAC95C1F1374A3D348573D0965836F89FC7FD8BC90E6F02F2BA912012392` (**MATCH**)

---

## 7. Required Remediation

To achieve milestone closure, the following single, narrowly scoped remediation is required:

### Finding: Scenario E State Invariant Guard
In `backend/app/services/ml_coordinator.py#L124-L129`:
```python
    def complete_loading(self) -> None:
        """Signal that model loading and contract validation completed successfully."""
        with self._state_lock:
            # Remediation: Do not revert DRAINING state to READY if shutdown initiated during load
            if self._state == ModelLifecycleState.DRAINING:
                logger.warning(
                    "Model loading completed but coordinator is in DRAINING state; retaining DRAINING."
                )
                return
            self._state = ModelLifecycleState.READY
            self._last_error = None
            logger.info("ML lifecycle transition -> READY")
```

Also add a test in `backend/tests/test_concurrency_coordinator.py`:
```python
def test_complete_loading_preserves_draining_state() -> None:
    """Verify that an in-progress load completing after start_draining does not revert to READY."""
    coord = MLExecutionCoordinator()
    coord.start_loading()
    coord.start_draining()
    assert coord.state == ModelLifecycleState.DRAINING

    coord.complete_loading()
    assert coord.state == ModelLifecycleState.DRAINING
```

---

## 8. Final Audit Verdict

The concurrency architecture is substantially sound, robust, and correctly resolves lock fragmentation. However, in accordance with strict auditing standards regarding Scenario E lifecycle invariants:

**PHASE 2.3.3.2 — REMEDIATION REQUIRED**
