"""Deterministic unit and integration test suite for MLExecutionCoordinator.

Validates the concurrency architecture, unified lock synchronization,
model lifecycle states (UNLOADED, LOADING, READY, DRAINING, ERROR),
in-flight operation tracking, and graceful shutdown draining.
Uses deterministic synchronization primitives (threading.Event, threading.Barrier).
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from typing import Any, Dict, Optional
import numpy as np
import pytest

from app.services.ml_coordinator import (
    MLExecutionCoordinator,
    MLNotReadyError,
    ModelLifecycleState,
    get_ml_coordinator,
)
from app.services.inference_service import (
    InferenceExecutionError,
    InferenceService,
    ModelUnavailableError,
    PredictionResult,
)
from app.services.gradcam_service import (
    GradCAMGenerationError,
    GradCAMResult,
    GradCAMService,
)
from app.services.image_preprocessor import ImagePreprocessor, PreprocessedImageResult


# ------------------------------------------------------------------------------
# Test Fixtures & Mocks
# ------------------------------------------------------------------------------

class MockKerasModel:
    """Mock model providing controllable execution delay and call tracking."""

    def __init__(self, output: Optional[np.ndarray] = None, raise_exc: Optional[Exception] = None) -> None:
        self.output = output if output is not None else np.array([[0.1, 0.7, 0.1, 0.1]], dtype=np.float32)
        self.raise_exc = raise_exc
        self.call_count = 0
        self.on_call_event: Optional[threading.Event] = None
        self.wait_event: Optional[threading.Event] = None

    def __call__(self, tensor: Any, training: bool = False) -> np.ndarray:
        self.call_count += 1
        if self.on_call_event:
            self.on_call_event.set()
        if self.wait_event:
            self.wait_event.wait(timeout=5.0)
        if self.raise_exc:
            raise self.raise_exc
        return self.output


class MockModelLoader:
    """Mock loader providing predictable lifecycle state."""

    def __init__(self, model: Optional[Any] = None, is_loaded: bool = True) -> None:
        self.model = model or MockKerasModel()
        self.is_loaded = is_loaded
        self.metadata = None
        self.unload_called = False

    def get_model(self) -> Any:
        return self.model

    def load_model(self) -> Any:
        self.is_loaded = True
        return self.model

    def unload(self) -> None:
        self.unload_called = True
        self.model = None
        self.is_loaded = False


# ------------------------------------------------------------------------------
# 1. State Transitions & Lifecycle Invariants
# ------------------------------------------------------------------------------

def test_initial_state_and_lifecycle_transitions() -> None:
    """Requirement 11 & 12: Test state transitions through LOADING, READY, ERROR, UNLOADED."""
    coord = MLExecutionCoordinator()
    assert coord.state == ModelLifecycleState.UNLOADED
    assert coord.active_operations == 0
    assert not coord.is_ready

    # Transition to LOADING
    coord.start_loading()
    assert coord.state == ModelLifecycleState.LOADING
    assert not coord.is_ready

    # Transition to READY
    coord.complete_loading()
    assert coord.state == ModelLifecycleState.READY
    assert coord.is_ready

    # Transition to ERROR
    coord.record_loading_error(RuntimeError("Disk corrupted"))
    assert coord.state == ModelLifecycleState.ERROR
    assert not coord.is_ready
    assert isinstance(coord.last_error, RuntimeError)

    # Transition to UNLOADED
    coord.complete_unloading()
    assert coord.state == ModelLifecycleState.UNLOADED
    assert not coord.is_ready


def test_operation_rejected_when_loading() -> None:
    """Requirement 4: An operation cannot start while lifecycle state is LOADING."""
    coord = MLExecutionCoordinator()
    coord.start_loading()

    with pytest.raises(MLNotReadyError) as exc_info:
        with coord.acquire_execution():
            pass

    assert exc_info.value.state == ModelLifecycleState.LOADING
    assert "currently loading" in str(exc_info.value).lower()
    assert coord.active_operations == 0


def test_operation_rejected_when_draining() -> None:
    """Requirement 5: New operations are rejected when coordinator is DRAINING."""
    coord = MLExecutionCoordinator()
    coord.complete_loading()
    assert coord.is_ready

    coord.start_draining()
    assert coord.is_draining

    with pytest.raises(MLNotReadyError) as exc_info:
        with coord.acquire_execution():
            pass

    assert exc_info.value.state == ModelLifecycleState.DRAINING
    assert "shutting down" in str(exc_info.value).lower()
    assert coord.active_operations == 0


# ------------------------------------------------------------------------------
# 2. Execution Lock & Serialization Tests
# ------------------------------------------------------------------------------

def test_two_simultaneous_predictions_serialize() -> None:
    """Requirement 1: Two simultaneous prediction operations serialize under the lock."""
    coord = MLExecutionCoordinator()
    coord.complete_loading()

    entered_threads = []
    thread_order = []
    lock_concurrency_detected = False

    t1_inside = threading.Event()
    t1_can_exit = threading.Event()

    def worker_1():
        with coord.acquire_execution():
            entered_threads.append("t1")
            t1_inside.set()
            # Wait for main thread to signal worker 2 has attempted entry
            t1_can_exit.wait(timeout=5.0)
            thread_order.append("t1_done")

    def worker_2():
        with coord.acquire_execution():
            entered_threads.append("t2")
            thread_order.append("t2_done")

    t1 = threading.Thread(target=worker_1)
    t2 = threading.Thread(target=worker_2)

    t1.start()
    t1_inside.wait(timeout=5.0)

    # At this point, worker 1 holds the lock
    assert coord.active_operations == 1

    t2.start()
    time.sleep(0.05)  # Ensure t2 is attempting to acquire the lock

    # Verify t2 has NOT entered while t1 is inside
    assert entered_threads == ["t1"]

    # Release worker 1
    t1_can_exit.set()

    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    # Both completed in strict sequence
    assert entered_threads == ["t1", "t2"]
    assert thread_order == ["t1_done", "t2_done"]
    assert coord.active_operations == 0


def test_prediction_and_gradcam_share_same_execution_lock() -> None:
    """Requirement 2: Prediction and Grad-CAM share the identical coordinator execution lock."""
    coord = MLExecutionCoordinator()
    coord.complete_loading()

    mock_loader = MockModelLoader()
    mock_prep = ImagePreprocessor()

    # Pass the shared coordinator to both services
    inf_service = InferenceService(model_loader=mock_loader, preprocessor=mock_prep, coordinator=coord)
    gradcam_service = GradCAMService(model_loader=mock_loader, coordinator=coord)

    assert inf_service.coordinator is coord
    assert gradcam_service.coordinator is coord
    assert inf_service.coordinator.execution_lock is gradcam_service.coordinator.execution_lock


def test_multiple_gradcam_operations_serialize() -> None:
    """Requirement 3: Multiple Grad-CAM operations serialize without data races."""
    coord = MLExecutionCoordinator()
    coord.complete_loading()

    execution_log = []
    barrier = threading.Barrier(3)

    def gradcam_task(task_id: int):
        barrier.wait()  # Synchronize starts
        with coord.acquire_execution():
            execution_log.append(f"start_{task_id}")
            time.sleep(0.02)
            execution_log.append(f"end_{task_id}")

    threads = [threading.Thread(target=gradcam_task, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert len(execution_log) == 6
    # Verify strict serialization: each start_X must be immediately followed by end_X
    for i in range(0, 6, 2):
        start_entry = execution_log[i]
        end_entry = execution_log[i + 1]
        task_id = start_entry.split("_")[1]
        assert end_entry == f"end_{task_id}"


# ------------------------------------------------------------------------------
# 3. Exception Safety & Counter Invariants
# ------------------------------------------------------------------------------

def test_exception_inside_execution_releases_lock_and_decrements_counter() -> None:
    """Requirement 8: Exception inside execution context releases lock and restores counter to 0."""
    coord = MLExecutionCoordinator()
    coord.complete_loading()

    with pytest.raises(ValueError) as exc_info:
        with coord.acquire_execution():
            assert coord.active_operations == 1
            raise ValueError("Injected runtime failure")

    assert str(exc_info.value) == "Injected runtime failure"
    assert coord.active_operations == 0

    # Lock must be immediately available for the next caller
    acquired = coord.execution_lock.acquire(blocking=False)
    assert acquired is True
    coord.execution_lock.release()


def test_subsequent_request_succeeds_after_operation_fails() -> None:
    """Requirement 9: Subsequent request succeeds cleanly after a preceding operation fails."""
    coord = MLExecutionCoordinator()
    coord.complete_loading()

    # Call 1: Fails
    with pytest.raises(RuntimeError):
        with coord.acquire_execution():
            raise RuntimeError("Operation 1 failure")

    assert coord.active_operations == 0

    # Call 2: Succeeds
    executed = False
    with coord.acquire_execution():
        executed = True

    assert executed is True
    assert coord.active_operations == 0


def test_concurrent_attempts_cannot_corrupt_counter() -> None:
    """Requirement 10: Highly concurrent lock acquisitions preserve counter integrity."""
    coord = MLExecutionCoordinator()
    coord.complete_loading()

    num_threads = 20
    barrier = threading.Barrier(num_threads)

    def worker():
        barrier.wait()
        for _ in range(10):
            with coord.acquire_execution():
                pass

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker) for _ in range(num_threads)]
        for f in futures:
            f.result()

    assert coord.active_operations == 0


# ------------------------------------------------------------------------------
# 4. Graceful Shutdown & Draining
# ------------------------------------------------------------------------------

def test_shutdown_waits_for_active_operation() -> None:
    """Requirement 6: Shutdown draining blocks until all active operations finish."""
    coord = MLExecutionCoordinator()
    coord.complete_loading()

    operation_running = threading.Event()
    operation_can_finish = threading.Event()
    operation_finished = threading.Event()

    def long_operation():
        with coord.acquire_execution():
            operation_running.set()
            operation_can_finish.wait(timeout=5.0)
        operation_finished.set()

    t = threading.Thread(target=long_operation)
    t.start()
    operation_running.wait(timeout=5.0)

    # Server initiates shutdown
    coord.start_draining()
    assert coord.active_operations == 1

    # Draining wait in separate thread
    drain_completed = threading.Event()

    def drain_task():
        drained = coord.wait_for_active_operations(timeout=5.0)
        if drained:
            drain_completed.set()

    drain_thread = threading.Thread(target=drain_task)
    drain_thread.start()

    # Verify drain has not completed while operation is active
    time.sleep(0.05)
    assert not drain_completed.is_set()

    # Complete the operation
    operation_can_finish.set()
    t.join(timeout=5.0)

    # Now drain should complete
    drain_completed.wait(timeout=5.0)
    assert drain_completed.is_set()
    assert coord.active_operations == 0


def test_model_unloading_happens_only_after_operations_reach_zero() -> None:
    """Requirement 7: Model unload is refused or waits until active operations reach zero."""
    coord = MLExecutionCoordinator()
    coord.complete_loading()

    op_active = threading.Event()
    op_can_exit = threading.Event()

    def active_worker():
        with coord.acquire_execution():
            op_active.set()
            op_can_exit.wait(timeout=5.0)

    t = threading.Thread(target=active_worker)
    t.start()
    op_active.wait(timeout=5.0)

    assert coord.active_operations == 1

    # Attempting complete_unloading while active operations exist is safely rejected
    coord.complete_unloading()
    assert coord.state == ModelLifecycleState.READY  # Remains READY, refusing UNLOADED

    # Release active operation
    op_can_exit.set()
    t.join(timeout=5.0)

    # Now unloading succeeds
    coord.complete_unloading()
    assert coord.state == ModelLifecycleState.UNLOADED


# ------------------------------------------------------------------------------
# 5. Service-Level Integration Smoke Test
# ------------------------------------------------------------------------------

def test_inference_service_maps_coordinator_rejection_to_model_unavailable() -> None:
    """Verify InferenceService translates MLNotReadyError into ModelUnavailableError."""
    coord = MLExecutionCoordinator()
    coord.start_draining()  # Coordinator will reject execution

    mock_loader = MockModelLoader()
    mock_prep = ImagePreprocessor()
    service = InferenceService(model_loader=mock_loader, preprocessor=mock_prep, coordinator=coord)

    dummy_tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)

    with pytest.raises(ModelUnavailableError) as exc_info:
        service.predict_tensor(dummy_tensor)

    assert "shutting down" in str(exc_info.value).lower()
    assert exc_info.value.details.get("lifecycle_state") == "DRAINING"


# ------------------------------------------------------------------------------
# 6. Scenario E: Loading and Shutdown Race Remediation Tests
# ------------------------------------------------------------------------------

def test_complete_loading_preserves_draining_state() -> None:
    """Requirement 11: An in-progress load completing after start_draining does not revert to READY.
    
    Verifies:
    1. Coordinator is put into LOADING.
    2. Coordinator transitions to DRAINING.
    3. complete_loading() is called.
    4. State remains DRAINING (does NOT overwrite to READY).
    5. New execution admission is rejected with MLNotReadyError.
    """
    coord = MLExecutionCoordinator()
    coord.start_loading()
    assert coord.state == ModelLifecycleState.LOADING

    coord.start_draining()
    assert coord.state == ModelLifecycleState.DRAINING

    coord.complete_loading()
    assert coord.state == ModelLifecycleState.DRAINING

    with pytest.raises(MLNotReadyError) as exc_info:
        with coord.acquire_execution():
            pass

    assert exc_info.value.state == ModelLifecycleState.DRAINING
    assert "shutting down" in str(exc_info.value).lower()


def test_concurrent_load_and_shutdown_race_preserves_draining() -> None:
    """Deterministic concurrency test for Scenario E: load completes concurrently with shutdown.
    
    Verifies that when a worker thread completes loading while main thread initiates
    shutdown draining, the resulting lifecycle state is guaranteed to remain DRAINING,
    and subsequent execution requests are rejected.
    """
    coord = MLExecutionCoordinator()
    load_started = threading.Event()
    drain_completed = threading.Event()

    def loading_worker() -> None:
        coord.start_loading()
        load_started.set()
        # Wait until main thread has initiated draining
        drain_completed.wait(timeout=5.0)
        # Attempt to complete loading after draining has started
        coord.complete_loading()

    t = threading.Thread(target=loading_worker)
    t.start()

    assert load_started.wait(timeout=5.0) is True
    coord.start_draining()
    assert coord.state == ModelLifecycleState.DRAINING
    drain_completed.set()

    t.join(timeout=5.0)
    assert not t.is_alive()

    # Invariant: State must remain DRAINING
    assert coord.state == ModelLifecycleState.DRAINING

    # New execution admission must fail
    with pytest.raises(MLNotReadyError) as exc_info:
        with coord.acquire_execution():
            pass

    assert exc_info.value.state == ModelLifecycleState.DRAINING
