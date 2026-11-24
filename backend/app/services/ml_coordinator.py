"""Centralized ML Execution Coordinator and Lifecycle Manager for LungCare AI.

Provides unified process-wide synchronization for TensorFlow/Keras inference
and Grad-CAM execution, eliminating race conditions while tracking in-flight
operations and managing model lifecycle states (UNLOADED, LOADING, READY, DRAINING, ERROR).
"""

from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
import logging
import threading
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger("lungcare.ml_coordinator")


class ModelLifecycleState(str, Enum):
    """Lifecycle states of the deep learning model subsystem."""

    UNLOADED = "UNLOADED"
    LOADING = "LOADING"
    READY = "READY"
    DRAINING = "DRAINING"
    ERROR = "ERROR"


class MLCoordinatorError(Exception):
    """Base exception for ML execution coordination and lifecycle violations."""

    def __init__(
        self,
        message: str,
        error_code: str = "ML_COORDINATOR_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class MLNotReadyError(MLCoordinatorError):
    """Raised when an ML operation is requested while the model is not in READY state."""

    def __init__(self, state: ModelLifecycleState, message: Optional[str] = None) -> None:
        self.state = state
        msg = message or f"Model is not ready for execution (current state: {state.value})."
        super().__init__(
            message=msg,
            error_code="MODEL_NOT_READY",
            details={"lifecycle_state": state.value},
        )


class MLExecutionCoordinator:
    """Thread-safe coordinator for ML operations and model lifecycle management.
    
    Guarantees:
    1. A single shared threading.Lock synchronizes all TensorFlow/Keras executions.
    2. A separate state lock and condition variable manages lifecycle state transitions
       and in-flight operation accounting.
    3. The state lock is NEVER held during TensorFlow/Keras execution.
    4. Active operations are tracked atomically and decremented guaranteed via finally.
    5. Graceful draining rejects new operations while allowing in-flight ones to finish.
    """

    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        self._state_condition = threading.Condition(self._state_lock)
        self._execution_lock = threading.Lock()

        self._state: ModelLifecycleState = ModelLifecycleState.UNLOADED
        self._active_operations: int = 0
        self._last_error: Optional[Exception] = None

    @property
    def state(self) -> ModelLifecycleState:
        """Return the current model lifecycle state."""
        with self._state_lock:
            return self._state

    @property
    def active_operations(self) -> int:
        """Return the count of currently active in-flight ML operations."""
        with self._state_lock:
            return self._active_operations

    @property
    def is_ready(self) -> bool:
        """Return True if the coordinator is in READY state."""
        with self._state_lock:
            return self._state == ModelLifecycleState.READY

    @property
    def is_draining(self) -> bool:
        """Return True if the coordinator is in DRAINING state."""
        with self._state_lock:
            return self._state == ModelLifecycleState.DRAINING

    @property
    def last_error(self) -> Optional[Exception]:
        """Return the last model loading or execution error, if any."""
        with self._state_lock:
            return self._last_error

    @property
    def execution_lock(self) -> threading.Lock:
        """Access the shared execution lock for synchronization verification."""
        return self._execution_lock

    # --------------------------------------------------------------------------
    # Lifecycle State Transitions
    # --------------------------------------------------------------------------

    def start_loading(self) -> None:
        """Signal that model loading and validation from disk has started."""
        with self._state_lock:
            if self._state == ModelLifecycleState.DRAINING:
                logger.warning(
                    "Model loading initiated while coordinator is in DRAINING state; retaining DRAINING."
                )
                return
            self._state = ModelLifecycleState.LOADING
            self._last_error = None
            logger.info("ML lifecycle transition -> LOADING")

    def complete_loading(self) -> None:
        """Signal that model loading and contract validation completed successfully."""
        with self._state_lock:
            if self._state == ModelLifecycleState.DRAINING:
                logger.warning(
                    "Model loading completed but coordinator is in DRAINING state; retaining DRAINING."
                )
                return
            self._state = ModelLifecycleState.READY
            self._last_error = None
            logger.info("ML lifecycle transition -> READY")

    def record_loading_error(self, exc: Exception) -> None:
        """Record model loading/validation failure and transition state to ERROR."""
        with self._state_lock:
            if self._state == ModelLifecycleState.DRAINING:
                logger.warning(
                    "Model loading failed while coordinator is in DRAINING state; retaining DRAINING (reason: %s).",
                    exc,
                )
                self._last_error = exc
                return
            self._state = ModelLifecycleState.ERROR
            self._last_error = exc
            logger.error("ML lifecycle transition -> ERROR (reason: %s)", exc)

    def start_draining(self) -> None:
        """Signal the start of shutdown draining; reject any subsequent incoming ML requests."""
        with self._state_lock:
            self._state = ModelLifecycleState.DRAINING
            logger.info(
                "ML lifecycle transition -> DRAINING (active in-flight operations: %d)",
                self._active_operations,
            )

    def wait_for_active_operations(self, timeout: Optional[float] = None) -> bool:
        """Wait until all active in-flight operations finish, with an optional bounded timeout.
        
        Args:
            timeout: Maximum seconds to wait. None waits indefinitely.

        Returns:
            True if all operations finished (active_operations == 0), False if timeout elapsed.
        """
        with self._state_condition:
            if self._active_operations == 0:
                return True
            logger.info(
                "Waiting for %d active ML operations to finish (timeout: %s)...",
                self._active_operations,
                f"{timeout:.1f}s" if timeout is not None else "None",
            )
            return self._state_condition.wait_for(
                lambda: self._active_operations == 0,
                timeout=timeout,
            )

    def complete_unloading(self) -> None:
        """Complete model unloading, resetting state to UNLOADED.
        
        Refuses state transition if active operations remain.
        """
        with self._state_lock:
            if self._active_operations > 0:
                logger.warning(
                    "Cannot cleanly transition to UNLOADED: %d operations are still active!",
                    self._active_operations,
                )
                return
            self._state = ModelLifecycleState.UNLOADED
            self._last_error = None
            logger.info("ML lifecycle transition -> UNLOADED")

    # --------------------------------------------------------------------------
    # Synchronized Execution Context
    # --------------------------------------------------------------------------

    @contextmanager
    def acquire_execution(self) -> Generator[None, None, None]:
        """Context manager to synchronize TensorFlow/Keras execution.
        
        Guarantees:
        1. Validates under _state_lock that state is READY.
        2. Atomically increments _active_operations under _state_lock.
        3. Releases _state_lock before acquiring _execution_lock (prevents lock inversion).
        4. Yields control inside with _execution_lock for forward/backward computation.
        5. In finally: releases _execution_lock, reacquires _state_condition,
           atomically decrements _active_operations, and notifies draining waiters
           if counter drops to zero.
        """
        # Step 1: Validate state and increment active operation counter
        with self._state_lock:
            if self._state != ModelLifecycleState.READY:
                if self._state == ModelLifecycleState.DRAINING:
                    msg = "Server is shutting down; new ML requests are rejected."
                elif self._state == ModelLifecycleState.LOADING:
                    msg = "DenseNet121 model is currently loading into memory."
                elif self._state == ModelLifecycleState.ERROR:
                    msg = f"DenseNet121 model is in error state: {self._last_error}"
                else:
                    msg = f"DenseNet121 model is not ready for execution (state: {self._state.value})."
                raise MLNotReadyError(self._state, msg)

            self._active_operations += 1

        try:
            # Step 2: Acquire the unified process-wide execution lock
            with self._execution_lock:
                yield
        finally:
            # Step 3: Decrement active operation counter and notify waiting threads
            with self._state_condition:
                self._active_operations -= 1
                if self._active_operations == 0:
                    self._state_condition.notify_all()

    def reset_for_test(self) -> None:
        """Reset coordinator state for testing isolation."""
        with self._state_condition:
            self._state = ModelLifecycleState.UNLOADED
            self._active_operations = 0
            self._last_error = None


# ------------------------------------------------------------------------------
# Process-Wide Singleton Accessor
# ------------------------------------------------------------------------------

_coordinator_instance: Optional[MLExecutionCoordinator] = None
_singleton_lock = threading.Lock()


def get_ml_coordinator() -> MLExecutionCoordinator:
    """Return the application-wide singleton MLExecutionCoordinator instance."""
    global _coordinator_instance
    if _coordinator_instance is None:
        with _singleton_lock:
            if _coordinator_instance is None:
                _coordinator_instance = MLExecutionCoordinator()
    return _coordinator_instance
