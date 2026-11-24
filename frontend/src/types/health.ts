/**
 * Health and Readiness Probe Type Contracts
 * Mirrors backend Pydantic schemas in backend/app/schemas/health.py
 */

/**
 * Lifecycle states of the ML execution coordinator.
 * Derived from backend/app/services/ml_coordinator.py ModelLifecycleState.
 */
export type ModelLifecycleState = 'UNLOADED' | 'LOADING' | 'READY' | 'DRAINING' | 'ERROR';

/**
 * Health Check (Liveness Probe) Response Model
 * Corresponds to backend/app/schemas/health.py HealthResponse
 */
export interface HealthResponse {
  status: 'healthy' | string;
  service: string;
  version: string;
  timestamp: string;
}

/**
 * Readiness Subsystem Dependencies Dictionary
 */
export interface ReadinessDependencies {
  ml_model?: string;
  database?: string;
  [key: string]: unknown;
}

/**
 * Readiness Check Probe Response Model (HTTP 200 OK)
 * Corresponds to backend/app/schemas/health.py ReadinessResponse
 */
export interface ReadinessResponse {
  status: 'ready' | string;
  service: string;
  version: string;
  model_loaded: boolean;
  environment: string;
  dependencies: ReadinessDependencies;
  timestamp: string;
}

/**
 * Readiness Error Details Payload (HTTP 503 Service Unavailable)
 * Included inside ErrorResponse when lifecycle is in ERROR or DRAINING
 */
export interface ReadinessErrorDetails {
  lifecycle_state: ModelLifecycleState;
  dependencies: ReadinessDependencies;
  [key: string]: unknown;
}

/**
 * Backend API connection liveness status
 */
export type BackendStatus = 'checking' | 'healthy' | 'unavailable';

/**
 * ML model readiness lifecycle status
 */
export type ModelStatus = 'checking' | 'ready' | 'not_loaded' | 'unavailable';

/**
 * Full state contract for live health and readiness monitoring
 */
export interface HealthCheckState {
  backendStatus: BackendStatus;
  modelStatus: ModelStatus;
  isLoading: boolean;
  lastChecked: Date | null;
  lastSuccess: Date | null;
  error: string | null;
  healthData: HealthResponse | null;
  readinessData: ReadinessResponse | null;
  refetch: () => Promise<void>;
}
