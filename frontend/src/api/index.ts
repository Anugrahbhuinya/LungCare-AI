export {
  apiClient,
  default as client,
  API_BASE_URL,
  DEFAULT_API_BASE_URL,
  generateRequestId,
} from './client';

export {
  getHealth,
  getReadiness,
} from './health';

export {
  explainPrediction,
} from './predictions';

export {
  ApiError,
  isApiError,
  type ApiErrorData,
  type BackendErrorDetail,
  type BackendErrorResponse,
} from '@/types/api';

export type {
  HealthResponse,
  ReadinessResponse,
  ReadinessDependencies,
  ReadinessErrorDetails,
  ModelLifecycleState,
  BackendStatus,
  ModelStatus,
  HealthCheckState,
} from '@/types/health';

export {
  MODEL_CLASSES,
  CLASS_DISPLAY_NAMES,
  getDisplayClassName,
  type RawModelClass,
  type ExplanationMetadata,
  type PredictionExplainResponse,
  type ExplainRequestOptions,
} from '@/types/prediction';
