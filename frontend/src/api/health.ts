import { apiClient } from './client';
import type { HealthResponse, ReadinessResponse } from '@/types/health';

/**
 * Query backend service liveness status.
 *
 * Hits the versioned endpoint GET /api/v1/health (resolved via apiClient baseURL).
 *
 * @param signal Optional AbortSignal for request cancellation.
 * @returns Promise resolving to strongly-typed HealthResponse.
 * @throws ApiError if network fails or backend returns an error.
 */
export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const response = await apiClient.get<HealthResponse>('/health', { signal });
  return response.data;
}

/**
 * Query backend service and ML model readiness status.
 *
 * Hits the versioned endpoint GET /api/v1/ready (resolved via apiClient baseURL).
 *
 * Expected backend lifecycle mappings:
 * - READY (model loaded into memory): HTTP 200 OK with model_loaded: true
 * - UNLOADED / Pending initialization: HTTP 200 OK with model_loaded: false
 * - ERROR / DRAINING: HTTP 503 Service Unavailable (rejected as ApiError with errorType 'MODEL_NOT_READY' or 'SERVICE_DRAINING')
 *
 * @param signal Optional AbortSignal for request cancellation.
 * @returns Promise resolving to strongly-typed ReadinessResponse.
 * @throws ApiError if service is not ready (HTTP 503) or if network fails.
 */
export async function getReadiness(signal?: AbortSignal): Promise<ReadinessResponse> {
  const response = await apiClient.get<ReadinessResponse>('/ready', { signal });
  return response.data;
}
