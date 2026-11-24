import { apiClient } from './client';
import type {
  PredictionExplainResponse,
  ExplainRequestOptions,
} from '@/types/prediction';

/**
 * Submit a chest radiograph image to POST /api/v1/predictions/explain.
 *
 * Transmits the image via multipart/form-data with exact field name 'image',
 * executes DenseNet121 four-class inference on the backend, and computes
 * class-specific Grad-CAM visual activation heatmaps and overlays.
 *
 * @param file The validated chest X-ray File object.
 * @param options Optional explanation options (targetClass, alpha, AbortSignal).
 * @returns Promise resolving to the strongly-typed PredictionExplainResponse.
 * @throws ApiError with preserved requestId and structured error details on failure.
 */
export async function explainPrediction(
  file: File,
  options?: ExplainRequestOptions
): Promise<PredictionExplainResponse> {
  const formData = new FormData();
  formData.append('image', file);

  if (options?.targetClass !== undefined && options.targetClass !== null) {
    formData.append('target_class', options.targetClass);
  }

  if (options?.alpha !== undefined && options.alpha !== null) {
    formData.append('alpha', options.alpha.toString());
  }

  // Note: Axios automatically generates multipart/form-data boundary when data is FormData.
  const response = await apiClient.post<PredictionExplainResponse>(
    '/predictions/explain',
    formData,
    {
      signal: options?.signal,
    }
  );

  return response.data;
}
