/**
 * Prediction and Grad-CAM Visual Explainability TypeScript Contracts
 * Mirrors backend Pydantic models from:
 * - backend/app/schemas/prediction.py (PredictionResponse)
 * - backend/app/schemas/gradcam.py (GradCAMExplainResponse, ExplanationMetadata)
 */

export type RawModelClass = 'COVID19' | 'NORMAL' | 'PNEUMONIA' | 'TURBERCULOSIS';

export const MODEL_CLASSES: readonly RawModelClass[] = [
  'COVID19',
  'NORMAL',
  'PNEUMONIA',
  'TURBERCULOSIS',
] as const;

/**
 * Display names mapping raw model identifiers to sanitized human-readable names.
 * Corrects the research dataset misspelling 'TURBERCULOSIS' -> 'Tuberculosis'.
 */
export const CLASS_DISPLAY_NAMES: Record<RawModelClass | string, string> = {
  COVID19: 'COVID-19',
  NORMAL: 'Normal',
  PNEUMONIA: 'Pneumonia',
  TURBERCULOSIS: 'Tuberculosis',
};

export function getDisplayClassName(rawClass: string): string {
  return CLASS_DISPLAY_NAMES[rawClass] || rawClass;
}

/**
 * Technical metadata for Grad-CAM backpropagation and visual overlay parameters.
 */
export interface ExplanationMetadata {
  target_layer: string;
  differentiation_target: string;
  status: 'success' | 'uninformative' | string;
  alpha: number;
  colormap: string;
  input_width: number;
  input_height: number;
  processed_width: number;
  processed_height: number;
  inference_time_ms: number;
  gradcam_time_ms: number;
  total_time_ms: number;
}

/**
 * Combined Prediction and Grad-CAM response returned by POST /api/v1/predictions/explain
 */
export interface PredictionExplainResponse {
  status: 'success';
  predicted_class: RawModelClass | string;
  predicted_label: string;
  display_name: string;
  confidence: number;
  probabilities: Record<RawModelClass | string, number>;
  selected_target_class: string;
  selected_target_label: string;
  target_class_index: number;
  /**
   * Base64-encoded PNG data URI of the colorized activation heatmap.
   * Format: "data:image/png;base64,iVBORw..."
   */
  heatmap_base64: string;
  /**
   * Base64-encoded PNG data URI of the blended Grad-CAM overlay on original radiograph.
   * Format: "data:image/png;base64,iVBORw..."
   */
  overlay_base64: string;
  explanation: ExplanationMetadata;
  timestamp: string;
  request_id?: string | null;
  disclaimer: string;
}

/**
 * Options for configuring prediction and explanation requests.
 */
export interface ExplainRequestOptions {
  /**
   * Optional classification target class (0-3 or name).
   * Defaults to predicted class if omitted.
   */
  targetClass?: string;
  /**
   * Optional alpha blend weight in [0.0, 1.0]. Defaults to 0.4.
   */
  alpha?: number;
  /**
   * Optional AbortSignal for request cancellation.
   */
  signal?: AbortSignal;
  /**
   * Optional flag indicating this is an explainability refresh (e.g. target class change)
   * on an existing analysis, which must not create a duplicate scan-history entry.
   */
  isRefresh?: boolean;
}
