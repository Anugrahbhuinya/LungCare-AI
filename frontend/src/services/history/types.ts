/**
 * Type definitions for browser-local scan history stored in IndexedDB.
 */

export interface HistoryAnalysisRecord {
  /** Schema version for forward-compatibility migrations */
  schema_version: number;
  /** Unique stable identifier for the record (e.g. analysis-timestamp-uuid) */
  id: string;
  /** ISO-8601 timestamp string representing when the analysis occurred */
  timestamp: string;
  /** File name of the evaluated radiograph */
  fileName: string;
  /** File size in bytes, if available */
  fileSize?: number;
  /** MIME type of the file, if available */
  fileType?: string;
  /** Raw model prediction identifier (e.g. 'COVID19', 'NORMAL', 'PNEUMONIA', 'TURBERCULOSIS') */
  predicted_class: string;
  /** Human-readable display label (e.g. 'COVID-19', 'Normal', 'Pneumonia', 'Tuberculosis') */
  predicted_label: string;
  /** Predicted class probability in [0.0, 1.0] */
  confidence: number;
  /** Probability distribution across all four evaluated classes */
  probabilities: Record<string, number>;
  /** Selected Grad-CAM explainability target class */
  selected_target_class?: string;
  /** Selected Grad-CAM human-readable target label */
  selected_target_label?: string;
  /** Target convolutional layer name evaluated for explainability */
  target_layer?: string;
  /** Model inference duration in milliseconds */
  inference_time_ms?: number;
  /** Total analysis execution duration in milliseconds */
  total_time_ms?: number;
  /** Backend tracking request identifier (X-Request-ID) */
  request_id?: string | null;
}

export class HistoryStorageError extends Error {
  constructor(message: string, public readonly originalError?: unknown) {
    super(message);
    this.name = 'HistoryStorageError';
  }
}

/**
 * Validates at runtime whether an unknown object conforms to the HistoryAnalysisRecord contract.
 * Protects consumers from malformed records without throwing exceptions.
 */
export function isValidHistoryRecord(item: unknown): item is HistoryAnalysisRecord {
  if (!item || typeof item !== 'object' || Array.isArray(item)) {
    return false;
  }

  const rec = item as Record<string, unknown>;

  // Required string identifiers
  if (typeof rec.id !== 'string' || rec.id.trim().length === 0) {
    return false;
  }

  if (typeof rec.fileName !== 'string' || rec.fileName.trim().length === 0) {
    return false;
  }

  if (typeof rec.predicted_class !== 'string' || rec.predicted_class.trim().length === 0) {
    return false;
  }

  if (typeof rec.predicted_label !== 'string' || rec.predicted_label.trim().length === 0) {
    return false;
  }

  // Required valid timestamp
  if (typeof rec.timestamp !== 'string') {
    return false;
  }
  const parsedTime = new Date(rec.timestamp).getTime();
  if (isNaN(parsedTime)) {
    return false;
  }

  // Required numerical confidence in [0.0, 1.0]
  if (typeof rec.confidence !== 'number' || isNaN(rec.confidence) || rec.confidence < 0 || rec.confidence > 1) {
    return false;
  }

  // Required probabilities map
  if (!rec.probabilities || typeof rec.probabilities !== 'object' || Array.isArray(rec.probabilities)) {
    return false;
  }
  const probEntries = Object.entries(rec.probabilities as Record<string, unknown>);
  if (probEntries.length === 0) {
    return false;
  }
  for (const [, val] of probEntries) {
    if (typeof val !== 'number' || isNaN(val)) {
      return false;
    }
  }

  // Optional numerical fields
  if (rec.fileSize !== undefined && (typeof rec.fileSize !== 'number' || isNaN(rec.fileSize))) {
    return false;
  }
  if (rec.inference_time_ms !== undefined && (typeof rec.inference_time_ms !== 'number' || isNaN(rec.inference_time_ms))) {
    return false;
  }
  if (rec.total_time_ms !== undefined && (typeof rec.total_time_ms !== 'number' || isNaN(rec.total_time_ms))) {
    return false;
  }

  // Optional string fields
  if (rec.fileType !== undefined && typeof rec.fileType !== 'string') {
    return false;
  }
  if (rec.selected_target_class !== undefined && typeof rec.selected_target_class !== 'string') {
    return false;
  }
  if (rec.selected_target_label !== undefined && typeof rec.selected_target_label !== 'string') {
    return false;
  }
  if (rec.target_layer !== undefined && typeof rec.target_layer !== 'string') {
    return false;
  }
  if (rec.request_id !== undefined && rec.request_id !== null && typeof rec.request_id !== 'string') {
    return false;
  }

  return true;
}

/**
 * Generates an anonymized, neutral display filename for stored analysis records
 * (e.g. radiograph-a1b2c3d4.png), ensuring personal health information (PHI)
 * in original file names is not persisted to IndexedDB or exported into PDF reports.
 */
export function generateNeutralFileName(
  requestId?: string | null,
  originalFileName?: string
): string {
  let shortId = '';
  if (requestId && requestId.trim().length > 0) {
    shortId = requestId.trim().replace(/[^a-zA-Z0-9]/g, '').slice(0, 8);
  }
  if (!shortId) {
    shortId = Math.random().toString(36).slice(2, 10);
  }

  let ext = 'png';
  if (originalFileName) {
    const parts = originalFileName.split('.');
    if (parts.length > 1) {
      const candidate = parts.pop()?.toLowerCase();
      if (candidate && ['png', 'jpg', 'jpeg', 'webp'].includes(candidate)) {
        ext = candidate === 'jpeg' ? 'jpg' : candidate;
      }
    }
  }

  return `radiograph-${shortId}.${ext}`;
}
