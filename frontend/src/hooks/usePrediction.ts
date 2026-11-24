import { useState, useCallback, useRef, useEffect, useContext } from 'react';
import { explainPrediction } from '@/api/predictions';
import type {
  PredictionExplainResponse,
  ExplainRequestOptions,
} from '@/types/prediction';
import { ApiError } from '@/types/api';
import SessionContext from '@/context/SessionContext';
import { useSettings } from './useSettings';
import {
  saveHistoryRecord,
  isIndexedDBAvailable,
  getAnalysisRecordId,
  generateNeutralFileName,
  type HistoryAnalysisRecord,
} from '@/services/history';

export type PredictionStatus = 'idle' | 'loading' | 'success' | 'error';

export interface UsePredictionReturn {
  status: PredictionStatus;
  isLoading: boolean;
  isSuccess: boolean;
  isError: boolean;
  result: PredictionExplainResponse | null;
  error: string | null;
  requestId: string | null;
  activeAlpha: number;
  activeTargetClass?: string;
  storageWarning: string | null;
  runAnalysis: (file: File, options?: ExplainRequestOptions) => Promise<boolean>;
  updateAlpha: (newAlpha: number) => void;
  resetPrediction: () => void;
  cancelAnalysis: () => void;
}

/**
 * Hook to manage DenseNet121 inference and Grad-CAM visual explainability execution.
 *
 * Enforces race-condition protection, in-flight cancellation via AbortSignal,
 * duplicate submission guards, and atomic state transitions.
 * Automatically synchronizes genuine prediction events with the application SessionContext.
 */
export function usePrediction(): UsePredictionReturn {
  const session = useContext(SessionContext);
  const { settings } = useSettings();
  const [status, setStatus] = useState<PredictionStatus>('idle');
  const [result, setResult] = useState<PredictionExplainResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [requestId, setRequestId] = useState<string | null>(null);
  const [activeAlpha, setActiveAlpha] = useState<number>(settings.defaultAlpha);
  const [activeTargetClass, setActiveTargetClass] = useState<string | undefined>(undefined);
  const [storageWarning, setStorageWarning] = useState<string | null>(null);

  // Synchronize activeAlpha with settings.defaultAlpha while idle so new analyses use the saved default
  useEffect(() => {
    if (status === 'idle') {
      setActiveAlpha(settings.defaultAlpha);
    }
  }, [settings.defaultAlpha, status]);

  // Monotonic sequence counter to prevent stale out-of-order responses from overwriting newer state
  const seqRef = useRef<number>(0);
  const abortControllerRef = useRef<AbortController | null>(null);
  const inFlightRef = useRef<boolean>(false);
  const persistedRecordIdsRef = useRef<Set<string>>(new Set());

  // Clean up in-flight request on unmount
  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
      inFlightRef.current = false;
    };
  }, []);

  const runAnalysis = useCallback(
    async (file: File, options?: ExplainRequestOptions): Promise<boolean> => {
      // 1. Prevent duplicate concurrent submissions (synchronous in-flight ref + reactive status guard)
      if (inFlightRef.current || status === 'loading') {
        return false;
      }
      inFlightRef.current = true;

      // 2. Abort any previous pending request
      abortControllerRef.current?.abort();

      const controller = new AbortController();
      abortControllerRef.current = controller;
      const currentSeq = ++seqRef.current;

      const alpha = options?.alpha ?? (status === 'idle' ? settings.defaultAlpha : activeAlpha);
      const targetClass = options?.targetClass ?? activeTargetClass;

      setStatus('loading');
      setError(null);
      setStorageWarning(null);
      setActiveAlpha(alpha);
      setActiveTargetClass(targetClass);

      try {
        const response = await explainPrediction(file, {
          targetClass,
          alpha,
          signal: controller.signal,
        });

        // 3. Stale response guard: ignore if a newer request was dispatched or canceled
        if (currentSeq !== seqRef.current || controller.signal.aborted) {
          return false;
        }

        setResult(response);
        setRequestId(response.request_id || null);
        setStatus('success');
        setError(null);

        const isExplainabilityRefresh = Boolean(options?.isRefresh);

        // F-01 & F-04: Only record initial analysis in session and IndexedDB history.
        // Subsequent explainability refreshes (target class toggles) update visualization without duplicate rows.
        if (!isExplainabilityRefresh) {
          const neutralName = generateNeutralFileName(response.request_id, file.name);

          // Record genuine successful prediction in session context with neutral name
          session?.recordAnalysisSuccess({
            predictedClass: response.predicted_class,
            predictedLabel: response.predicted_label,
            confidence: response.confidence,
            fileName: neutralName,
            durationMs: response.explanation.total_time_ms,
            requestId: response.request_id,
          });

          // Persist metadata to browser-local IndexedDB history (non-blocking & idempotent)
          try {
            if (isIndexedDBAvailable()) {
              const recordId = getAnalysisRecordId(
                response.request_id,
                response.timestamp || new Date().toISOString()
              );

              // In-memory idempotency guard: prevent duplicate writes if this exact analysis was already recorded
              if (!persistedRecordIdsRef.current.has(recordId)) {
                const historyRecord: HistoryAnalysisRecord = {
                  schema_version: 1,
                  id: recordId,
                  timestamp: response.timestamp || new Date().toISOString(),
                  fileName: neutralName,
                  fileSize: file.size,
                  fileType: file.type,
                  predicted_class: response.predicted_class,
                  predicted_label: response.predicted_label || response.display_name,
                  confidence: response.confidence,
                  probabilities: response.probabilities,
                  selected_target_class: response.selected_target_class,
                  selected_target_label: response.selected_target_label,
                  target_layer: response.explanation?.target_layer,
                  inference_time_ms: response.explanation?.inference_time_ms,
                  total_time_ms: response.explanation?.total_time_ms,
                  request_id: response.request_id,
                };

                // F-02: Commit to disk first
                await saveHistoryRecord(historyRecord);

                // F-07: ONLY update idempotency tracking after transaction successfully commits
                persistedRecordIdsRef.current.add(recordId);
              }
              setStorageWarning(null);
            } else {
              setStorageWarning('Browser local storage (IndexedDB) is unavailable; analysis was not persisted.');
            }
          } catch (storageErr) {
            console.warn('[usePrediction] Non-fatal error persisting scan history to IndexedDB:', storageErr);
            setStorageWarning('Analysis succeeded, but local browser history could not be saved.');
          }
        }

        return true;
      } catch (err: unknown) {
        if (currentSeq !== seqRef.current || controller.signal.aborted) {
          return false;
        }

        // Ignore user/system intentional aborts
        if (err instanceof ApiError && err.isCanceled) {
          return false;
        }

        let userMessage = 'Analysis failed. Please verify the backend service is running and try again.';
        let resRequestId: string | null = null;

        if (err instanceof ApiError) {
          resRequestId = err.requestId;

          if (err.isNetworkError) {
            userMessage = 'Unable to connect to LungCare AI backend server. Please verify the API is running at http://localhost:8000.';
          } else if (err.isTimeout) {
            userMessage = 'Inference request timed out after 15 seconds. High-resolution images may require additional processing time.';
          } else if (err.statusCode === 413) {
            userMessage = 'Image size exceeds the backend upload limit of 10 MB.';
          } else if (err.statusCode === 415) {
            userMessage = 'Unsupported image media format. Supported formats: JPEG, PNG, and WEBP.';
          } else if (err.statusCode === 422) {
            userMessage = 'Image validation failure. The file may be corrupt or contains an unreadable byte sequence.';
          } else if (err.statusCode === 503) {
            userMessage = 'DenseNet121 model is not loaded into memory or the API service is currently draining (HTTP 503).';
          } else if (err.statusCode >= 500) {
            userMessage = err.message || 'An internal neural network execution error occurred.';
          } else {
            userMessage = err.message || userMessage;
          }
        } else if (err instanceof Error) {
          userMessage = err.message;
        }

        setError(userMessage);
        setRequestId(resRequestId);
        setStatus('error');

        // Record genuine failed prediction in session context with neutral name
        const failNeutralName = generateNeutralFileName(resRequestId, file.name);
        session?.recordAnalysisFailure({
          fileName: failNeutralName,
          error: userMessage,
          requestId: resRequestId,
        });

        return false;
      } finally {
        inFlightRef.current = false;
      }
    },
    [status, activeAlpha, activeTargetClass, session, settings.defaultAlpha]
  );

  // F-01: Update opacity locally without triggering backend re-inference
  const updateAlpha = useCallback((newAlpha: number) => {
    setActiveAlpha(newAlpha);
    setResult((prev) => {
      if (!prev) return null;
      return {
        ...prev,
        explanation: {
          ...prev.explanation,
          alpha: newAlpha,
        },
      };
    });
  }, []);

  const resetPrediction = useCallback(() => {
    abortControllerRef.current?.abort();
    inFlightRef.current = false;
    seqRef.current++;
    setStatus('idle');
    setResult(null);
    setError(null);
    setRequestId(null);
    setActiveAlpha(settings.defaultAlpha);
    setActiveTargetClass(undefined);
    setStorageWarning(null);
  }, [settings.defaultAlpha]);

  const cancelAnalysis = useCallback(() => {
    abortControllerRef.current?.abort();
    inFlightRef.current = false;
    seqRef.current++;
    setStatus('idle');
  }, []);

  return {
    status,
    isLoading: status === 'loading',
    isSuccess: status === 'success',
    isError: status === 'error',
    result,
    error,
    requestId,
    activeAlpha,
    activeTargetClass,
    storageWarning,
    runAnalysis,
    updateAlpha,
    resetPrediction,
    cancelAnalysis,
  };
}

export default usePrediction;
