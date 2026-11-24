import React, { createContext, useContext, useEffect, useState, useRef, useCallback } from 'react';
import { getHealth, getReadiness } from '@/api/health';
import type {
  BackendStatus,
  ModelStatus,
  HealthCheckState,
  HealthResponse,
  ReadinessResponse,
} from '@/types/health';
import { ApiError } from '@/types/api';

export const HealthContext = createContext<HealthCheckState | null>(null);

export interface HealthProviderProps {
  children: React.ReactNode;
  /**
   * Polling cadence in milliseconds. Defaults to 30,000ms (30 seconds).
   */
  pollingInterval?: number;
  /**
   * Whether to enable automatic polling. Defaults to true.
   */
  enabled?: boolean;
}

export function HealthProvider({
  children,
  pollingInterval = 30000,
  enabled = true,
}: HealthProviderProps) {
  const [backendStatus, setBackendStatus] = useState<BackendStatus>('checking');
  const [modelStatus, setModelStatus] = useState<ModelStatus>('checking');
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);
  const [lastSuccess, setLastSuccess] = useState<Date | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [healthData, setHealthData] = useState<HealthResponse | null>(null);
  const [readinessData, setReadinessData] = useState<ReadinessResponse | null>(null);

  // Guards against overlapping in-flight calls and stale out-of-order responses
  const inFlightRef = useRef<boolean>(false);
  const seqRef = useRef<number>(0);
  const abortControllerRef = useRef<AbortController | null>(null);

  const performCheck = useCallback(async (isManual = false) => {
    // 1. Prevent overlapping polling requests
    if (inFlightRef.current) {
      if (!isManual) {
        // Interval tick: skip this cycle if a check is already in-flight
        return;
      }
      // Manual trigger: cancel existing in-flight request to immediately dispatch fresh check
      abortControllerRef.current?.abort();
    }

    const controller = new AbortController();
    abortControllerRef.current = controller;
    inFlightRef.current = true;
    const currentSeq = ++seqRef.current;

    setIsLoading(true);

    try {
      // Execute health and readiness concurrently with shared AbortSignal
      const [healthResult, readinessResult] = await Promise.allSettled([
        getHealth(controller.signal),
        getReadiness(controller.signal),
      ]);

      // 2. Prevent stale out-of-order responses from overwriting newer state
      if (currentSeq !== seqRef.current || controller.signal.aborted) {
        return;
      }

      const checkTime = new Date();
      let nextBackendStatus: BackendStatus = 'unavailable';
      let nextModelStatus: ModelStatus = 'unavailable';
      let nextHealthData: HealthResponse | null = null;
      let nextReadinessData: ReadinessResponse | null = null;
      let resolvedError: string | null = null;
      let isBackendHealthy = false;

      // --- Evaluate Health (Backend Liveness) ---
      if (healthResult.status === 'fulfilled') {
        nextBackendStatus = 'healthy';
        nextHealthData = healthResult.value;
        isBackendHealthy = true;
      } else {
        const hErr = healthResult.reason;
        if (hErr instanceof ApiError && hErr.isCanceled) {
          return; // Superceded / aborted, do not modify state
        }
        nextBackendStatus = 'unavailable';
        resolvedError =
          hErr instanceof Error
            ? hErr.message
            : 'Unable to connect to LungCare AI backend service.';
      }

      // --- Evaluate Readiness (ML Model Lifecycle) ---
      // Note: A healthy backend does NOT imply the ML model is ready.
      if (readinessResult.status === 'fulfilled') {
        nextReadinessData = readinessResult.value;
        if (nextReadinessData.model_loaded) {
          nextModelStatus = 'ready';
        } else {
          // Backend is alive, but model has not completed loading
          nextModelStatus = 'not_loaded';
        }
      } else {
        const rErr = readinessResult.reason;
        if (rErr instanceof ApiError && rErr.isCanceled) {
          return; // Superceded / aborted
        }

        if (rErr instanceof ApiError) {
          if (rErr.statusCode === 503) {
            // Backend explicitly returned 503 (MODEL_NOT_READY or SERVICE_DRAINING)
            nextModelStatus = 'unavailable';
            if (!resolvedError) {
              resolvedError =
                rErr.message ||
                'ML model lifecycle state is not ready for inference (HTTP 503).';
            }
          } else if (rErr.isNetworkError || rErr.isTimeout) {
            nextModelStatus = 'unavailable';
            if (!resolvedError) {
              resolvedError = rErr.message;
            }
          } else {
            nextModelStatus = 'unavailable';
            if (!resolvedError) {
              resolvedError = rErr.message;
            }
          }
        } else {
          nextModelStatus = 'unavailable';
          if (!resolvedError) {
            resolvedError =
              rErr instanceof Error ? rErr.message : 'Readiness probe failed.';
          }
        }
      }

      // Commit consolidated state atomically
      setBackendStatus(nextBackendStatus);
      setModelStatus(nextModelStatus);
      setHealthData(nextHealthData);
      setReadinessData(nextReadinessData);
      setLastChecked(checkTime);

      if (isBackendHealthy && nextModelStatus === 'ready') {
        setLastSuccess(checkTime);
        setError(null);
      } else if (isBackendHealthy && nextModelStatus === 'not_loaded') {
        setLastSuccess(checkTime);
        setError('ML model is currently not loaded into memory.');
      } else {
        setError(resolvedError);
      }
    } catch (unexpectedErr) {
      if (currentSeq !== seqRef.current || controller.signal.aborted) {
        return;
      }
      setBackendStatus('unavailable');
      setModelStatus('unavailable');
      setError(
        unexpectedErr instanceof Error
          ? unexpectedErr.message
          : 'Unexpected error during connection health check.'
      );
      setLastChecked(new Date());
    } finally {
      if (currentSeq === seqRef.current) {
        inFlightRef.current = false;
        setIsLoading(false);
      }
    }
  }, []);

  // Polling lifecycle management
  useEffect(() => {
    // Initial mount probe
    performCheck(false);

    if (!enabled) {
      return () => {
        abortControllerRef.current?.abort();
      };
    }

    // Interval polling
    const timer = setInterval(() => {
      performCheck(false);
    }, pollingInterval);

    // Unmount cleanup: clear interval and cancel in-flight network requests
    return () => {
      clearInterval(timer);
      abortControllerRef.current?.abort();
    };
  }, [performCheck, pollingInterval, enabled]);

  const refetch = useCallback(async () => {
    await performCheck(true);
  }, [performCheck]);

  const value: HealthCheckState = {
    backendStatus,
    modelStatus,
    isLoading,
    lastChecked,
    lastSuccess,
    error,
    healthData,
    readinessData,
    refetch,
  };

  return (
    <HealthContext.Provider value={value}>
      {children}
    </HealthContext.Provider>
  );
}

/**
 * Access the shared health and readiness context.
 * Must be used within a HealthProvider.
 */
export function useHealthContext(): HealthCheckState {
  const context = useContext(HealthContext);
  if (!context) {
    throw new Error('useHealthContext must be used within a <HealthProvider>');
  }
  return context;
}
