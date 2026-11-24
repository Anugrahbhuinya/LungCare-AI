import { useContext, useState, useEffect, useRef, useCallback } from 'react';
import { HealthContext } from '@/context/HealthContext';
import { getHealth, getReadiness } from '@/api/health';
import type {
  BackendStatus,
  ModelStatus,
  HealthCheckState,
  HealthResponse,
  ReadinessResponse,
} from '@/types/health';
import { ApiError } from '@/types/api';

export interface UseHealthCheckOptions {
  /**
   * If true, forces the hook to maintain an isolated independent polling cycle
   * rather than subscribing to the shared application HealthProvider context.
   * Defaults to false.
   */
  standalone?: boolean;
  /**
   * Polling interval in milliseconds for standalone mode. Defaults to 30000ms.
   */
  pollingInterval?: number;
  /**
   * Whether polling is enabled in standalone mode. Defaults to true.
   */
  enabled?: boolean;
}

/**
 * Isolated hook implementation for standalone or un-contextualized execution.
 */
function useStandaloneHealthCheck(options?: UseHealthCheckOptions): HealthCheckState {
  const pollingInterval = options?.pollingInterval ?? 30000;
  const enabled = options?.enabled ?? true;

  const [backendStatus, setBackendStatus] = useState<BackendStatus>('checking');
  const [modelStatus, setModelStatus] = useState<ModelStatus>('checking');
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);
  const [lastSuccess, setLastSuccess] = useState<Date | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [healthData, setHealthData] = useState<HealthResponse | null>(null);
  const [readinessData, setReadinessData] = useState<ReadinessResponse | null>(null);

  const inFlightRef = useRef<boolean>(false);
  const seqRef = useRef<number>(0);
  const abortControllerRef = useRef<AbortController | null>(null);

  const performCheck = useCallback(async (isManual = false) => {
    if (inFlightRef.current) {
      if (!isManual) return;
      abortControllerRef.current?.abort();
    }

    const controller = new AbortController();
    abortControllerRef.current = controller;
    inFlightRef.current = true;
    const currentSeq = ++seqRef.current;

    setIsLoading(true);

    try {
      const [healthResult, readinessResult] = await Promise.allSettled([
        getHealth(controller.signal),
        getReadiness(controller.signal),
      ]);

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

      if (healthResult.status === 'fulfilled') {
        nextBackendStatus = 'healthy';
        nextHealthData = healthResult.value;
        isBackendHealthy = true;
      } else {
        const hErr = healthResult.reason;
        if (hErr instanceof ApiError && hErr.isCanceled) {
          return;
        }
        nextBackendStatus = 'unavailable';
        resolvedError =
          hErr instanceof Error
            ? hErr.message
            : 'Unable to connect to LungCare AI backend service.';
      }

      if (readinessResult.status === 'fulfilled') {
        nextReadinessData = readinessResult.value;
        if (nextReadinessData.model_loaded) {
          nextModelStatus = 'ready';
        } else {
          nextModelStatus = 'not_loaded';
        }
      } else {
        const rErr = readinessResult.reason;
        if (rErr instanceof ApiError && rErr.isCanceled) {
          return;
        }

        if (rErr instanceof ApiError) {
          if (rErr.statusCode === 503) {
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

  useEffect(() => {
    performCheck(false);

    if (!enabled) {
      return () => {
        abortControllerRef.current?.abort();
      };
    }

    const timer = setInterval(() => {
      performCheck(false);
    }, pollingInterval);

    return () => {
      clearInterval(timer);
      abortControllerRef.current?.abort();
    };
  }, [performCheck, pollingInterval, enabled]);

  const refetch = useCallback(async () => {
    await performCheck(true);
  }, [performCheck]);

  return {
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
}

/**
 * Live backend and ML model connection status hook.
 *
 * Automatically consumes the shared application <HealthProvider> context
 * when available, preventing redundant polling loops and duplicated API calls
 * between UI components (such as Header and Dashboard).
 *
 * If used outside a HealthProvider or with { standalone: true }, initializes
 * an isolated polling instance with independent lifecycle management.
 *
 * @param options Configuration options for polling and standalone execution.
 * @returns HealthCheckState containing live connection states, metadata, and refetch handler.
 */
export function useHealthCheck(options?: UseHealthCheckOptions): HealthCheckState {
  const sharedContext = useContext(HealthContext);

  // If inside HealthProvider and standalone not explicitly requested, use shared state
  if (sharedContext && !options?.standalone) {
    return sharedContext;
  }

  // Fallback to standalone execution (e.g. for isolated tests or decoupled components)
  return useStandaloneHealthCheck(options);
}

export default useHealthCheck;
