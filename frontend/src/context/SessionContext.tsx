import React, { createContext, useContext, useState, useCallback, useMemo } from 'react';
import { getDisplayClassName } from '@/types/prediction';

export interface SessionAnalysisRecord {
  id: string;
  timestamp: Date;
  status: 'success' | 'failed';
  predictedClass?: string;
  predictedLabel?: string;
  confidence?: number;
  fileName: string;
  error?: string;
  durationMs?: number;
  requestId?: string | null;
}

export interface SessionStats {
  totalAnalyses: number;
  successfulAnalyses: number;
  failedAnalyses: number;
  lastAnalysisTimestamp: Date | null;
  recentActivity: SessionAnalysisRecord[];
}

export interface SessionContextType extends SessionStats {
  recordAnalysisSuccess: (entry: {
    predictedClass: string;
    predictedLabel?: string;
    confidence: number;
    fileName: string;
    durationMs?: number;
    requestId?: string | null;
  }) => void;
  recordAnalysisFailure: (entry: {
    fileName: string;
    error: string;
    requestId?: string | null;
  }) => void;
  clearSessionHistory: () => void;
}

const SessionContext = createContext<SessionContextType | null>(null);

export interface SessionProviderProps {
  children: React.ReactNode;
}

/**
 * Provider maintaining in-memory session analysis metrics and activity history.
 *
 * Persists strictly within the active browser session across client-side route navigation.
 * No historical scans or mock records are fabricated.
 */
export function SessionProvider({ children }: SessionProviderProps) {
  const [recentActivity, setRecentActivity] = useState<SessionAnalysisRecord[]>([]);

  const recordAnalysisSuccess = useCallback(
    (entry: {
      predictedClass: string;
      predictedLabel?: string;
      confidence: number;
      fileName: string;
      durationMs?: number;
      requestId?: string | null;
    }) => {
      const newRecord: SessionAnalysisRecord = {
        id: `sess-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        timestamp: new Date(),
        status: 'success',
        predictedClass: entry.predictedClass,
        predictedLabel: entry.predictedLabel || getDisplayClassName(entry.predictedClass),
        confidence: entry.confidence,
        fileName: entry.fileName,
        durationMs: entry.durationMs,
        requestId: entry.requestId,
      };

      setRecentActivity((prev) => [newRecord, ...prev].slice(0, 20));
    },
    []
  );

  const recordAnalysisFailure = useCallback(
    (entry: {
      fileName: string;
      error: string;
      requestId?: string | null;
    }) => {
      const newRecord: SessionAnalysisRecord = {
        id: `sess-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        timestamp: new Date(),
        status: 'failed',
        fileName: entry.fileName,
        error: entry.error,
        requestId: entry.requestId,
      };

      setRecentActivity((prev) => [newRecord, ...prev].slice(0, 20));
    },
    []
  );

  const clearSessionHistory = useCallback(() => {
    setRecentActivity([]);
  }, []);

  // Compute aggregated session metrics derived directly from genuine events
  const stats: SessionStats = useMemo(() => {
    const totalAnalyses = recentActivity.length;
    const successfulAnalyses = recentActivity.filter((r) => r.status === 'success').length;
    const failedAnalyses = recentActivity.filter((r) => r.status === 'failed').length;
    const lastAnalysisTimestamp = recentActivity.length > 0 ? recentActivity[0].timestamp : null;

    return {
      totalAnalyses,
      successfulAnalyses,
      failedAnalyses,
      lastAnalysisTimestamp,
      recentActivity,
    };
  }, [recentActivity]);

  const value: SessionContextType = useMemo(
    () => ({
      ...stats,
      recordAnalysisSuccess,
      recordAnalysisFailure,
      clearSessionHistory,
    }),
    [stats, recordAnalysisSuccess, recordAnalysisFailure, clearSessionHistory]
  );

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

/**
 * Access the application session analysis telemetry and activity log.
 */
export function useSessionStats(): SessionContextType {
  const context = useContext(SessionContext);
  if (!context) {
    throw new Error('useSessionStats must be used within a <SessionProvider>');
  }
  return context;
}

export default SessionContext;
