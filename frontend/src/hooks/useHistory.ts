import { useState, useEffect, useCallback } from 'react';
import {
  type HistoryAnalysisRecord,
  isIndexedDBAvailable,
  listHistoryRecords,
  deleteHistoryRecord,
  clearHistoryRecords,
} from '@/services/history';

export interface UseHistoryReturn {
  records: HistoryAnalysisRecord[];
  isLoading: boolean;
  error: string | null;
  isAvailable: boolean;
  refresh: () => Promise<void>;
  deleteRecord: (id: string) => Promise<boolean>;
  clearAll: () => Promise<boolean>;
}

/**
 * Hook to interface with browser-local IndexedDB scan history.
 *
 * Provides reactive list of analysis records, loading/error states,
 * and deletion operations.
 */
export function useHistory(): UseHistoryReturn {
  const [records, setRecords] = useState<HistoryAnalysisRecord[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [isAvailable, setIsAvailable] = useState<boolean>(isIndexedDBAvailable());

  const loadRecords = useCallback(async () => {
    if (!isIndexedDBAvailable()) {
      setIsLoading(false);
      setIsAvailable(false);
      setError('Local storage (IndexedDB) is not accessible in this browser profile.');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const items = await listHistoryRecords();
      setRecords(items);
      setIsAvailable(true);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to retrieve scan history.';
      if (
        message.toLowerCase().includes('security') ||
        message.toLowerCase().includes('blocked') ||
        message.toLowerCase().includes('not supported') ||
        message.toLowerCase().includes('not accessible')
      ) {
        setIsAvailable(false);
      }
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadRecords();
  }, [loadRecords]);

  const deleteRecord = useCallback(
    async (id: string): Promise<boolean> => {
      try {
        await deleteHistoryRecord(id);
        setRecords((prev) => prev.filter((r) => r.id !== id));
        return true;
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : 'Failed to delete record.';
        setError(message);
        return false;
      }
    },
    []
  );

  const clearAll = useCallback(async (): Promise<boolean> => {
    try {
      await clearHistoryRecords();
      setRecords([]);
      return true;
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to clear history.';
      setError(message);
      return false;
    }
  }, []);

  return {
    records,
    isLoading,
    error,
    isAvailable,
    refresh: loadRecords,
    deleteRecord,
    clearAll,
  };
}

export default useHistory;
