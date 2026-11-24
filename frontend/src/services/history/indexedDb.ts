/**
 * Browser-local IndexedDB storage layer for LungCare AI scan history.
 *
 * Implements persistent, client-side metadata-only storage for successful
 * chest radiograph analyses. Does not store full image blobs or patient identifiers.
 */

import {
  HistoryAnalysisRecord,
  HistoryStorageError,
  isValidHistoryRecord,
} from "./types";

export const DB_NAME = "lungcare-history";
export const STORE_NAME = "analyses";
export const DB_VERSION = 1;
export const TIMESTAMP_INDEX = "timestamp";

let isDbAccessBlocked = false;

/**
 * Mark IndexedDB as blocked in the current session (e.g. after a SecurityError)
 * to prevent repeated unhandled initialization loops.
 */
export function setIndexedDBBlocked() {
  isDbAccessBlocked = true;
}

/**
 * Reset blocked state (useful for tests or environment recovery).
 */
export function resetIndexedDBBlockedState() {
  isDbAccessBlocked = false;
}

/**
 * Checks if IndexedDB is available and accessible in the current execution environment.
 */
export function isIndexedDBAvailable(): boolean {
  if (isDbAccessBlocked) {
    return false;
  }
  try {
    if (typeof window === "undefined" || !window.indexedDB) {
      return false;
    }
    return true;
  } catch {
    return false;
  }
}

/**
 * Generates a stable, deterministic record identifier for an analysis result.
 * Avoids using raw file names to prevent patient identifier leakage.
 */
export function getAnalysisRecordId(
  requestId: string | null | undefined,
  timestamp: string,
  fileName?: string,
  fileSize?: number,
  predictedClass?: string,
): string {
  void fileName;
  void fileSize;
  if (requestId && requestId.trim().length > 0) {
    return `analysis-${requestId.trim()}`;
  }
  // Neutral deterministic fallback avoiding patient identifiers in filenames
  const safeTime = (timestamp || new Date().toISOString()).replace(
    /[^a-zA-Z0-9_-]/g,
    "_",
  );
  const safeClass = (predictedClass || "unknown").replace(
    /[^a-zA-Z0-9_-]/g,
    "_",
  );
  return `analysis-${safeTime}-${safeClass}`;
}

/**
 * Opens or initializes the 'lungcare-history' IndexedDB database.
 */
export function openHistoryDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (!isIndexedDBAvailable()) {
      return reject(
        new HistoryStorageError(
          "IndexedDB is not supported or accessible in this environment.",
        ),
      );
    }

    let settled = false;

    const safeReject = (err: HistoryStorageError) => {
      if (!settled) {
        settled = true;
        reject(err);
      }
    };

    const safeResolve = (db: IDBDatabase) => {
      if (!settled) {
        settled = true;
        resolve(db);
      }
    };

    try {
      const request = window.indexedDB.open(DB_NAME, DB_VERSION);

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;

        if (!db.objectStoreNames.contains(STORE_NAME)) {
          const store = db.createObjectStore(STORE_NAME, { keyPath: "id" });
          store.createIndex(TIMESTAMP_INDEX, "timestamp", { unique: false });
        }
      };

      request.onsuccess = () => {
        const db = request.result;

        // F-03: Close database when a version upgrade is requested by another tab/connection
        db.onversionchange = () => {
          try {
            db.close();
          } catch {}
        };

        safeResolve(db);
      };

      request.onerror = () => {
        const errorMsg = request.error?.message || "Unknown error";
        if (request.error?.name === "SecurityError") {
          setIndexedDBBlocked();
        }
        safeReject(
          new HistoryStorageError(
            `Failed to open IndexedDB '${DB_NAME}': ${errorMsg}`,
            request.error,
          ),
        );
      };

      request.onblocked = () => {
        safeReject(
          new HistoryStorageError(
            `IndexedDB database '${DB_NAME}' open request blocked by another tab or connection.`,
          ),
        );
      };
    } catch (err: unknown) {
      const isSecurity =
        err instanceof Error &&
        (err.name === "SecurityError" || err.message.includes("Security"));
      if (isSecurity) {
        setIndexedDBBlocked();
      }
      safeReject(
        new HistoryStorageError(
          "Unexpected error attempting to open IndexedDB.",
          err,
        ),
      );
    }
  });
}

/**
 * Persists a validated analysis record into IndexedDB.
 * Resolves only after the transaction's oncomplete event fires.
 */
export async function saveHistoryRecord(
  record: HistoryAnalysisRecord,
): Promise<void> {
  if (!isValidHistoryRecord(record)) {
    throw new HistoryStorageError(
      "Cannot persist invalid or malformed history record.",
    );
  }

  const db = await openHistoryDB();

  return new Promise((resolve, reject) => {
    let settled = false;

    const safeClose = () => {
      try {
        db.close();
      } catch {}
    };

    const safeResolve = () => {
      if (!settled) {
        settled = true;
        safeClose();
        resolve();
      }
    };

    const safeReject = (err: HistoryStorageError) => {
      if (!settled) {
        settled = true;
        safeClose();
        reject(err);
      }
    };

    try {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);

      const request = store.put(record);

      request.onerror = () => {
        safeReject(
          new HistoryStorageError(
            `Failed to insert history record: ${request.error?.message || "Transaction failed"}`,
            request.error,
          ),
        );
      };

      // Resolve ONLY upon successful transaction commit
      tx.oncomplete = () => {
        safeResolve();
      };

      tx.onerror = () => {
        safeReject(
          new HistoryStorageError(
            `Transaction error inserting history record: ${tx.error?.message || "Unknown error"}`,
            tx.error,
          ),
        );
      };

      tx.onabort = () => {
        safeReject(
          new HistoryStorageError(
            `Transaction aborted inserting history record: ${tx.error?.message || "Storage aborted"}`,
            tx.error,
          ),
        );
      };
    } catch (err) {
      safeReject(
        new HistoryStorageError(
          "Exception executing saveHistoryRecord transaction.",
          err,
        ),
      );
    }
  });
}

/**
 * Retrieves a single history record by its unique identifier.
 * Returns null if the record does not exist or fails runtime validation.
 */
export async function getHistoryRecord(
  id: string,
): Promise<HistoryAnalysisRecord | null> {
  const db = await openHistoryDB();

  return new Promise((resolve, reject) => {
    let settled = false;

    const safeClose = () => {
      try {
        db.close();
      } catch {}
    };

    const safeResolve = (val: HistoryAnalysisRecord | null) => {
      if (!settled) {
        settled = true;
        safeClose();
        resolve(val);
      }
    };

    const safeReject = (err: HistoryStorageError) => {
      if (!settled) {
        settled = true;
        safeClose();
        reject(err);
      }
    };

    try {
      const tx = db.transaction(STORE_NAME, "readonly");
      const store = tx.objectStore(STORE_NAME);
      const request = store.get(id);

      let recordResult: HistoryAnalysisRecord | null = null;

      request.onsuccess = () => {
        const raw = request.result;
        recordResult = raw && isValidHistoryRecord(raw) ? raw : null;
      };

      request.onerror = () => {
        safeReject(
          new HistoryStorageError(
            `Failed to retrieve record '${id}': ${request.error?.message || "Query failed"}`,
            request.error,
          ),
        );
      };

      tx.oncomplete = () => {
        safeResolve(recordResult);
      };

      tx.onerror = () => {
        safeReject(
          new HistoryStorageError(
            `Transaction error retrieving record '${id}': ${tx.error?.message || "Unknown error"}`,
            tx.error,
          ),
        );
      };

      tx.onabort = () => {
        safeReject(
          new HistoryStorageError(
            `Transaction aborted retrieving record '${id}': ${tx.error?.message || "Aborted"}`,
            tx.error,
          ),
        );
      };
    } catch (err) {
      safeReject(
        new HistoryStorageError("Exception executing getHistoryRecord.", err),
      );
    }
  });
}

/**
 * Lists all history records, sorted newest first.
 * Filters out malformed records at runtime without throwing.
 */
export async function listHistoryRecords(): Promise<HistoryAnalysisRecord[]> {
  const db = await openHistoryDB();

  return new Promise((resolve, reject) => {
    let settled = false;

    const safeClose = () => {
      try {
        db.close();
      } catch {}
    };

    const safeResolve = (records: HistoryAnalysisRecord[]) => {
      if (!settled) {
        settled = true;
        safeClose();
        resolve(records);
      }
    };

    const safeReject = (err: HistoryStorageError) => {
      if (!settled) {
        settled = true;
        safeClose();
        reject(err);
      }
    };

    try {
      const tx = db.transaction(STORE_NAME, "readonly");
      const store = tx.objectStore(STORE_NAME);
      const records: HistoryAnalysisRecord[] = [];

      // If the timestamp index exists, traverse with cursor direction 'prev' for newest-first order
      if (store.indexNames.contains(TIMESTAMP_INDEX)) {
        const index = store.index(TIMESTAMP_INDEX);
        const cursorRequest = index.openCursor(null, "prev");

        cursorRequest.onsuccess = (event) => {
          const cursor = (event.target as IDBRequest<IDBCursorWithValue>)
            .result;
          if (cursor) {
            if (isValidHistoryRecord(cursor.value)) {
              records.push(cursor.value);
            }
            cursor.continue();
          }
        };

        cursorRequest.onerror = () => {
          safeReject(
            new HistoryStorageError(
              `Cursor error listing records: ${cursorRequest.error?.message || "Query failed"}`,
              cursorRequest.error,
            ),
          );
        };
      } else {
        // Fallback: read all and sort descending by timestamp in memory
        const getAllRequest = store.getAll();

        getAllRequest.onsuccess = () => {
          const all = (getAllRequest.result as unknown[]) || [];
          for (const item of all) {
            if (isValidHistoryRecord(item)) {
              records.push(item);
            }
          }
          records.sort((a, b) => {
            const timeA = new Date(a.timestamp).getTime();
            const timeB = new Date(b.timestamp).getTime();
            return (isNaN(timeB) ? 0 : timeB) - (isNaN(timeA) ? 0 : timeA);
          });
        };

        getAllRequest.onerror = () => {
          safeReject(
            new HistoryStorageError(
              `Failed to fetch all records: ${getAllRequest.error?.message || "Query failed"}`,
              getAllRequest.error,
            ),
          );
        };
      }

      tx.oncomplete = () => {
        safeResolve(records);
      };

      tx.onerror = () => {
        safeReject(
          new HistoryStorageError(
            `Transaction error listing records: ${tx.error?.message || "Unknown error"}`,
            tx.error,
          ),
        );
      };

      tx.onabort = () => {
        safeReject(
          new HistoryStorageError(
            `Transaction aborted listing records: ${tx.error?.message || "Aborted"}`,
            tx.error,
          ),
        );
      };
    } catch (err) {
      safeReject(
        new HistoryStorageError("Exception executing listHistoryRecords.", err),
      );
    }
  });
}

/**
 * Deletes a single history record by its unique identifier.
 * Resolves only after the transaction commits.
 */
export async function deleteHistoryRecord(id: string): Promise<void> {
  const db = await openHistoryDB();

  return new Promise((resolve, reject) => {
    let settled = false;

    const safeClose = () => {
      try {
        db.close();
      } catch {}
    };

    const safeResolve = () => {
      if (!settled) {
        settled = true;
        safeClose();
        resolve();
      }
    };

    const safeReject = (err: HistoryStorageError) => {
      if (!settled) {
        settled = true;
        safeClose();
        reject(err);
      }
    };

    try {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);
      const request = store.delete(id);

      request.onerror = () => {
        safeReject(
          new HistoryStorageError(
            `Failed to delete record '${id}': ${request.error?.message || "Delete failed"}`,
            request.error,
          ),
        );
      };

      tx.oncomplete = () => {
        safeResolve();
      };

      tx.onerror = () => {
        safeReject(
          new HistoryStorageError(
            `Transaction error deleting record '${id}': ${tx.error?.message || "Unknown error"}`,
            tx.error,
          ),
        );
      };

      tx.onabort = () => {
        safeReject(
          new HistoryStorageError(
            `Transaction aborted deleting record '${id}': ${tx.error?.message || "Aborted"}`,
            tx.error,
          ),
        );
      };
    } catch (err) {
      safeReject(
        new HistoryStorageError(
          "Exception executing deleteHistoryRecord.",
          err,
        ),
      );
    }
  });
}

/**
 * Clears all records from the history object store.
 * Resolves only after the transaction commits.
 */
export async function clearHistoryRecords(): Promise<void> {
  const db = await openHistoryDB();

  return new Promise((resolve, reject) => {
    let settled = false;

    const safeClose = () => {
      try {
        db.close();
      } catch {}
    };

    const safeResolve = () => {
      if (!settled) {
        settled = true;
        safeClose();
        resolve();
      }
    };

    const safeReject = (err: HistoryStorageError) => {
      if (!settled) {
        settled = true;
        safeClose();
        reject(err);
      }
    };

    try {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);
      const request = store.clear();

      request.onerror = () => {
        safeReject(
          new HistoryStorageError(
            `Failed to clear history store: ${request.error?.message || "Clear failed"}`,
            request.error,
          ),
        );
      };

      tx.oncomplete = () => {
        safeResolve();
      };

      tx.onerror = () => {
        safeReject(
          new HistoryStorageError(
            `Transaction error clearing history store: ${tx.error?.message || "Unknown error"}`,
            tx.error,
          ),
        );
      };

      tx.onabort = () => {
        safeReject(
          new HistoryStorageError(
            `Transaction aborted clearing history store: ${tx.error?.message || "Aborted"}`,
            tx.error,
          ),
        );
      };
    } catch (err) {
      safeReject(
        new HistoryStorageError(
          "Exception executing clearHistoryRecords.",
          err,
        ),
      );
    }
  });
}

/**
 * Returns the total count of records stored in the history database.
 */
export async function getHistoryCount(): Promise<number> {
  const db = await openHistoryDB();

  return new Promise((resolve, reject) => {
    let settled = false;

    const safeClose = () => {
      try {
        db.close();
      } catch {}
    };

    const safeResolve = (count: number) => {
      if (!settled) {
        settled = true;
        safeClose();
        resolve(count);
      }
    };

    const safeReject = (err: HistoryStorageError) => {
      if (!settled) {
        settled = true;
        safeClose();
        reject(err);
      }
    };

    try {
      const tx = db.transaction(STORE_NAME, "readonly");
      const store = tx.objectStore(STORE_NAME);
      const request = store.count();

      let countResult = 0;

      request.onsuccess = () => {
        countResult = request.result || 0;
      };

      request.onerror = () => {
        safeReject(
          new HistoryStorageError(
            `Failed to count history records: ${request.error?.message || "Count failed"}`,
            request.error,
          ),
        );
      };

      tx.oncomplete = () => {
        safeResolve(countResult);
      };

      tx.onerror = () => {
        safeReject(
          new HistoryStorageError(
            `Transaction error counting history records: ${tx.error?.message || "Unknown error"}`,
            tx.error,
          ),
        );
      };

      tx.onabort = () => {
        safeReject(
          new HistoryStorageError(
            `Transaction aborted counting history records: ${tx.error?.message || "Aborted"}`,
            tx.error,
          ),
        );
      };
    } catch (err) {
      safeReject(
        new HistoryStorageError("Exception executing getHistoryCount.", err),
      );
    }
  });
}
