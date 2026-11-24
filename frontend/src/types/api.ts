/**
 * API Type Contracts for LungCare AI
 * Matches backend schemas from backend/app/schemas/error.py
 */

export interface BackendErrorDetail {
  message: string;
  error_type: string;
  details?: unknown;
}

export interface BackendErrorResponse {
  status: 'error';
  error: BackendErrorDetail;
  timestamp: string;
  request_id?: string | null;
}

export interface ApiErrorData {
  message: string;
  errorType: string;
  statusCode: number;
  requestId: string | null;
  details?: unknown;
  isNetworkError?: boolean;
  isTimeout?: boolean;
  isCanceled?: boolean;
}

export class ApiError extends Error {
  readonly errorType: string;
  readonly statusCode: number;
  readonly requestId: string | null;
  readonly details?: unknown;
  readonly isNetworkError: boolean;
  readonly isTimeout: boolean;
  readonly isCanceled: boolean;

  constructor(data: ApiErrorData) {
    super(data.message);
    this.name = 'ApiError';
    this.errorType = data.errorType;
    this.statusCode = data.statusCode;
    this.requestId = data.requestId;
    this.details = data.details;
    this.isNetworkError = Boolean(data.isNetworkError);
    this.isTimeout = Boolean(data.isTimeout);
    this.isCanceled = Boolean(data.isCanceled);

    // Maintain prototype chain across transpilation targets
    Object.setPrototypeOf(this, ApiError.prototype);
  }
}

/**
 * Type guard to check if an unknown error is an ApiError instance.
 */
export function isApiError(error: unknown): error is ApiError {
  return error instanceof ApiError;
}
