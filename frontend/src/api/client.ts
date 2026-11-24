import axios, { AxiosError, AxiosInstance, InternalAxiosRequestConfig } from 'axios';
import { ApiError } from '@/types/api';

/**
 * Default API Base URL with safe fallback for local development.
 * Reads VITE_API_BASE_URL if configured, otherwise falls back to http://localhost:8000/api/v1.
 */
export const DEFAULT_API_BASE_URL = 'http://localhost:8000/api/v1';

export const API_BASE_URL: string =
  (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_API_BASE_URL) ||
  DEFAULT_API_BASE_URL;

/**
 * Generate a cryptographically random UUIDv4 string.
 * Uses crypto.randomUUID() where supported, with a reliable fallback for older or non-secure contexts.
 */
export function generateRequestId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }

  // Compliant pseudo-random UUIDv4 generator fallback
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/**
 * Centralized, pre-configured Axios instance for LungCare AI.
 * Enforces base URL, 15s timeout, JSON accept headers, and request ID correlation.
 */
export const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000, // 15 seconds
  headers: {
    Accept: 'application/json',
  },
});

// Request Interceptor: Injects X-Request-ID correlation header without overwriting existing
apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    let existingId: string | undefined;

    if (config.headers) {
      if (typeof config.headers.get === 'function') {
        existingId = (config.headers.get('X-Request-ID') || config.headers.get('x-request-id')) as string | undefined;
      } else {
        existingId = (config.headers['X-Request-ID'] || config.headers['x-request-id']) as string | undefined;
      }
    }

    if (!existingId) {
      const generatedId = generateRequestId();
      if (config.headers) {
        if (typeof config.headers.set === 'function') {
          config.headers.set('X-Request-ID', generatedId);
        } else {
          config.headers['X-Request-ID'] = generatedId;
        }
      }
    }

    return config;
  },
  (error: unknown) => {
    return Promise.reject(error);
  }
);

// Response Interceptor: Transforms Axios and backend errors into standardized ApiError representations
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error: AxiosError) => {
    // 1. Resolve request ID from response body, response headers, or request headers
    let requestId: string | null = null;

    if (error.response?.data && typeof error.response.data === 'object' && 'request_id' in error.response.data) {
      requestId = (error.response.data as { request_id?: string | null }).request_id || null;
    }

    if (!requestId && error.response?.headers) {
      const headers = error.response.headers;
      if (typeof headers.get === 'function') {
        requestId = (headers.get('x-request-id') || headers.get('X-Request-ID')) as string | null;
      } else {
        requestId = ((headers['x-request-id'] || headers['X-Request-ID']) as string) || null;
      }
    }

    if (!requestId && error.config?.headers) {
      const reqHeaders = error.config.headers;
      if (typeof reqHeaders.get === 'function') {
        requestId = (reqHeaders.get('X-Request-ID') || reqHeaders.get('x-request-id')) as string | null;
      } else {
        requestId = ((reqHeaders['X-Request-ID'] || reqHeaders['x-request-id']) as string) || null;
      }
    }

    // 2. Handle HTTP Response Errors (4xx, 5xx)
    if (error.response) {
      const statusCode = error.response.status;
      const data = error.response.data;

      let message = `Server error (HTTP ${statusCode})`;
      let errorType = 'HTTP_ERROR';
      let details: unknown = undefined;

      if (data && typeof data === 'object') {
        // LungCare AI Centralized ErrorResponse contract ({ status: 'error', error: { message, error_type, details } })
        if ('error' in data && data.error && typeof data.error === 'object') {
          const errPayload = data.error as Record<string, unknown>;
          if (typeof errPayload.message === 'string' && errPayload.message.trim().length > 0) {
            message = errPayload.message;
          }
          if (typeof errPayload.error_type === 'string' && errPayload.error_type.trim().length > 0) {
            errorType = errPayload.error_type;
          }
          details = errPayload.details;
        } else if ('detail' in data) {
          // FastAPI default error payload ({ detail: "..." } or { detail: [{ loc, msg, type }] })
          if (typeof data.detail === 'string') {
            message = data.detail;
            errorType = 'VALIDATION_ERROR';
          } else if (Array.isArray(data.detail)) {
            message = 'Invalid request parameters.';
            errorType = 'VALIDATION_ERROR';
            details = data.detail;
          }
        }
      }

      // Defensive sanitization: Prevent leakage of raw stack traces or internal server details
      if (statusCode >= 500 && errorType === 'HTTP_ERROR') {
        message = 'An internal server error occurred. Please try again.';
        errorType = 'INTERNAL_SERVER_ERROR';
      }

      return Promise.reject(
        new ApiError({
          message,
          errorType,
          statusCode,
          requestId,
          details,
        })
      );
    }

    // 3. Handle Request Timeout Errors
    if (error.code === 'ECONNABORTED' || (error.message && error.message.toLowerCase().includes('timeout'))) {
      return Promise.reject(
        new ApiError({
          message: 'Request timed out after 15 seconds. Please check your connection and try again.',
          errorType: 'REQUEST_TIMEOUT',
          statusCode: 408,
          requestId,
          isTimeout: true,
        })
      );
    }

    // 4. Handle Canceled Requests (e.g. unmount or superseded polling)
    if (axios.isCancel(error) || error.code === 'ERR_CANCELED' || error.name === 'CanceledError') {
      return Promise.reject(
        new ApiError({
          message: 'Request was canceled.',
          errorType: 'REQUEST_CANCELED',
          statusCode: 0,
          requestId,
          isCanceled: true,
        })
      );
    }

    // 5. Handle Network Errors (No response received, offline, DNS or connection refused)
    return Promise.reject(
      new ApiError({
        message: 'Unable to connect to LungCare AI backend server. Please verify the API is running.',
        errorType: 'NETWORK_ERROR',
        statusCode: 0,
        requestId,
        isNetworkError: true,
      })
    );
  }
);

export default apiClient;
