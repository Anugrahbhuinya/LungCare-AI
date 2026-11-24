/**
 * Browser localStorage persistence service for LungCare AI settings.
 *
 * Enforces schema validation, safe fallbacks against malformed storage,
 * and cross-component/cross-tab reactive notification via events.
 */

import {
  AppSettings,
  DEFAULT_SETTINGS,
  SETTINGS_STORAGE_KEY,
  ALLOWED_ALPHAS,
  GradCamAlpha,
} from './types';

/**
 * Custom event name dispatched when settings change within the same window.
 */
export const SETTINGS_CHANGE_EVENT = 'lungcare_settings_changed';

/**
 * Checks whether localStorage is supported and accessible in the current execution environment.
 */
export function isLocalStorageAvailable(): boolean {
  try {
    if (typeof window === 'undefined' || !window.localStorage) {
      return false;
    }
    const testKey = '__lungcare_storage_test__';
    window.localStorage.setItem(testKey, testKey);
    window.localStorage.removeItem(testKey);
    return true;
  } catch {
    return false;
  }
}

/**
 * Type guard validating that a given value is one of the supported Grad-CAM alpha values.
 */
export function isValidAlpha(value: unknown): value is GradCamAlpha {
  return typeof value === 'number' && ALLOWED_ALPHAS.includes(value as GradCamAlpha);
}

/**
 * Retrieves the current application settings from localStorage.
 * Gracefully falls back to defaults if localStorage is unavailable, empty, or malformed.
 */
export function getSettings(): AppSettings {
  if (!isLocalStorageAvailable()) {
    return { ...DEFAULT_SETTINGS };
  }

  try {
    const raw = window.localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!raw) {
      return { ...DEFAULT_SETTINGS };
    }

    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object') {
      return { ...DEFAULT_SETTINGS };
    }

    const defaultAlpha = isValidAlpha(parsed.defaultAlpha)
      ? parsed.defaultAlpha
      : DEFAULT_SETTINGS.defaultAlpha;

    return {
      defaultAlpha,
    };
  } catch {
    // Malformed JSON, corrupted data, or access error: fall back safely without throwing
    return { ...DEFAULT_SETTINGS };
  }
}

/**
 * Saves updated settings to localStorage and notifies listeners.
 */
export function saveSettings(updates: Partial<AppSettings>): AppSettings {
  const current = getSettings();
  const next: AppSettings = {
    ...current,
    ...updates,
  };

  // Enforce validity of alpha parameter
  if (!isValidAlpha(next.defaultAlpha)) {
    next.defaultAlpha = DEFAULT_SETTINGS.defaultAlpha;
  }

  if (isLocalStorageAvailable()) {
    try {
      window.localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(next));
      if (typeof window.dispatchEvent === 'function') {
        window.dispatchEvent(new CustomEvent(SETTINGS_CHANGE_EVENT, { detail: next }));
      }
    } catch {
      // Storage quota exceeded or disabled: settings remain updated in-memory
    }
  }

  return next;
}

/**
 * Resets settings to default values and removes the storage key.
 */
export function resetSettings(): AppSettings {
  if (isLocalStorageAvailable()) {
    try {
      window.localStorage.removeItem(SETTINGS_STORAGE_KEY);
      if (typeof window.dispatchEvent === 'function') {
        window.dispatchEvent(
          new CustomEvent(SETTINGS_CHANGE_EVENT, { detail: DEFAULT_SETTINGS })
        );
      }
    } catch {
      // Ignore storage exception
    }
  }
  return { ...DEFAULT_SETTINGS };
}
