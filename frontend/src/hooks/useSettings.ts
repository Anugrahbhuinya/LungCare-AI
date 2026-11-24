import { useState, useEffect, useCallback } from 'react';
import {
  type AppSettings,
  type GradCamAlpha,
  getSettings,
  saveSettings,
  resetSettings,
  SETTINGS_CHANGE_EVENT,
} from '@/services/settings';

export interface UseSettingsReturn {
  settings: AppSettings;
  defaultAlpha: GradCamAlpha;
  setDefaultAlpha: (alpha: GradCamAlpha) => void;
  resetToDefaults: () => void;
}

/**
 * Hook to manage persistent user preferences across the LungCare AI application.
 * Synchronizes reactively via localStorage and custom change events.
 */
export function useSettings(): UseSettingsReturn {
  const [settings, setSettings] = useState<AppSettings>(() => getSettings());

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    const handleUpdate = () => {
      setSettings(getSettings());
    };

    // Listen to internal custom event for same-tab updates
    window.addEventListener(SETTINGS_CHANGE_EVENT, handleUpdate);
    // Listen to native storage event for cross-tab updates
    window.addEventListener('storage', handleUpdate);

    return () => {
      window.removeEventListener(SETTINGS_CHANGE_EVENT, handleUpdate);
      window.removeEventListener('storage', handleUpdate);
    };
  }, []);

  const setDefaultAlpha = useCallback((alpha: GradCamAlpha) => {
    const updated = saveSettings({ defaultAlpha: alpha });
    setSettings(updated);
  }, []);

  const resetToDefaults = useCallback(() => {
    const defaults = resetSettings();
    setSettings(defaults);
  }, []);

  return {
    settings,
    defaultAlpha: settings.defaultAlpha,
    setDefaultAlpha,
    resetToDefaults,
  };
}

export default useSettings;
