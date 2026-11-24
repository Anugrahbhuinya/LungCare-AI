/**
 * Type definitions for LungCare AI persistent user preferences.
 * Phase 3.8.1: Settings Page & User Preferences.
 */

export type GradCamAlpha = 0.2 | 0.4 | 0.6 | 0.8;

export const ALLOWED_ALPHAS: readonly GradCamAlpha[] = [0.2, 0.4, 0.6, 0.8] as const;

export const DEFAULT_ALPHA: GradCamAlpha = 0.4;

export const SETTINGS_STORAGE_KEY = 'lungcare_settings';

export interface AppSettings {
  /**
   * Default alpha blend factor used for Grad-CAM heatmap overlays on new analyses.
   */
  defaultAlpha: GradCamAlpha;
}

export const DEFAULT_SETTINGS: AppSettings = {
  defaultAlpha: DEFAULT_ALPHA,
};
