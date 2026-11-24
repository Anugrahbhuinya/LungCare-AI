import { useState, useCallback } from 'react';
import {
  Server,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  Clock,
  Sliders,
  Database,
  Trash2,
  ShieldCheck,
  Info,
  Layers,
  RotateCcw,
} from 'lucide-react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { useHealthCheck } from '@/hooks/useHealthCheck';
import { useSettings } from '@/hooks/useSettings';
import { useHistory } from '@/hooks/useHistory';
import { API_BASE_URL } from '@/api/client';
import { ALLOWED_ALPHAS, DEFAULT_ALPHA, type GradCamAlpha } from '@/services/settings';
import { cn } from '@/utils/cn';

export function SettingsPage() {
  // 1. Live Backend Connectivity
  const {
    backendStatus,
    modelStatus,
    isLoading: isCheckingHealth,
    lastChecked,
    lastSuccess,
    error: healthError,
    refetch: refetchHealth,
  } = useHealthCheck();

  // 2. Persistent User Settings
  const { defaultAlpha, setDefaultAlpha, resetToDefaults } = useSettings();
  const [settingsFeedback, setSettingsFeedback] = useState<string | null>(null);

  // 3. Local Storage Management (IndexedDB)
  const {
    records,
    isLoading: isHistoryLoading,
    isAvailable: isDbAvailable,
    clearAll: clearHistory,
  } = useHistory();

  const [isConfirmingClear, setIsConfirmingClear] = useState<boolean>(false);
  const [isClearing, setIsClearing] = useState<boolean>(false);
  const [storageFeedback, setStorageFeedback] = useState<string | null>(null);
  const [storageError, setStorageError] = useState<string | null>(null);

  // Handle alpha change with temporary visual feedback
  const handleAlphaChange = useCallback(
    (alpha: GradCamAlpha) => {
      setDefaultAlpha(alpha);
      setSettingsFeedback(`Default alpha updated to ${alpha.toFixed(1)}`);
      setTimeout(() => setSettingsFeedback(null), 3000);
    },
    [setDefaultAlpha]
  );

  // Handle reset settings
  const handleResetSettings = useCallback(() => {
    resetToDefaults();
    setSettingsFeedback(`Preferences reset to default (${DEFAULT_ALPHA})`);
    setTimeout(() => setSettingsFeedback(null), 3000);
  }, [resetToDefaults]);

  // Handle history clear confirmation
  const handleConfirmClear = async () => {
    setIsClearing(true);
    setStorageError(null);
    setStorageFeedback(null);

    try {
      const success = await clearHistory();
      if (success) {
        setIsConfirmingClear(false);
        setStorageFeedback('All local scan records have been permanently cleared.');
        setTimeout(() => setStorageFeedback(null), 4000);
      } else {
        setStorageError('Failed to clear local scan records.');
      }
    } catch (err: unknown) {
      setStorageError(err instanceof Error ? err.message : 'Storage operation failed.');
    } finally {
      setIsClearing(false);
    }
  };

  // Status badge styling helper
  const isAllHealthy = backendStatus === 'healthy' && modelStatus === 'ready';

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900">
            System Settings
          </h2>
          <p className="text-slate-500 text-sm mt-0.5">
            Manage live backend connectivity, explainability visualization preferences, and browser-local storage.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {isAllHealthy ? (
            <Badge variant="clinical" size="md" className="flex items-center gap-1.5 font-medium">
              <CheckCircle2 className="w-3.5 h-3.5 text-clinical-600" />
              System Operational
            </Badge>
          ) : isCheckingHealth ? (
            <Badge variant="outline" size="md" className="flex items-center gap-1.5 text-slate-600 bg-slate-50">
              <RefreshCw className="w-3.5 h-3.5 animate-spin motion-reduce:animate-none text-slate-500" />
              Probing Connection...
            </Badge>
          ) : (
            <Badge variant="warning" size="md" className="flex items-center gap-1.5">
              <AlertCircle className="w-3.5 h-3.5 text-amber-600" />
              Degraded / Service Unavailable
            </Badge>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ========================================================= */}
        {/* Card 1: Backend API Connectivity & Runtime Configuration  */}
        {/* ========================================================= */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Server className="w-4 h-4 text-clinical-600" />
                <CardTitle>Backend API & Inference Service</CardTitle>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => refetchHealth()}
                isLoading={isCheckingHealth}
                leftIcon={<RefreshCw className={cn('w-3.5 h-3.5', isCheckingHealth && 'animate-spin motion-reduce:animate-none')} />}
              >
                Check Now
              </Button>
            </div>
            <CardDescription>
              Target FastAPI backend connectivity and live health probe monitoring.
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-4 text-xs">
            {/* Live Service Status Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div className="p-3 rounded-xl border border-slate-200/80 bg-slate-50 space-y-1.5">
                <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block">
                  Backend API Liveness
                </span>
                <div className="flex items-center gap-2">
                  {backendStatus === 'healthy' ? (
                    <Badge variant="success" size="sm" className="font-medium">
                      Healthy (HTTP 200)
                    </Badge>
                  ) : backendStatus === 'checking' ? (
                    <Badge variant="outline" size="sm" className="text-slate-500">
                      Checking...
                    </Badge>
                  ) : (
                    <Badge variant="danger" size="sm" className="font-medium">
                      Unavailable (Offline)
                    </Badge>
                  )}
                </div>
              </div>

              <div className="p-3 rounded-xl border border-slate-200/80 bg-slate-50 space-y-1.5">
                <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block">
                  ML Model Readiness
                </span>
                <div className="flex items-center gap-2">
                  {modelStatus === 'ready' ? (
                    <Badge variant="success" size="sm" className="font-medium">
                      Ready for Inference
                    </Badge>
                  ) : modelStatus === 'not_loaded' ? (
                    <Badge variant="warning" size="sm" className="font-medium">
                      Model Not Loaded
                    </Badge>
                  ) : modelStatus === 'checking' ? (
                    <Badge variant="outline" size="sm" className="text-slate-500">
                      Checking...
                    </Badge>
                  ) : (
                    <Badge variant="danger" size="sm" className="font-medium">
                      Unavailable
                    </Badge>
                  )}
                </div>
              </div>
            </div>

            {/* Read-Only Runtime API Endpoint */}
            <div className="space-y-1">
              <label htmlFor="api-base-url-input" className="font-semibold text-slate-700 block">
                Runtime API Base URL (Read-Only)
              </label>
              <input
                id="api-base-url-input"
                type="text"
                readOnly
                value={API_BASE_URL}
                className="w-full px-3 py-2 rounded-lg border border-slate-500 bg-slate-50 font-mono text-slate-700 select-all cursor-text text-xs focus:outline-none focus:ring-1 focus:ring-clinical-500"
              />
              <p className="text-[11px] text-slate-500 mt-1 leading-normal">
                Resolved from build environment (<code className="bg-slate-100 px-1 py-0.5 rounded font-mono text-[10px] text-slate-600">VITE_API_BASE_URL</code>) or local development fallback. Axios client enforces 15-second request timeouts and UUIDv4 correlation tracking.
              </p>
            </div>

            {/* Live Timestamp & Error Details */}
            <div className="pt-2 border-t border-slate-100 flex flex-col sm:flex-row sm:items-center justify-between text-[11px] text-slate-500 gap-1.5">
              <span className="flex items-center gap-1.5">
                <Clock className="w-3.5 h-3.5 text-slate-500" />
                Last Checked: {lastChecked ? lastChecked.toLocaleTimeString() : 'Never'}
              </span>
              {lastSuccess && (
                <span>
                  Last Success: {lastSuccess.toLocaleTimeString()}
                </span>
              )}
            </div>

            {healthError && (
              <div className="p-3 rounded-lg border border-rose-200 bg-rose-50 text-rose-800 text-xs flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
                <div className="leading-relaxed">
                  <span className="font-semibold">Connection Error: </span>
                  {healthError}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* ========================================================= */}
        {/* Card 2: User Explainability & Visualization Preferences   */}
        {/* ========================================================= */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Sliders className="w-4 h-4 text-clinical-600" />
                <CardTitle>Explainability & Visualization Defaults</CardTitle>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={handleResetSettings}
                leftIcon={<RotateCcw className="w-3.5 h-3.5" />}
                title="Reset preferences to default"
              >
                Reset Default
              </Button>
            </div>
            <CardDescription>
              Personalize client-side visualization defaults. Stored locally in your browser.
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-4 text-xs">
            {/* Default Alpha Preference Selector */}
            <div className="space-y-2 p-3.5 rounded-xl border border-slate-200/80 bg-slate-50">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div>
                  <span className="font-semibold text-slate-800 block">
                    Default Grad-CAM Overlay Alpha Blend
                  </span>
                  <span className="text-[11px] text-slate-500">
                    Initial opacity blend factor applied when generating Grad-CAM overlays for new analyses.
                  </span>
                </div>
                <Badge variant="outline" size="sm" className="font-mono text-[11px] bg-white self-start sm:self-auto">
                  Active Default: {defaultAlpha.toFixed(1)}
                </Badge>
              </div>

              {/* Interactive Alpha Selector Buttons */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1.5">
                {ALLOWED_ALPHAS.map((alpha) => {
                  const isSelected = defaultAlpha === alpha;
                  const labelMap: Record<number, string> = {
                    0.2: 'Subtle (20%)',
                    0.4: 'Standard (40%)',
                    0.6: 'Prominent (60%)',
                    0.8: 'Intense (80%)',
                  };

                  return (
                    <button
                      key={alpha}
                      type="button"
                      onClick={() => handleAlphaChange(alpha)}
                      aria-pressed={isSelected}
                      className={cn(
                        'px-3 py-2 rounded-lg text-xs font-medium transition-all text-center flex flex-col items-center justify-center gap-0.5 border',
                        isSelected
                          ? 'bg-clinical-700 text-white border-clinical-700 shadow-xs font-semibold ring-2 ring-clinical-500/20'
                          : 'bg-white text-slate-700 border-slate-200 hover:bg-slate-100/80 hover:border-slate-300'
                      )}
                    >
                      <span className="font-mono text-sm leading-none">{alpha.toFixed(1)}</span>
                      <span className={cn('text-[10px]', isSelected ? 'text-clinical-100' : 'text-slate-500')}>
                        {labelMap[alpha]}
                      </span>
                    </button>
                  );
                })}
              </div>

              {settingsFeedback && (
                <div className="text-[11px] text-clinical-700 font-medium flex items-center gap-1.5 pt-1">
                  <CheckCircle2 className="w-3.5 h-3.5 text-clinical-600" />
                  {settingsFeedback}
                </div>
              )}
            </div>

            {/* Explanation Note */}
            <div className="p-3 rounded-lg border border-slate-200/60 bg-white text-[11px] text-slate-600 flex items-start gap-2.5 leading-relaxed">
              <Info className="w-4 h-4 text-slate-400 shrink-0 mt-0.5" />
              <div>
                <span className="font-semibold text-slate-800">Session Isolation:</span>{' '}
                Modifying the default alpha applies to all new analysis scans and resets. Existing in-progress analyses or custom alpha adjustments made inside the active Grad-CAM Viewer remain unchanged.
              </div>
            </div>
          </CardContent>
        </Card>

        {/* ========================================================= */}
        {/* Card 3: Local Storage Management (IndexedDB)              */}
        {/* ========================================================= */}
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Database className="w-4 h-4 text-clinical-600" />
              <CardTitle>Browser Local Storage & Scan History</CardTitle>
            </div>
            <CardDescription>
              Manage persistent, browser-local IndexedDB storage and history retention.
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-4 text-xs">
            {/* Storage Metrics Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div className="p-3 rounded-xl border border-slate-200/80 bg-slate-50 space-y-1">
                <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block">
                  Storage Engine
                </span>
                <p className="font-mono font-medium text-slate-800 text-xs">
                  IndexedDB: lungcare-history
                </p>
                <div className="pt-0.5">
                  <Badge
                    variant={isDbAvailable ? 'success' : 'danger'}
                    size="sm"
                    className="text-[10px]"
                  >
                    {isDbAvailable ? 'Active & Accessible' : 'Unavailable'}
                  </Badge>
                </div>
              </div>

              <div className="p-3 rounded-xl border border-slate-200/80 bg-slate-50 space-y-1">
                <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block">
                  Saved Scan Records
                </span>
                <p className="font-semibold text-slate-800 text-sm">
                  {isHistoryLoading ? 'Loading...' : `${records.length} record${records.length === 1 ? '' : 's'}`}
                </p>
                <span className="text-[10px] text-slate-400 block">
                  Object Store: analyses
                </span>
              </div>
            </div>

            {/* Privacy Architecture Notice */}
            <div className="p-3 rounded-lg border border-slate-200/60 bg-white text-[11px] text-slate-600 flex items-start gap-2.5 leading-relaxed">
              <ShieldCheck className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
              <div>
                <span className="font-semibold text-slate-800">Local-First Privacy Architecture:</span>{' '}
                Scan history metadata (predictions, execution durations, request IDs) is stored strictly in your browser's IndexedDB. Radiograph images and heatmaps are never persisted. Clearing your scan history does not delete your application preferences.
              </div>
            </div>

            {/* Clear History Danger Zone */}
            <div className="p-3.5 rounded-xl border border-rose-100 bg-rose-50/40 space-y-2.5">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div>
                  <span className="font-semibold text-rose-950 block">
                    Clear Local Scan History
                  </span>
                  <span className="text-[11px] text-rose-700/80">
                    Permanently delete all locally persisted analysis records from IndexedDB.
                  </span>
                </div>
                {!isConfirmingClear && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setIsConfirmingClear(true)}
                    disabled={records.length === 0 || isHistoryLoading}
                    className="text-rose-700 border-rose-300 hover:bg-rose-100 self-start sm:self-auto shrink-0"
                    leftIcon={<Trash2 className="w-3.5 h-3.5 text-rose-600" />}
                  >
                    Clear All Scan History
                  </Button>
                )}
              </div>

              {/* Inline Confirmation Panel */}
              {isConfirmingClear && (
                <div className="p-3 rounded-lg bg-white border border-rose-200 space-y-2.5">
                  <div className="flex items-start gap-2 text-rose-900 text-xs">
                    <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
                    <span>
                      Are you sure you want to permanently delete all <strong>{records.length}</strong> saved analysis record{records.length === 1 ? '' : 's'}? This action cannot be reversed.
                    </span>
                  </div>
                  <div className="flex items-center gap-2 justify-end pt-1">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setIsConfirmingClear(false)}
                      disabled={isClearing}
                    >
                      Cancel
                    </Button>
                    <Button
                      variant="danger"
                      size="sm"
                      onClick={handleConfirmClear}
                      isLoading={isClearing}
                      leftIcon={<Trash2 className="w-3.5 h-3.5" />}
                    >
                      Yes, Clear All History
                    </Button>
                  </div>
                </div>
              )}

              {storageFeedback && (
                <div className="text-[11px] text-emerald-700 font-medium flex items-center gap-1.5 pt-1">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                  {storageFeedback}
                </div>
              )}

              {storageError && (
                <div className="text-[11px] text-rose-700 font-medium flex items-center gap-1.5 pt-1">
                  <AlertCircle className="w-3.5 h-3.5 text-rose-600" />
                  {storageError}
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* ========================================================= */}
        {/* Card 4: Fixed Model Architecture & Runtime Invariants     */}
        {/* ========================================================= */}
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-slate-600" />
              <CardTitle>Fixed Model Architecture & Parameters</CardTitle>
            </div>
            <CardDescription>
              Foundational model invariants and server-side visual explanation specifications.
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-3 text-xs text-slate-600">
            <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 border border-slate-100">
              <div className="space-y-0.5">
                <span className="font-medium text-slate-800 block">Neural Backbone</span>
                <span className="text-[11px] text-slate-400">1024-d GAP • BatchNorm • 0.4 Dropout • 4 Classes</span>
              </div>
              <Badge variant="outline" size="sm" className="font-mono bg-white">
                DenseNet121
              </Badge>
            </div>

            <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 border border-slate-100">
              <div className="space-y-0.5">
                <span className="font-medium text-slate-800 block">Input Tensor Dimensions</span>
                <span className="text-[11px] text-slate-400">TensorFlow/Keras DenseNet121 Normalization</span>
              </div>
              <Badge variant="outline" size="sm" className="font-mono bg-white">
                224 × 224 RGB
              </Badge>
            </div>

            <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 border border-slate-100">
              <div className="space-y-0.5">
                <span className="font-medium text-slate-800 block">Grad-CAM Colormap</span>
                <span className="text-[11px] text-slate-400">Server-Side Vectorized NumPy Implementation</span>
              </div>
              <Badge variant="outline" size="sm" className="font-mono bg-white">
                JET (Fixed)
              </Badge>
            </div>

            {/* Note on Invariants */}
            <div className="p-3 rounded-lg border border-slate-200/60 bg-white text-[11px] text-slate-500 flex items-start gap-2.5 leading-relaxed mt-2">
              <Info className="w-4 h-4 text-slate-400 shrink-0 mt-0.5" />
              <div>
                <span className="font-semibold text-slate-700">Architectural Invariants:</span>{' '}
                These specifications are intrinsic parameters of the trained neural network model and backend computer-vision pipeline. They are deterministic and cannot be modified at client runtime.
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export default SettingsPage;
