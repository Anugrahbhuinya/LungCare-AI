import { useNavigate } from 'react-router-dom';
import {
  Activity,
  Scan,
  ShieldCheck,
  Clock,
  ArrowRight,
  History,
  FileText,
  RefreshCw,
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  XCircle,
} from 'lucide-react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { useHealthCheck } from '@/hooks/useHealthCheck';
import { useSessionStats } from '@/context/SessionContext';

export function DashboardPage() {
  const navigate = useNavigate();
  const {
    backendStatus,
    modelStatus,
    isLoading: isHealthLoading,
    lastChecked,
    error: healthError,
    refetch: refetchHealth,
  } = useHealthCheck();

  const {
    totalAnalyses,
    successfulAnalyses,
    failedAnalyses,
    lastAnalysisTimestamp,
    recentActivity,
  } = useSessionStats();

  const formattedLastChecked = lastChecked
    ? lastChecked.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    : null;

  const formattedLastAnalysis = lastAnalysisTimestamp
    ? lastAnalysisTimestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    : null;

  const successRate =
    totalAnalyses > 0 ? ((successfulAnalyses / totalAnalyses) * 100).toFixed(0) : null;

  return (
    <div className="space-y-6">
      {/* Real-time Connection Notice Banner (when error or unreadiness is detected) */}
      {healthError && (
        <div
          className={`p-4 rounded-xl border flex items-start gap-3 transition-colors ${
            backendStatus === 'unavailable' || modelStatus === 'unavailable'
              ? 'bg-rose-50/70 border-rose-200 text-rose-900'
              : 'bg-amber-50/70 border-amber-200 text-amber-900'
          }`}
        >
          <AlertCircle
            className={`w-5 h-5 shrink-0 mt-0.5 ${
              backendStatus === 'unavailable' || modelStatus === 'unavailable'
                ? 'text-rose-600'
                : 'text-amber-600'
            }`}
          />
          <div className="flex-1 min-w-0">
            <h4 className="text-sm font-semibold">
              {backendStatus === 'unavailable'
                ? 'Backend Service Connection Notice'
                : modelStatus === 'unavailable'
                ? 'ML Model Service Notice'
                : 'ML Model Not Ready'}
            </h4>
            <p className="text-xs mt-0.5 opacity-90 leading-relaxed">
              {healthError}
            </p>
            <div className="mt-2.5 flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => refetchHealth()}
                disabled={isHealthLoading}
                className="text-xs py-1 h-auto bg-white"
                leftIcon={<RefreshCw className={`w-3 h-3 ${isHealthLoading ? 'animate-spin motion-reduce:animate-none' : ''}`} />}
              >
                Retry Connection
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Hero Welcome Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-slate-800 to-clinical-950 rounded-2xl p-6 sm:p-8 text-white shadow-md relative overflow-hidden">
        <div className="relative z-10 max-w-2xl space-y-3">
          <Badge
            variant="clinical"
            size="sm"
            className="bg-clinical-500/20 text-clinical-300 border-clinical-400/30"
          >
            Phase 3.6 • Session Telemetry Active
          </Badge>
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight">
            Radiology Research Workstation
          </h2>
          <p className="text-slate-300 text-sm sm:text-base leading-relaxed">
            Welcome to LungCare AI. Perform real-time four-class chest radiograph classification
            powered by DenseNet121 transfer learning and visual Grad-CAM explainability.
          </p>
          <div className="pt-2 flex flex-wrap gap-3">
            <Button
              variant="primary"
              size="md"
              leftIcon={<Scan className="w-4 h-4" />}
              onClick={() => navigate('/analysis')}
            >
              Start New Analysis
            </Button>
            <Button
              variant="outline"
              size="md"
              className="bg-slate-800/60 border-slate-700 text-white hover:bg-slate-800 hover:text-white"
              onClick={() => navigate('/model-info')}
            >
              Model Specifications
            </Button>
          </div>
        </div>

        <div className="absolute right-0 bottom-0 translate-x-12 translate-y-12 w-80 h-80 rounded-full bg-clinical-500/10 blur-3xl pointer-events-none" />
      </div>

      {/* Dynamic Session & Telemetry Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Card 1: Session Total Analyses */}
        <Card hoverEffect>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
              Session Analyses
            </CardTitle>
            <Activity className="w-4 h-4 text-clinical-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-slate-900 font-mono">
              {totalAnalyses}
            </div>
            <p className="text-xs text-slate-500 mt-1">
              {totalAnalyses === 0
                ? 'No analyses run this session'
                : `${successfulAnalyses} successful • ${failedAnalyses} failed`}
            </p>
          </CardContent>
        </Card>

        {/* Card 2: Session Activity & Recency */}
        <Card hoverEffect>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
              Session Activity
            </CardTitle>
            <Clock className="w-4 h-4 text-clinical-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-slate-900 truncate">
              {formattedLastAnalysis || 'No Activity'}
            </div>
            <p className="text-xs text-slate-500 mt-1">
              {totalAnalyses === 0
                ? 'Awaiting first scan input'
                : `Success rate: ${successRate}%`}
            </p>
          </CardContent>
        </Card>

        {/* Card 3: Model Architecture & Benchmark */}
        <Card hoverEffect>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
              Model Architecture
            </CardTitle>
            <ShieldCheck className="w-4 h-4 text-clinical-600" />
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-slate-900">DenseNet121</div>
            <p className="text-xs text-clinical-700 mt-1 font-medium">
              Val Accuracy: 97.68% (949 samples)
            </p>
          </CardContent>
        </Card>

        {/* Card 4: Live System Status (API & Model Readiness) */}
        <Card hoverEffect>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
              System Status
            </CardTitle>
            {backendStatus === 'healthy' && modelStatus === 'ready' ? (
              <ShieldCheck className="w-4 h-4 text-emerald-600" />
            ) : backendStatus === 'checking' || modelStatus === 'checking' ? (
              <RefreshCw className="w-4 h-4 text-slate-500 animate-spin motion-reduce:animate-none" />
            ) : backendStatus === 'healthy' && modelStatus === 'not_loaded' ? (
              <AlertTriangle className="w-4 h-4 text-amber-500" />
            ) : (
              <AlertCircle className="w-4 h-4 text-rose-500" />
            )}
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-slate-900 truncate">
              {backendStatus === 'checking' && modelStatus === 'checking'
                ? 'Checking...'
                : backendStatus === 'healthy' && modelStatus === 'ready'
                ? 'Connected & Ready'
                : backendStatus === 'healthy' && modelStatus === 'not_loaded'
                ? 'Model Not Loaded'
                : backendStatus === 'healthy' && modelStatus === 'unavailable'
                ? 'Model Unavailable'
                : 'API Unavailable'}
            </div>

            {/* Separate status indicators */}
            <div className="flex items-center gap-2 mt-1.5 text-xs text-slate-600 font-medium">
              <span className="flex items-center gap-1">
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    backendStatus === 'healthy'
                      ? 'bg-emerald-500'
                      : backendStatus === 'checking'
                      ? 'bg-amber-400 animate-pulse motion-reduce:animate-none'
                      : 'bg-rose-500'
                  }`}
                />
                API: {backendStatus === 'healthy' ? 'Healthy' : backendStatus === 'checking' ? 'Checking' : 'Down'}
              </span>
              <span className="text-slate-300">•</span>
              <span className="flex items-center gap-1">
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    modelStatus === 'ready'
                      ? 'bg-clinical-600'
                      : modelStatus === 'checking'
                      ? 'bg-amber-400 animate-pulse motion-reduce:animate-none'
                      : modelStatus === 'not_loaded'
                      ? 'bg-amber-500'
                      : 'bg-rose-500'
                  }`}
                />
                Model:{' '}
                {modelStatus === 'ready'
                  ? 'Ready'
                  : modelStatus === 'checking'
                  ? 'Checking'
                  : modelStatus === 'not_loaded'
                  ? 'Not Loaded'
                  : 'Unavailable'}
              </span>
            </div>

            {/* Polling time & manual refresh button */}
            <div className="flex items-center justify-between text-[11px] text-slate-500 mt-2.5 pt-2 border-t border-slate-100">
              <span title={lastChecked ? lastChecked.toISOString() : undefined}>
                {formattedLastChecked ? `Checked: ${formattedLastChecked}` : 'Checking...'}
              </span>
              <button
                type="button"
                onClick={() => refetchHealth()}
                disabled={isHealthLoading}
                className="text-clinical-600 hover:text-clinical-700 font-medium flex items-center gap-1 transition-colors disabled:opacity-50 cursor-pointer"
                aria-label="Refresh status now"
                title="Refresh connection status"
              >
                <RefreshCw className={`w-3 h-3 ${isHealthLoading ? 'animate-spin motion-reduce:animate-none' : ''}`} />
                <span>Refresh</span>
              </button>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Middle Section: Recent Activity & Quick Navigation */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recent Session Analyses Feed (2 cols on lg) */}
        <Card className="lg:col-span-2">
          <CardHeader className="flex flex-row items-center justify-between pb-3">
            <div>
              <CardTitle className="text-base font-semibold text-slate-900">
                Recent Session Analyses
              </CardTitle>
              <CardDescription>
                Analyses conducted during the current browser session. Not persisted to permanent storage.
              </CardDescription>
            </div>
            <Badge variant="outline" size="sm" className="font-mono text-xs">
              {recentActivity.length} Events
            </Badge>
          </CardHeader>
          <CardContent>
            {recentActivity.length === 0 ? (
              /* Genuine Empty State */
              <div className="h-56 rounded-xl bg-slate-50/70 border border-slate-200/80 flex flex-col items-center justify-center text-center p-6 text-slate-500 space-y-2">
                <div className="w-12 h-12 rounded-xl bg-slate-100 text-slate-500 flex items-center justify-center mb-1">
                  <History className="w-6 h-6" />
                </div>
                <h5 className="text-sm font-semibold text-slate-800">
                  No Analyses Recorded This Session
                </h5>
                <p className="text-xs text-slate-500 max-w-sm leading-relaxed">
                  Upload and analyze a chest radiograph to evaluate DenseNet121 classification categories and Grad-CAM visual heatmaps.
                </p>
                <Button
                  variant="outline"
                  size="sm"
                  className="mt-2 text-xs bg-white"
                  onClick={() => navigate('/analysis')}
                  leftIcon={<Scan className="w-3.5 h-3.5" />}
                >
                  Start New Analysis
                </Button>
              </div>
            ) : (
              /* Genuine Recent Activity List */
              <div className="space-y-2.5">
                {recentActivity.slice(0, 5).map((entry) => (
                  <div
                    key={entry.id}
                    className="p-3 rounded-lg border border-slate-100 bg-white hover:bg-slate-50/60 transition-colors flex items-center justify-between gap-3 text-xs"
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      {entry.status === 'success' ? (
                        <div className="w-7 h-7 rounded-lg bg-emerald-50 text-emerald-600 flex items-center justify-center shrink-0">
                          <CheckCircle2 className="w-4 h-4" />
                        </div>
                      ) : (
                        <div className="w-7 h-7 rounded-lg bg-rose-50 text-rose-600 flex items-center justify-center shrink-0">
                          <XCircle className="w-4 h-4" />
                        </div>
                      )}

                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="font-semibold text-slate-900 truncate">
                            {entry.status === 'success'
                              ? entry.predictedLabel
                              : 'Analysis Failed'}
                          </span>
                          {entry.status === 'success' && entry.confidence !== undefined && (
                            <span className="text-[11px] font-mono font-medium text-slate-500">
                              ({(entry.confidence * 100).toFixed(1)}%)
                            </span>
                          )}
                        </div>
                        <div className="text-[11px] text-slate-500 truncate">
                          File: {entry.fileName}
                          {entry.durationMs && ` • ${entry.durationMs.toFixed(0)} ms`}
                        </div>
                      </div>
                    </div>

                    <div className="text-right shrink-0">
                      <div className="font-mono text-[11px] text-slate-500">
                        {entry.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </div>
                      <Badge
                        variant={entry.status === 'success' ? 'success' : 'danger'}
                        size="sm"
                        className="text-[10px] mt-0.5"
                      >
                        {entry.status === 'success' ? 'Success' : 'Error'}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Workflow Actions Column (1 col on lg) */}
        <Card>
          <CardHeader>
            <CardTitle>Workflow Actions</CardTitle>
            <CardDescription>Direct navigation to primary modules.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2.5">
            <button
              type="button"
              onClick={() => navigate('/analysis')}
              className="w-full flex items-center justify-between p-3 rounded-lg border border-slate-200/80 bg-white hover:bg-slate-50 text-left transition-colors group cursor-pointer"
            >
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-clinical-50 text-clinical-600 flex items-center justify-center">
                  <Scan className="w-4 h-4" />
                </div>
                <div>
                  <div className="text-xs font-semibold text-slate-800">X-ray Analysis</div>
                  <div className="text-[11px] text-slate-500">Upload and inspect heatmaps</div>
                </div>
              </div>
              <ArrowRight className="w-4 h-4 text-slate-500 group-hover:text-clinical-600 transition-colors" />
            </button>

            <button
              type="button"
              onClick={() => navigate('/model-info')}
              className="w-full flex items-center justify-between p-3 rounded-lg border border-slate-200/80 bg-white hover:bg-slate-50 text-left transition-colors group cursor-pointer"
            >
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-slate-100 text-slate-600 flex items-center justify-center">
                  <ShieldCheck className="w-4 h-4" />
                </div>
                <div>
                  <div className="text-xs font-semibold text-slate-800">Model Specifications</div>
                  <div className="text-[11px] text-slate-500">Architecture & provenance metrics</div>
                </div>
              </div>
              <ArrowRight className="w-4 h-4 text-slate-500 group-hover:text-slate-600 transition-colors" />
            </button>

            <button
              type="button"
              onClick={() => navigate('/reports')}
              className="w-full flex items-center justify-between p-3 rounded-lg border border-slate-200/80 bg-white hover:bg-slate-50 text-left transition-colors group cursor-pointer"
            >
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-slate-100 text-slate-600 flex items-center justify-center">
                  <FileText className="w-4 h-4" />
                </div>
                <div>
                  <div className="text-xs font-semibold text-slate-800">Clinical Reports</div>
                  <div className="text-[11px] text-slate-500">Synthesize documentation</div>
                </div>
              </div>
              <ArrowRight className="w-4 h-4 text-slate-500 group-hover:text-slate-600 transition-colors" />
            </button>
          </CardContent>
        </Card>
      </div>

      {/* Documented Classification Spectrum */}
      <Card>
        <CardHeader>
          <CardTitle>Documented Classification Categories</CardTitle>
          <CardDescription>
            Trained radiological categories from the research dataset (jtiptj/chest-xray-pneumoniacovid19tuberculosis).
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            <div className="p-3.5 rounded-lg border border-slate-100 bg-slate-50/70">
              <div className="flex items-center justify-between mb-1">
                <span className="font-semibold text-slate-800 text-sm">COVID-19</span>
                <Badge variant="clinical" size="sm">Index 0</Badge>
              </div>
              <p className="text-xs text-slate-500 leading-relaxed">
                Viral respiratory infection demonstrating bilateral ground-glass opacities and peripheral consolidations.
              </p>
            </div>

            <div className="p-3.5 rounded-lg border border-slate-100 bg-slate-50/70">
              <div className="flex items-center justify-between mb-1">
                <span className="font-semibold text-slate-800 text-sm">Normal</span>
                <Badge variant="success" size="sm">Index 1</Badge>
              </div>
              <p className="text-xs text-slate-500 leading-relaxed">
                Clear pulmonary lung fields without focal consolidation, pneumothorax, or abnormal pleural thickening.
              </p>
            </div>

            <div className="p-3.5 rounded-lg border border-slate-100 bg-slate-50/70">
              <div className="flex items-center justify-between mb-1">
                <span className="font-semibold text-slate-800 text-sm">Pneumonia</span>
                <Badge variant="warning" size="sm">Index 2</Badge>
              </div>
              <p className="text-xs text-slate-500 leading-relaxed">
                Infectious acute inflammatory condition manifesting pulmonary infiltrates, air bronchograms, or lobar consolidation.
              </p>
            </div>

            <div className="p-3.5 rounded-lg border border-slate-100 bg-slate-50/70">
              <div className="flex items-center justify-between mb-1">
                <span className="font-semibold text-slate-800 text-sm">Tuberculosis</span>
                <Badge variant="danger" size="sm">Index 3</Badge>
              </div>
              <p className="text-xs text-slate-500 leading-relaxed">
                Mycobacterial disease presenting apical consolidations, cavitary lesions, or adenopathy. Corrected spelling.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default DashboardPage;
