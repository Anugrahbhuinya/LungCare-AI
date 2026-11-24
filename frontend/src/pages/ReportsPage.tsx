import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  FileText,
  Download,
  Calendar,
  HardDrive,
  AlertCircle,
  ArrowRight,
} from 'lucide-react';
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Spinner } from '@/components/ui/Spinner';
import { useHistory } from '@/hooks/useHistory';
import {
  downloadAnalysisReportPDF,
  getReportFileName,
} from '@/services/reports';
import type { HistoryAnalysisRecord } from '@/services/history';
import { MODEL_CLASSES, getDisplayClassName } from '@/types/prediction';

function getClassBadgeVariant(className?: string): 'clinical' | 'success' | 'warning' | 'danger' {
  switch ((className || '').toUpperCase()) {
    case 'COVID19':
      return 'clinical';
    case 'NORMAL':
      return 'success';
    case 'PNEUMONIA':
      return 'warning';
    case 'TURBERCULOSIS':
      return 'danger';
    default:
      return 'clinical';
  }
}

export function ReportsPage() {
  const navigate = useNavigate();
  const { records, isLoading, error, isAvailable } = useHistory();

  const [selectedRecordId, setSelectedRecordId] = useState<string | null>(null);
  const [downloadingId, setDownloadingId] = useState<string | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  // Automatically select the first record when records load if none is selected
  const activeRecord: HistoryAnalysisRecord | null =
    records.find((r) => r.id === (selectedRecordId || records[0]?.id)) || null;

  const handleDownload = async (record: HistoryAnalysisRecord) => {
    if (!record) return;
    setDownloadingId(record.id);
    setDownloadError(null);
    try {
      await downloadAnalysisReportPDF(record);
    } catch (err: unknown) {
      console.error('[PDF] Download error:', err);
      setDownloadError(
        err instanceof Error ? err.message : 'Failed to generate PDF analysis report.'
      );
    } finally {
      setDownloadingId(null);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900">
            Analysis Reports
          </h2>
          <p className="text-slate-500 text-sm mt-0.5">
            Generate and export client-side PDF research summaries from saved scan records.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="clinical" size="md" className="font-mono text-xs">
            PDF Engine: jsPDF
          </Badge>
          <Badge variant="outline" size="md">
            {records.length} {records.length === 1 ? 'Record Available' : 'Records Available'}
          </Badge>
        </div>
      </div>

      {/* Storage Unavailable Alert */}
      {!isAvailable && (
        <div className="p-6 rounded-xl border border-amber-200 bg-amber-50/70 text-amber-900 text-center space-y-2">
          <HardDrive className="w-10 h-10 text-amber-600 mx-auto" />
          <h4 className="text-sm font-semibold">IndexedDB Storage Unavailable</h4>
          <p className="text-xs text-amber-800/90 max-w-md mx-auto leading-relaxed">
            Your browser environment does not support or has restricted IndexedDB storage. Historical reports require local storage to retrieve analysis records.
          </p>
        </div>
      )}

      {/* Query Error Alert */}
      {isAvailable && error && (
        <div className="p-4 rounded-xl border border-rose-200 bg-rose-50 text-rose-900 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
          <div>
            <h4 className="text-sm font-semibold">Error Loading Saved Records</h4>
            <p className="text-xs text-rose-700 mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* PDF Generation Error Banner */}
      {downloadError && (
        <div className="p-4 rounded-xl border border-rose-200 bg-rose-50 text-rose-900 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
          <div className="flex-1">
            <h4 className="text-sm font-semibold">PDF Generation Failed</h4>
            <p className="text-xs text-rose-700 mt-0.5">{downloadError}</p>
          </div>
          <button
            onClick={() => setDownloadError(null)}
            className="text-xs font-semibold text-rose-700 hover:text-rose-900"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Loading State */}
      {isAvailable && !error && isLoading && (
        <div className="h-64 flex flex-col items-center justify-center text-center p-6 space-y-3 text-slate-500">
          <Spinner size="lg" className="text-clinical-600" />
          <p className="text-xs font-medium">Loading saved analysis records...</p>
        </div>
      )}

      {/* Empty State */}
      {isAvailable && !error && !isLoading && records.length === 0 && (
        <Card>
          <CardContent className="pt-6">
            <div className="h-64 rounded-xl border border-dashed border-slate-200 flex flex-col items-center justify-center text-center p-6 text-slate-500 bg-slate-50/50 space-y-2">
              <div className="w-12 h-12 rounded-xl bg-slate-100 flex items-center justify-center text-slate-500 mb-1">
                <FileText className="w-6 h-6" />
              </div>
              <h4 className="text-sm font-semibold text-slate-700">
                No Analyses Stored for Report Generation
              </h4>
              <p className="text-xs text-slate-500 max-w-md leading-relaxed">
                PDF reports are generated from completed analyses stored in your browser’s IndexedDB. Run an analysis on the Analysis page to generate and export reports.
              </p>
              <Button
                variant="primary"
                size="sm"
                className="mt-2"
                onClick={() => navigate('/analysis')}
                leftIcon={<ArrowRight className="w-3.5 h-3.5" />}
              >
                Run Analysis
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Active Records Workflow Grid */}
      {isAvailable && !error && !isLoading && records.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left Column: List of Saved Analyses (7 cols on lg) */}
          <div className="lg:col-span-7 space-y-4">
            <Card>
              <CardHeader className="pb-3 border-b border-slate-100">
                <CardTitle>Available Analysis Records</CardTitle>
                <CardDescription>
                  Select a record to preview report metadata or download its PDF directly.
                </CardDescription>
              </CardHeader>
              <CardContent className="pt-3 divide-y divide-slate-100">
                {records.map((record) => {
                  const isSelected = activeRecord?.id === record.id;
                  const isRecordDownloading = downloadingId === record.id;
                  const dateStr = new Date(record.timestamp).toLocaleString(undefined, {
                    dateStyle: 'short',
                    timeStyle: 'short',
                  });
                  const confPct = (record.confidence * 100).toFixed(2);

                  return (
                    <div
                      key={record.id}
                      className={`p-3.5 rounded-xl transition-all flex flex-col sm:flex-row sm:items-center justify-between gap-3 ${
                        isSelected
                          ? 'bg-clinical-50/60 border border-clinical-200'
                          : 'hover:bg-slate-50/70 border border-transparent'
                      }`}
                    >
                      <div
                        className="space-y-1 cursor-pointer flex-1"
                        onClick={() => setSelectedRecordId(record.id)}
                      >
                        <div className="flex items-center gap-2 flex-wrap">
                          <Badge
                            variant={getClassBadgeVariant(record.predicted_class)}
                            size="sm"
                          >
                            {record.predicted_label || getDisplayClassName(record.predicted_class)}
                          </Badge>
                          <span className="text-xs font-mono font-bold text-slate-800">
                            {confPct}%
                          </span>
                          <span className="text-[11px] text-slate-500 flex items-center gap-1">
                            <Calendar className="w-3 h-3 text-slate-500" />
                            {dateStr}
                          </span>
                        </div>
                        <p className="text-xs font-semibold text-slate-800 truncate max-w-sm" title={record.fileName}>
                          {record.fileName}
                        </p>
                        <p className="text-[10px] text-slate-500 font-mono truncate">
                          ID: {record.id}
                        </p>
                      </div>

                      <div className="flex items-center gap-2 shrink-0">
                        <Button
                          variant={isSelected ? 'primary' : 'outline'}
                          size="sm"
                          onClick={() => setSelectedRecordId(record.id)}
                          className="text-xs h-8"
                        >
                          {isSelected ? 'Selected' : 'Preview'}
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleDownload(record)}
                          disabled={isRecordDownloading}
                          isLoading={isRecordDownloading}
                          className="text-xs h-8 hover:text-clinical-700 hover:border-clinical-300"
                          leftIcon={<Download className="w-3.5 h-3.5" />}
                          title="Download PDF report for this record"
                        >
                          PDF
                        </Button>
                      </div>
                    </div>
                  );
                })}
              </CardContent>
            </Card>
          </div>

          {/* Right Column: Selected Record Summary & Download Panel (5 cols on lg) */}
          <div className="lg:col-span-5 space-y-4">
            {activeRecord ? (
              <Card className="sticky top-6">
                <CardHeader className="pb-3 border-b border-slate-100">
                  <div className="flex items-center justify-between">
                    <CardTitle>Report Summary Preview</CardTitle>
                    <Badge variant="clinical" size="sm">
                      Ready to Export
                    </Badge>
                  </div>
                  <CardDescription className="font-mono text-[11px] truncate" title={getReportFileName(activeRecord.id)}>
                    {getReportFileName(activeRecord.id)}
                  </CardDescription>
                </CardHeader>

                <CardContent className="pt-4 space-y-4 text-xs">
                  {/* Primary Classification Card */}
                  <div className="p-3.5 rounded-xl border border-slate-200 bg-slate-50 flex items-center justify-between">
                    <div>
                      <span className="text-[10px] uppercase font-semibold text-slate-500">
                        Primary Prediction
                      </span>
                      <div className="text-base font-bold text-slate-900 mt-0.5">
                        {activeRecord.predicted_label || getDisplayClassName(activeRecord.predicted_class)}
                      </div>
                    </div>
                    <div className="text-right">
                      <span className="text-[10px] uppercase font-semibold text-slate-500">
                        Confidence
                      </span>
                      <div className="text-lg font-extrabold text-slate-900 font-mono mt-0.5">
                        {(activeRecord.confidence * 100).toFixed(2)}%
                      </div>
                    </div>
                  </div>

                  {/* Softmax Probability Spectrum */}
                  <div className="space-y-2">
                    <span className="text-[11px] font-semibold text-slate-700 uppercase tracking-wider block">
                      Softmax Probabilities
                    </span>
                    <div className="space-y-1.5 p-3 rounded-lg border border-slate-100 bg-white">
                      {MODEL_CLASSES.map((rawClass) => {
                        const prob = activeRecord.probabilities?.[rawClass] ?? 0;
                        const pct = (prob * 100).toFixed(2);
                        const isWinner = activeRecord.predicted_class.toUpperCase() === rawClass.toUpperCase();
                        const label = getDisplayClassName(rawClass);

                        return (
                          <div key={rawClass} className="flex items-center justify-between text-[11px]">
                            <span className={isWinner ? 'font-bold text-slate-900' : 'text-slate-600'}>
                              {label}
                            </span>
                            <span className="font-mono font-semibold text-slate-800">{pct}%</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Execution Parameters */}
                  <div className="grid grid-cols-2 gap-2 text-[11px]">
                    <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-100">
                      <span className="text-slate-500 block text-[10px]">Target Layer</span>
                      <span className="font-mono font-semibold text-slate-800 truncate block">
                        {activeRecord.target_layer || 'conv5_block16_2'}
                      </span>
                    </div>
                    <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-100">
                      <span className="text-slate-500 block text-[10px]">Execution Time</span>
                      <span className="font-semibold text-slate-800 block">
                        {activeRecord.total_time_ms ? `${activeRecord.total_time_ms.toFixed(1)} ms` : 'N/A'}
                      </span>
                    </div>
                  </div>

                  {/* Image Retention Notice */}
                  <div className="p-3 rounded-lg border border-amber-200 bg-amber-50/70 text-amber-900 text-[11px] leading-relaxed">
                    <span className="font-semibold">Notice: </span>
                    Original radiograph and Grad-CAM color overlays are not stored in local history and therefore are omitted from this report.
                  </div>

                  {/* Download Action */}
                  <Button
                    variant="primary"
                    size="lg"
                    className="w-full"
                    onClick={() => handleDownload(activeRecord)}
                    disabled={downloadingId === activeRecord.id}
                    isLoading={downloadingId === activeRecord.id}
                    leftIcon={<Download className="w-4 h-4" />}
                  >
                    Download PDF Analysis Report
                  </Button>
                </CardContent>
              </Card>
            ) : null}
          </div>
        </div>
      )}
    </div>
  );
}

export default ReportsPage;
