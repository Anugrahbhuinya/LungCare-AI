import { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  History,
  Trash2,
  RefreshCw,
  Eye,
  Shield,
  AlertCircle,
  Database,
  Calendar,
  Layers,
  ArrowRight,
  HardDrive,
  Download,
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
import { useFocusTrap } from '@/hooks/useFocusTrap';
import { HistoryDetailModal } from '@/components/history/HistoryDetailModal';
import { downloadAnalysisReportPDF } from '@/services/reports';
import type { HistoryAnalysisRecord } from '@/services/history';
import { getDisplayClassName } from '@/types/prediction';

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

export function HistoryPage() {
  const navigate = useNavigate();
  const { records, isLoading, error, isAvailable, refresh, deleteRecord, clearAll } =
    useHistory();

  const [selectedRecord, setSelectedRecord] = useState<HistoryAnalysisRecord | null>(null);
  const [isClearModalOpen, setIsClearModalOpen] = useState<boolean>(false);
  const [isDeletingAll, setIsDeletingAll] = useState<boolean>(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [downloadingId, setDownloadingId] = useState<string | null>(null);
  const [pdfError, setPdfError] = useState<string | null>(null);

  const clearButtonRef = useRef<HTMLButtonElement | null>(null);
  const cancelButtonRef = useRef<HTMLButtonElement | null>(null);

  const clearDialogRef = useFocusTrap<HTMLDivElement>({
    isActive: isClearModalOpen,
    onClose: () => {
      if (!isDeletingAll) setIsClearModalOpen(false);
    },
    triggerRef: clearButtonRef,
    initialFocusRef: cancelButtonRef,
    setInert: true,
  });

  const handleDeleteRecord = async (id: string) => {
    setDeletingId(id);
    await deleteRecord(id);
    setDeletingId(null);
  };

  const handleDownloadPDF = async (record: HistoryAnalysisRecord) => {
    setDownloadingId(record.id);
    setPdfError(null);
    try {
      await downloadAnalysisReportPDF(record);
    } catch (err: unknown) {
      console.error('[PDF] Failed to download PDF from HistoryPage:', err);
      setPdfError(err instanceof Error ? err.message : 'Failed to generate PDF report.');
    } finally {
      setDownloadingId(null);
    }
  };

  const handleConfirmClearAll = async () => {
    setIsDeletingAll(true);
    await clearAll();
    setIsDeletingAll(false);
    setIsClearModalOpen(false);
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900">
            Scan History
          </h2>
          <p className="text-slate-500 text-sm mt-0.5">
            Persistent browser-local analysis records stored in IndexedDB.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="clinical" size="md" className="flex items-center gap-1.5 font-mono text-xs">
            <Database className="w-3.5 h-3.5" />
            IndexedDB: lungcare-history
          </Badge>
          <Badge variant="outline" size="md">
            {records.length} {records.length === 1 ? 'Record' : 'Records'}
          </Badge>
        </div>
      </div>

      {/* Mandatory Privacy & Browser-Local Storage Disclosure */}
      <Card className="border-slate-200/90 bg-slate-50/60">
        <CardContent className="p-4 sm:p-5">
          <div className="flex items-start gap-3">
            <div className="w-9 h-9 rounded-xl bg-clinical-50 text-clinical-700 flex items-center justify-center shrink-0 mt-0.5">
              <Shield className="w-5 h-5" />
            </div>
            <div className="space-y-1.5 flex-1">
              <h4 className="text-sm font-semibold text-slate-800">
                Browser-Local Storage & Privacy Disclosure
              </h4>
              <p className="text-xs text-slate-600 leading-relaxed">
                Analysis records displayed on this page are stored strictly in this browser’s client-side IndexedDB (<code className="font-mono bg-white px-1 py-0.5 rounded border border-slate-200 text-slate-700">lungcare-history</code>).
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1 text-xs text-slate-500">
                <div className="flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-slate-400 shrink-0" />
                  <span><strong>No Cloud Synchronization:</strong> Records are never sent to external servers or cloud EHRs.</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-slate-400 shrink-0" />
                  <span><strong>Storage Deletion & Eviction:</strong> Clearing browser cache or cookies alone does not necessarily delete IndexedDB. Users must clear the website's stored site data or delete the database in browser settings; automatic browser storage eviction under disk pressure may also remove records.</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-slate-400 shrink-0" />
                  <span><strong>Profile Shared:</strong> Anyone with access to this browser profile can view history.</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-slate-400 shrink-0" />
                  <span><strong>Research Prototype:</strong> Not certified for HIPAA or GDPR clinical record retention.</span>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Main History Table / Action Card */}
      <Card>
        <CardHeader className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-4">
          <div>
            <CardTitle>Recorded Analysis Sessions</CardTitle>
            <CardDescription>
              Chronological log of successful DenseNet121 inferences (newest first).
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => refresh()}
              disabled={isLoading}
              leftIcon={<RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin motion-reduce:animate-none' : ''}`} />}
            >
              Refresh
            </Button>
            <Button
              ref={clearButtonRef}
              variant="danger"
              size="sm"
              onClick={() => setIsClearModalOpen(true)}
              disabled={isLoading || records.length === 0}
              leftIcon={<Trash2 className="w-3.5 h-3.5" />}
            >
              Clear All
            </Button>
          </div>
        </CardHeader>
        <CardContent className="pt-6 space-y-4">
          {/* PDF Generation Error Banner */}
          {pdfError && (
            <div className="p-3.5 rounded-lg border border-rose-200 bg-rose-50 text-rose-800 text-xs flex items-start gap-2">
              <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
              <div className="flex-1 leading-relaxed">
                <span className="font-semibold">PDF Export Error: </span>
                {pdfError}
              </div>
              <button
                onClick={() => setPdfError(null)}
                className="text-xs font-semibold text-rose-700 hover:text-rose-900"
              >
                Dismiss
              </button>
            </div>
          )}

          {/* State 1: IndexedDB Storage Unavailable */}
          {!isAvailable && (
            <div className="p-6 rounded-xl border border-amber-200 bg-amber-50/70 text-amber-900 text-center space-y-2">
              <HardDrive className="w-10 h-10 text-amber-600 mx-auto" />
              <h4 className="text-sm font-semibold">IndexedDB Storage Unavailable</h4>
              <p className="text-xs text-amber-800/90 max-w-md mx-auto leading-relaxed">
                Your browser environment has disabled or blocked IndexedDB access (often occurring in strict private windows or restricted enterprise containers). Scan history cannot be saved locally.
              </p>
            </div>
          )}

          {/* State 2: General Storage Error */}
          {isAvailable && error && (
            <div className="p-6 rounded-xl border border-rose-200 bg-rose-50/70 text-rose-900 text-center space-y-2">
              <AlertCircle className="w-10 h-10 text-rose-600 mx-auto" />
              <h4 className="text-sm font-semibold">Unable to Load Scan History</h4>
              <p className="text-xs text-rose-800/90 max-w-md mx-auto leading-relaxed">
                {error}
              </p>
              <Button
                variant="outline"
                size="sm"
                onClick={() => refresh()}
                className="mt-2 bg-white"
              >
                Retry Query
              </Button>
            </div>
          )}

          {/* State 3: Active Loading State */}
          {isAvailable && !error && isLoading && (
            <div className="h-64 flex flex-col items-center justify-center text-center p-6 space-y-3 text-slate-500">
              <Spinner size="lg" className="text-clinical-600" />
              <p className="text-xs font-medium">Accessing browser IndexedDB records...</p>
            </div>
          )}

          {/* State 4: Empty State */}
          {isAvailable && !error && !isLoading && records.length === 0 && (
            <div className="h-64 rounded-xl border border-dashed border-slate-200 flex flex-col items-center justify-center text-center p-6 text-slate-500 bg-slate-50/50 space-y-2">
              <div className="w-12 h-12 rounded-xl bg-slate-100 flex items-center justify-center text-slate-500 mb-1">
                <History className="w-6 h-6" />
              </div>
              <h4 className="text-sm font-semibold text-slate-700">
                No Analyses Recorded Yet
              </h4>
              <p className="text-xs text-slate-500 max-w-md leading-relaxed">
                Successful chest X-ray analyses executed on the Analysis page are automatically stored in this browser’s IndexedDB for later review.
              </p>
              <Button
                variant="primary"
                size="sm"
                className="mt-2"
                onClick={() => navigate('/analysis')}
                leftIcon={<ArrowRight className="w-3.5 h-3.5" />}
              >
                Start New Analysis
              </Button>
            </div>
          )}

          {/* State 5: Records Present */}
          {isAvailable && !error && !isLoading && records.length > 0 && (
            <div className="space-y-3">
              <div className="overflow-x-auto rounded-xl border border-slate-200">
                <table
                  className="w-full text-xs text-left"
                  aria-label="Scan analysis history records"
                >
                  <thead className="bg-slate-50/80 text-slate-600 border-b border-slate-200 font-semibold">
                    <tr>
                      <th scope="col" className="py-3 px-3.5">Analysis Date & Time</th>
                      <th scope="col" className="py-3 px-3.5">Radiograph File</th>
                      <th scope="col" className="py-3 px-3.5">Predicted Class</th>
                      <th scope="col" className="py-3 px-3.5 text-right">Confidence</th>
                      <th scope="col" className="py-3 px-3.5">Grad-CAM Target</th>
                      <th scope="col" className="py-3 px-3.5 text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {records.map((record) => {
                      const dateStr = !isNaN(new Date(record.timestamp).getTime())
                        ? new Date(record.timestamp).toLocaleString(undefined, {
                            dateStyle: 'short',
                            timeStyle: 'short',
                          })
                        : 'N/A';
                      const confPercent = (typeof record.confidence === 'number' ? record.confidence * 100 : 0).toFixed(2);
                      const isItemDeleting = deletingId === record.id;

                      return (
                        <tr
                          key={record.id}
                          className="hover:bg-slate-50/70 transition-colors group"
                        >
                          <th
                            scope="row"
                            className="py-3 px-3.5 text-slate-700 whitespace-nowrap font-normal text-left"
                          >
                            <span className="flex items-center gap-1.5 font-medium">
                              <Calendar className="w-3.5 h-3.5 text-slate-500 shrink-0" />
                              {dateStr}
                            </span>
                          </th>
                          <td className="py-3 px-3.5 font-medium text-slate-800 max-w-[180px] truncate" title={record.fileName}>
                            {record.fileName}
                          </td>
                          <td className="py-3 px-3.5 whitespace-nowrap">
                            <Badge
                              variant={getClassBadgeVariant(record.predicted_class)}
                              size="sm"
                            >
                              {record.predicted_label || getDisplayClassName(record.predicted_class)}
                            </Badge>
                          </td>
                          <td className="py-3 px-3.5 text-right font-mono font-bold text-slate-900 whitespace-nowrap">
                            {confPercent}%
                          </td>
                          <td className="py-3 px-3.5 whitespace-nowrap">
                            <span className="text-[11px] font-mono text-slate-600 flex items-center gap-1">
                              <Layers className="w-3 h-3 text-slate-500" />
                              {record.selected_target_label || record.selected_target_class || record.predicted_label}
                            </span>
                          </td>
                          <td className="py-3 px-3.5 text-right whitespace-nowrap">
                            <div className="flex items-center justify-end gap-1.5">
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => setSelectedRecord(record)}
                                className="h-7 px-2 text-xs text-slate-700 hover:text-clinical-700"
                                leftIcon={<Eye className="w-3.5 h-3.5" />}
                              >
                                View
                              </Button>
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => handleDownloadPDF(record)}
                                disabled={downloadingId === record.id}
                                className="h-7 px-2 text-xs text-slate-700 hover:text-clinical-700 hover:border-clinical-300"
                                leftIcon={
                                  downloadingId === record.id ? (
                                    <Spinner size="sm" className="text-clinical-600" />
                                  ) : (
                                    <Download className="w-3.5 h-3.5" />
                                  )
                                }
                                title="Download PDF report"
                              >
                                PDF
                              </Button>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => handleDeleteRecord(record.id)}
                                disabled={isItemDeleting}
                                className="h-7 w-7 p-0 text-slate-500 hover:text-rose-600 hover:bg-rose-50"
                                title="Delete record"
                                aria-label={`Delete record ${record.id}`}
                              >
                                {isItemDeleting ? (
                                  <Spinner size="sm" className="text-rose-600" />
                                ) : (
                                  <Trash2 className="w-3.5 h-3.5" />
                                )}
                              </Button>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Record Detail Modal */}
      <HistoryDetailModal
        record={selectedRecord}
        isOpen={selectedRecord !== null}
        onClose={() => setSelectedRecord(null)}
      />

      {/* Confirmation Modal for Clearing All History */}
      {isClearModalOpen && (
        <div
          ref={clearDialogRef}
          role="dialog"
          aria-modal="true"
          aria-labelledby="clear-history-title"
          tabIndex={-1}
          onClick={(e) => {
            if (e.target === e.currentTarget && !isDeletingAll) {
              setIsClearModalOpen(false);
            }
          }}
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm"
        >
          <div className="bg-white rounded-2xl max-w-md w-full p-6 shadow-2xl border border-slate-200 space-y-4">
            <div className="w-11 h-11 rounded-xl bg-rose-100 text-rose-600 flex items-center justify-center mb-1">
              <Trash2 className="w-6 h-6" />
            </div>
            <div>
              <h3 id="clear-history-title" className="text-base font-bold text-slate-900">
                Clear All Scan History?
              </h3>
              <p className="text-xs text-slate-500 mt-1.5 leading-relaxed">
                This will permanently delete all <strong>{records.length}</strong> locally stored analysis records from this browser’s IndexedDB database (<code className="font-mono text-slate-700 bg-slate-100 px-1 py-0.5 rounded">lungcare-history</code>). This action cannot be undone.
              </p>
            </div>
            <div className="flex items-center justify-end gap-2.5 pt-2 border-t border-slate-100">
              <Button
                ref={cancelButtonRef}
                variant="outline"
                size="sm"
                onClick={() => setIsClearModalOpen(false)}
                disabled={isDeletingAll}
              >
                Cancel
              </Button>
              <Button
                variant="danger"
                size="sm"
                onClick={handleConfirmClearAll}
                isLoading={isDeletingAll}
                leftIcon={<Trash2 className="w-3.5 h-3.5" />}
              >
                Clear All Records
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default HistoryPage;
