import { useState, useEffect, useRef } from "react";
import {
  X,
  Clock,
  FileText,
  Cpu,
  Layers,
  ShieldAlert,
  Hash,
  Database,
  Download,
  AlertCircle,
} from "lucide-react";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { useFocusTrap } from "@/hooks/useFocusTrap";
import type { HistoryAnalysisRecord } from "@/services/history";
import { MODEL_CLASSES, getDisplayClassName } from "@/types/prediction";
import { downloadAnalysisReportPDF } from "@/services/reports";

export interface HistoryDetailModalProps {
  record: HistoryAnalysisRecord | null;
  isOpen: boolean;
  onClose: () => void;
}

function getClassBadgeVariant(
  className?: string,
): "clinical" | "success" | "warning" | "danger" {
  switch ((className || "").toUpperCase()) {
    case "COVID19":
      return "clinical";
    case "NORMAL":
      return "success";
    case "PNEUMONIA":
      return "warning";
    case "TURBERCULOSIS":
      return "danger";
    default:
      return "clinical";
  }
}

function getClassBarColor(className: string): string {
  switch (className.toUpperCase()) {
    case "COVID19":
      return "bg-clinical-600";
    case "NORMAL":
      return "bg-emerald-600";
    case "PNEUMONIA":
      return "bg-amber-600";
    case "TURBERCULOSIS":
      return "bg-rose-600";
    default:
      return "bg-slate-600";
  }
}

export function HistoryDetailModal({
  record,
  isOpen,
  onClose,
}: HistoryDetailModalProps) {
  const [isDownloading, setIsDownloading] = useState<boolean>(false);
  const [pdfError, setPdfError] = useState<string | null>(null);
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);

  // Focus trap and Escape key containment
  const dialogRef = useFocusTrap<HTMLDivElement>({
    isActive: isOpen && record !== null,
    onClose,
    initialFocusRef: closeButtonRef,
    setInert: true,
  });

  // Reset errors when dialog reopens or record changes
  useEffect(() => {
    setPdfError(null);
    setIsDownloading(false);
  }, [record, isOpen]);

  if (!isOpen || !record) return null;

  const handleDownloadPDF = async () => {
    if (!record) return;
    setIsDownloading(true);
    setPdfError(null);
    try {
      await downloadAnalysisReportPDF(record);
    } catch (err: unknown) {
      console.error("[PDF] Failed to generate PDF report:", err);
      setPdfError(
        err instanceof Error
          ? err.message
          : "Failed to generate PDF analysis report.",
      );
    } finally {
      setIsDownloading(false);
    }
  };

  const formattedDate = new Date(record.timestamp).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "medium",
  });

  const formattedConfidence = (record.confidence * 100).toFixed(2);
  const formattedFileSize = record.fileSize
    ? record.fileSize < 1024 * 1024
      ? `${(record.fileSize / 1024).toFixed(1)} KB`
      : `${(record.fileSize / (1024 * 1024)).toFixed(2)} MB`
    : "Unknown";

  return (
    <div
      ref={dialogRef}
      role="dialog"
      aria-modal="true"
      aria-labelledby="history-detail-title"
      tabIndex={-1}
      onClick={(e) => {
        if (e.target === e.currentTarget) {
          onClose();
        }
      }}
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm outline-none"
    >
      <div className="bg-white rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto shadow-2xl border border-slate-200 flex flex-col">
        {/* Header */}
        <div className="p-5 border-b border-slate-100 flex items-start justify-between gap-3 sticky top-0 bg-white z-10">
          <div>
            <div className="flex items-center gap-2">
              <h3
                id="history-detail-title"
                className="text-lg font-bold text-slate-900"
              >
                Analysis Record Details
              </h3>
              <Badge
                variant="outline"
                size="sm"
                className="font-mono text-[10px]"
              >
                v{record.schema_version}
              </Badge>
            </div>
            <p className="text-xs text-slate-500 font-mono mt-0.5 truncate max-w-md">
              ID: {record.id}
            </p>
          </div>
          <button
            ref={closeButtonRef}
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-500 hover:text-slate-600 hover:bg-slate-100 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-clinical-500"
            aria-label="Close details"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Top Classification Summary Card */}
          <div className="p-4 rounded-xl border border-slate-200 bg-slate-50/70 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="space-y-1">
              <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
                Predicted Classification
              </span>
              <div className="flex items-center gap-2">
                <span className="text-xl font-bold text-slate-900">
                  {record.predicted_label ||
                    getDisplayClassName(record.predicted_class)}
                </span>
                <Badge
                  variant={getClassBadgeVariant(record.predicted_class)}
                  size="md"
                >
                  {record.predicted_class}
                </Badge>
              </div>
            </div>

            <div className="sm:text-right">
              <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block">
                Confidence Score
              </span>
              <span className="text-2xl font-extrabold text-slate-900 font-mono">
                {formattedConfidence}%
              </span>
            </div>
          </div>

          {/* Probability Distribution */}
          <div className="space-y-3">
            <h4 className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
              Softmax Probability Distribution
            </h4>
            <div className="space-y-2.5 p-4 rounded-xl border border-slate-200/80 bg-white">
              {MODEL_CLASSES.map((rawClass) => {
                const prob = record.probabilities?.[rawClass] ?? 0;
                const percent = (prob * 100).toFixed(2);
                const isWinner =
                  (record.predicted_class || "").toUpperCase() ===
                  rawClass.toUpperCase();
                const displayName = getDisplayClassName(rawClass);
                const barColor = getClassBarColor(rawClass);

                return (
                  <div key={rawClass} className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-2">
                        <span
                          className={`font-medium ${
                            isWinner
                              ? "text-slate-900 font-bold"
                              : "text-slate-600"
                          }`}
                        >
                          {displayName}
                        </span>
                        {isWinner && (
                          <Badge
                            variant="outline"
                            size="sm"
                            className="text-[10px] py-0 px-1.5"
                          >
                            Predicted
                          </Badge>
                        )}
                      </div>
                      <span className="font-mono font-semibold text-slate-800">
                        {percent}%
                      </span>
                    </div>
                    <div className="h-2 rounded-full bg-slate-100 overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-300 ${barColor}`}
                        style={{ width: `${Math.max(Number(percent), 0.5)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Technical Metadata Grid */}
          <div className="space-y-3">
            <h4 className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
              Technical & Inference Metadata
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
              <div className="p-3 rounded-lg border border-slate-100 bg-slate-50 space-y-1">
                <span className="text-slate-500 text-[11px] flex items-center gap-1.5">
                  <Clock className="w-3.5 h-3.5 text-slate-500" />
                  Analysis Timestamp
                </span>
                <p className="font-semibold text-slate-800">{formattedDate}</p>
              </div>

              <div className="p-3 rounded-lg border border-slate-100 bg-slate-50 space-y-1">
                <span className="text-slate-500 text-[11px] flex items-center gap-1.5">
                  <FileText className="w-3.5 h-3.5 text-slate-500" />
                  File Information
                </span>
                <p
                  className="font-semibold text-slate-800 truncate"
                  title={record.fileName}
                >
                  {record.fileName}
                </p>
                <p className="text-[10px] text-slate-500">
                  Size: {formattedFileSize}{" "}
                  {record.fileType ? `• ${record.fileType}` : ""}
                </p>
              </div>

              <div className="p-3 rounded-lg border border-slate-100 bg-slate-50 space-y-1">
                <span className="text-slate-500 text-[11px] flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5 text-slate-500" />
                  Explainability Target Layer
                </span>
                <p className="font-mono font-semibold text-slate-800 truncate">
                  {record.target_layer || "conv5_block16_2_conv"}
                </p>
                <p className="text-[10px] text-slate-500">
                  Target Class:{" "}
                  {record.selected_target_label ||
                    record.selected_target_class ||
                    record.predicted_label}
                </p>
              </div>

              <div className="p-3 rounded-lg border border-slate-100 bg-slate-50 space-y-1">
                <span className="text-slate-500 text-[11px] flex items-center gap-1.5">
                  <Cpu className="w-3.5 h-3.5 text-slate-500" />
                  Execution Duration
                </span>
                <p className="font-semibold text-slate-800">
                  {record.total_time_ms
                    ? `${record.total_time_ms.toFixed(1)} ms`
                    : "N/A"}
                </p>
                <p className="text-[10px] text-slate-500">
                  Inference:{" "}
                  {record.inference_time_ms
                    ? `${record.inference_time_ms.toFixed(1)} ms`
                    : "N/A"}
                </p>
              </div>

              {record.request_id && (
                <div className="sm:col-span-2 p-3 rounded-lg border border-slate-100 bg-slate-50 space-y-1">
                  <span className="text-slate-400 text-[11px] flex items-center gap-1.5">
                    <Hash className="w-3.5 h-3.5 text-slate-500" />
                    Request Tracking Identifier
                  </span>
                  <p className="font-mono text-slate-700 text-[11px] truncate">
                    {record.request_id}
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Explicit Image Omission & Privacy Notice */}
          <div className="p-3.5 rounded-xl border border-slate-200 bg-slate-50 text-slate-600 text-xs flex items-start gap-2.5 leading-relaxed">
            <ShieldAlert className="w-4 h-4 text-slate-500 shrink-0 mt-0.5" />
            <div>
              <span className="font-semibold text-slate-800">
                Image Retention Policy:{" "}
              </span>
              Original chest radiograph and Grad-CAM heatmap images are
              deliberately not stored in browser history. This metadata-only
              approach prevents unbounded local storage consumption and avoids
              persisting sensitive patient imagery in client browser storage.
            </div>
          </div>
          {/* PDF Generation Failure Notice */}
          {pdfError && (
            <div className="p-3 rounded-lg border border-rose-200 bg-rose-50 text-rose-800 text-xs flex items-start gap-2">
              <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
              <div className="flex-1 leading-relaxed">
                <span className="font-semibold">PDF Export Error: </span>
                {pdfError}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-slate-100 bg-slate-50/50 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <span className="text-[11px] text-slate-400 flex items-center gap-1">
            <Database className="w-3.5 h-3.5 text-slate-400" />
            Stored locally in IndexedDB: lungcare-history
          </span>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={onClose}
              disabled={isDownloading}
            >
              Close
            </Button>
            <Button
              variant="primary"
              size="sm"
              onClick={handleDownloadPDF}
              isLoading={isDownloading}
              leftIcon={<Download className="w-3.5 h-3.5" />}
            >
              Download PDF Report
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default HistoryDetailModal;
