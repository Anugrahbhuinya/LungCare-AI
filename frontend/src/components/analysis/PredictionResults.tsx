import { Hash } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import {
  MODEL_CLASSES,
  getDisplayClassName,
  type PredictionExplainResponse,
} from '@/types/prediction';
import { cn } from '@/utils/cn';

export interface PredictionResultsProps {
  result: PredictionExplainResponse;
  className?: string;
}

/**
 * Classification category badge and color theme helper.
 */
function getClassTheme(className: string) {
  switch (className.toUpperCase()) {
    case 'COVID19':
      return {
        variant: 'clinical' as const,
        barColor: 'bg-clinical-600',
        textColor: 'text-clinical-700',
        badgeBg: 'bg-clinical-50 text-clinical-800 border-clinical-200',
      };
    case 'NORMAL':
      return {
        variant: 'success' as const,
        barColor: 'bg-emerald-600',
        textColor: 'text-emerald-700',
        badgeBg: 'bg-emerald-50 text-emerald-800 border-emerald-200',
      };
    case 'PNEUMONIA':
      return {
        variant: 'warning' as const,
        barColor: 'bg-amber-600',
        textColor: 'text-amber-700',
        badgeBg: 'bg-amber-50 text-amber-800 border-amber-200',
      };
    case 'TURBERCULOSIS':
      return {
        variant: 'danger' as const,
        barColor: 'bg-rose-600',
        textColor: 'text-rose-700',
        badgeBg: 'bg-rose-50 text-rose-800 border-rose-200',
      };
    default:
      return {
        variant: 'default' as const,
        barColor: 'bg-slate-600',
        textColor: 'text-slate-700',
        badgeBg: 'bg-slate-100 text-slate-800 border-slate-200',
      };
  }
}

/**
 * Four-Class DenseNet121 Prediction Results & Softmax Distribution UI
 * Phase 3.5: Prediction & Grad-CAM Visualization
 */
export function PredictionResults({ result, className }: PredictionResultsProps) {
  const predictedTheme = getClassTheme(result.predicted_class);
  const formattedConfidence = (result.confidence * 100).toFixed(2);
  const displayPredicted = getDisplayClassName(result.predicted_class);

  return (
    <div className={cn('space-y-4', className)}>
      {/* Top Banner: Predicted Category & Primary Confidence */}
      <div className="p-4 sm:p-5 rounded-xl border border-slate-200 bg-white shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div className="space-y-1">
            <div className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Model Predicted Class
            </div>
            <div className="flex items-center gap-2.5">
              <h3 className="text-2xl font-bold text-slate-900 tracking-tight">
                {displayPredicted}
              </h3>
              <Badge
                variant={predictedTheme.variant}
                size="md"
                className="font-semibold text-xs"
              >
                Top Classification
              </Badge>
            </div>
          </div>

          <div className="sm:text-right space-y-0.5">
            <div className="text-xs font-medium text-slate-500">
              Softmax Probability
            </div>
            <div className="text-3xl font-extrabold text-slate-900 tracking-tight font-mono">
              {formattedConfidence}%
            </div>
          </div>
        </div>

        <p className="text-xs text-slate-500 mt-3 pt-3 border-t border-slate-100 leading-relaxed">
          DenseNet121 transfer learning prediction evaluated at resolution 224 × 224 RGB.
          Confidence reflects research model probability distribution; it is not a clinical measure of severity or diagnostic certainty.
        </p>
      </div>

      {/* Probability Distribution Spectrum */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold flex items-center justify-between">
            <span>Softmax Probability Distribution</span>
            <span className="text-xs font-normal text-slate-500">
              4 Classification Categories
            </span>
          </CardTitle>
          <CardDescription>
            Categorical likelihood across the four trained radiograph classes.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {MODEL_CLASSES.map((rawClass) => {
            const prob = result.probabilities[rawClass] ?? 0;
            const probPercent = (prob * 100).toFixed(2);
            const isPredicted = result.predicted_class.toUpperCase() === rawClass.toUpperCase();
            const theme = getClassTheme(rawClass);
            const label = getDisplayClassName(rawClass);

            return (
              <div
                key={rawClass}
                className={cn(
                  'p-3 rounded-lg border transition-all',
                  isPredicted
                    ? 'border-slate-300 bg-slate-50/80 shadow-xs ring-1 ring-slate-200'
                    : 'border-slate-100 bg-white hover:bg-slate-50/50'
                )}
              >
                <div className="flex items-center justify-between text-xs mb-1.5">
                  <div className="flex items-center gap-2">
                    <span
                      className={cn(
                        'font-semibold',
                        isPredicted ? 'text-slate-900' : 'text-slate-700'
                      )}
                    >
                      {label}
                    </span>
                    {isPredicted && (
                      <span className="inline-flex items-center px-1.5 py-0.2 rounded text-[10px] font-semibold bg-slate-800 text-white">
                        Predicted
                      </span>
                    )}
                  </div>
                  <span className="font-mono font-semibold text-slate-800">
                    {probPercent}%
                  </span>
                </div>

                {/* Progress Bar */}
                <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                  <div
                    className={cn(
                      'h-full rounded-full transition-all duration-500',
                      isPredicted ? theme.barColor : 'bg-slate-400'
                    )}
                    style={{ width: `${Math.max(prob * 100, 0.5)}%` }}
                    role="progressbar"
                    aria-valuenow={prob * 100}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-label={`${label} probability`}
                  />
                </div>
              </div>
            );
          })}
        </CardContent>
      </Card>

      {/* Model Telemetry & Provenance Strip */}
      <div className="p-3.5 rounded-lg border border-slate-200/80 bg-slate-50 text-[11px] text-slate-600 space-y-2">
        <div className="font-semibold text-slate-700 text-xs flex items-center justify-between">
          <span>Inference & Telemetry</span>
          <span className="font-mono text-slate-400">
            Total: {result.explanation.total_time_ms.toFixed(1)} ms
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1 border-t border-slate-200/60">
          <div>
            <span className="text-slate-400 block">Forward Pass:</span>
            <span className="font-mono font-medium text-slate-800">
              {result.explanation.inference_time_ms.toFixed(1)} ms
            </span>
          </div>
          <div>
            <span className="text-slate-400 block">Grad-CAM:</span>
            <span className="font-mono font-medium text-slate-800">
              {result.explanation.gradcam_time_ms.toFixed(1)} ms
            </span>
          </div>
          <div>
            <span className="text-slate-400 block">Backbone Layer:</span>
            <span className="font-mono font-medium text-slate-800">
              {result.explanation.target_layer}
            </span>
          </div>
          <div>
            <span className="text-slate-400 block">Colormap:</span>
            <span className="font-mono font-medium text-slate-800 uppercase">
              {result.explanation.colormap}
            </span>
          </div>
        </div>

        {result.request_id && (
          <div className="flex items-center gap-1.5 pt-1 text-[10px] text-slate-400 font-mono truncate">
            <Hash className="w-3 h-3 shrink-0" />
            <span className="truncate">Correlation ID: {result.request_id}</span>
          </div>
        )}
      </div>
    </div>
  );
}

export default PredictionResults;
