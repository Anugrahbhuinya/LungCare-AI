import { useState } from 'react';
import {
  Sliders,
  Info,
  RefreshCw,
  AlertCircle,
} from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import {
  MODEL_CLASSES,
  getDisplayClassName,
  type PredictionExplainResponse,
} from '@/types/prediction';
import { cn } from '@/utils/cn';

export type ViewMode = 'overlay' | 'heatmap' | 'original' | 'split';

export interface GradCAMViewerProps {
  result: PredictionExplainResponse;
  originalImageUrl?: string;
  onAlphaChange?: (alpha: number) => void;
  onTargetClassChange?: (targetClass: string) => void;
  isLoading?: boolean;
  className?: string;
}

/**
 * Visual Explainability & Grad-CAM Heatmap Viewer Component
 * Phase 3.5: Prediction & Grad-CAM Visualization
 */
export function GradCAMViewer({
  result,
  originalImageUrl,
  onAlphaChange,
  onTargetClassChange,
  isLoading = false,
  className,
}: GradCAMViewerProps) {
  const [viewMode, setViewMode] = useState<ViewMode>('overlay');

  const hasOverlay = Boolean(result.overlay_base64);
  const hasHeatmap = Boolean(result.heatmap_base64);

  return (
    <div className={cn('space-y-4', className)}>
      {/* View Switcher Controls */}
      <div className="flex flex-wrap items-center justify-between gap-2.5 p-2 bg-slate-100/80 rounded-xl border border-slate-200/80">
        <div
          className="flex items-center gap-1"
          role="group"
          aria-label="Grad-CAM visualization display modes"
        >
          <button
            type="button"
            onClick={() => setViewMode('overlay')}
            disabled={!hasOverlay}
            aria-pressed={viewMode === 'overlay'}
            aria-label="View blended overlay of activation heatmap on radiograph"
            className={cn(
              'px-3 py-1.5 rounded-lg text-xs font-semibold transition-all select-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-clinical-500',
              viewMode === 'overlay'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
            )}
          >
            Blended Overlay
          </button>
          <button
            type="button"
            onClick={() => setViewMode('heatmap')}
            disabled={!hasHeatmap}
            aria-pressed={viewMode === 'heatmap'}
            aria-label="View standalone activation heatmap"
            className={cn(
              'px-3 py-1.5 rounded-lg text-xs font-semibold transition-all select-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-clinical-500',
              viewMode === 'heatmap'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
            )}
          >
            Activation Heatmap
          </button>
          {originalImageUrl && (
            <button
              type="button"
              onClick={() => setViewMode('original')}
              aria-pressed={viewMode === 'original'}
              aria-label="View original chest radiograph without heatmap"
              className={cn(
                'px-3 py-1.5 rounded-lg text-xs font-semibold transition-all select-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-clinical-500',
                viewMode === 'original'
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
              )}
            >
              Original X-ray
            </button>
          )}
          {originalImageUrl && (
            <button
              type="button"
              onClick={() => setViewMode('split')}
              aria-pressed={viewMode === 'split'}
              aria-label="View original radiograph and blended overlay side-by-side"
              className={cn(
                'px-3 py-1.5 rounded-lg text-xs font-semibold transition-all select-none hidden sm:inline-block focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-clinical-500',
                viewMode === 'split'
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
              )}
            >
              Side-by-Side
            </button>
          )}
        </div>

        {/* View mode metadata indicator */}
        <div className="flex items-center gap-2 pr-2">
          <Badge variant="outline" size="sm" className="font-mono text-[11px] bg-white">
            Alpha: {result.explanation.alpha}
          </Badge>
          <Badge variant="outline" size="sm" className="font-mono text-[11px] bg-white uppercase">
            {result.explanation.colormap}
          </Badge>
        </div>
      </div>

      {/* Main Radiology Inspection Viewport */}
      <div className="relative rounded-xl overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center min-h-[340px] max-h-[480px] p-4 shadow-inner">
        {/* Loading Overlay when re-calculating with different alpha or target class */}
        {isLoading && (
          <div className="absolute inset-0 bg-slate-950/70 backdrop-blur-xs flex flex-col items-center justify-center z-20 text-white gap-2">
            <RefreshCw className="w-6 h-6 animate-spin motion-reduce:animate-none text-clinical-400" />
            <span className="text-xs font-medium text-slate-300">
              Updating Grad-CAM visualization...
            </span>
          </div>
        )}

        {/* Single View Mode (Overlay, Heatmap, or Original) */}
        {viewMode !== 'split' && (
          <div className="flex items-center justify-center w-full h-full">
            {viewMode === 'overlay' && (
              <div className="relative inline-flex items-center justify-center max-h-[440px] max-w-full">
                {originalImageUrl ? (
                  <>
                    <img
                      src={originalImageUrl}
                      alt="Original radiograph base"
                      className="max-h-[440px] w-auto max-w-full object-contain rounded select-none shadow"
                    />
                    {hasHeatmap && (
                      <img
                        src={result.heatmap_base64}
                        alt={`Grad-CAM overlay blending activation heatmap with radiograph at alpha ${result.explanation.alpha} for category ${getDisplayClassName(result.selected_target_class)}. Visualizes neural network attention regions without confirmed clinical pathology.`}
                        style={{ opacity: result.explanation.alpha }}
                        className="absolute inset-0 w-full h-full object-contain rounded select-none pointer-events-none transition-opacity duration-150 motion-reduce:transition-none"
                      />
                    )}
                  </>
                ) : hasOverlay ? (
                  <img
                    src={result.overlay_base64}
                    alt={`Grad-CAM overlay blending activation heatmap with radiograph at alpha ${result.explanation.alpha} for category ${getDisplayClassName(result.selected_target_class)}. Visualizes neural network attention regions without confirmed clinical pathology.`}
                    className="max-h-[440px] w-auto max-w-full object-contain rounded select-none shadow"
                  />
                ) : null}
              </div>
            )}
            {viewMode === 'heatmap' && hasHeatmap && (
              <img
                src={result.heatmap_base64}
                alt={`Grad-CAM feature activation heatmap for category ${getDisplayClassName(result.selected_target_class)}. Warm color gradients denote model attention areas, not confirmed medical findings.`}
                className="max-h-[440px] w-auto max-w-full object-contain rounded select-none shadow"
              />
            )}
            {viewMode === 'original' && originalImageUrl && (
              <img
                src={originalImageUrl}
                alt="Original input chest radiograph before model feature extraction"
                className="max-h-[440px] w-auto max-w-full object-contain rounded select-none shadow"
              />
            )}

            {/* Error fallback if visualization data is missing */}
            {((viewMode === 'overlay' && !hasOverlay && !originalImageUrl) ||
              (viewMode === 'heatmap' && !hasHeatmap)) && (
              <div className="text-center p-6 text-slate-400 flex flex-col items-center">
                <AlertCircle className="w-8 h-8 text-amber-500 mb-2" />
                <p className="text-sm font-medium text-slate-200">
                  Visualization Not Available
                </p>
                <p className="text-xs text-slate-400 mt-1 max-w-xs">
                  The backend did not return image data for this visualization mode.
                </p>
              </div>
            )}
          </div>
        )}

        {/* Split Side-by-Side View Mode */}
        {viewMode === 'split' && originalImageUrl && (
          <div className="grid grid-cols-2 gap-3 w-full h-full items-center">
            <div className="flex flex-col items-center">
              <span className="text-[11px] font-semibold text-slate-400 mb-1.5 uppercase tracking-wider">
                Original Radiograph
              </span>
              <img
                src={originalImageUrl}
                alt="Original input chest radiograph displayed for visual comparison"
                className="max-h-[400px] w-auto max-w-full object-contain rounded border border-slate-800 shadow"
              />
            </div>
            <div className="flex flex-col items-center">
              <span className="text-[11px] font-semibold text-clinical-400 mb-1.5 uppercase tracking-wider">
                Grad-CAM Overlay ({getDisplayClassName(result.selected_target_class)})
              </span>
              <div className="relative inline-flex items-center justify-center max-h-[400px]">
                <img
                  src={originalImageUrl}
                  alt="Original radiograph base"
                  className="max-h-[400px] w-auto max-w-full object-contain rounded border border-slate-800 shadow"
                />
                {hasHeatmap ? (
                  <img
                    src={result.heatmap_base64}
                    alt={`Grad-CAM overlay blending activation heatmap with radiograph for category ${getDisplayClassName(result.selected_target_class)}. Visualizes neural network attention regions.`}
                    style={{ opacity: result.explanation.alpha }}
                    className="absolute inset-0 w-full h-full object-contain rounded pointer-events-none transition-opacity duration-150 motion-reduce:transition-none"
                  />
                ) : hasOverlay ? (
                  <img
                    src={result.overlay_base64}
                    alt={`Grad-CAM overlay blending activation heatmap with radiograph for category ${getDisplayClassName(result.selected_target_class)}. Visualizes neural network attention regions.`}
                    className="max-h-[400px] w-auto max-w-full object-contain rounded border border-slate-800 shadow"
                  />
                ) : null}
              </div>
            </div>
          </div>
        )}

        {/* Viewport Floating Tag */}
        <div className="absolute top-3 left-3 pointer-events-none">
          <Badge
            variant="clinical"
            size="sm"
            className="bg-slate-900/80 text-clinical-300 border-slate-700 backdrop-blur-xs font-mono text-[10px]"
          >
            Target: {getDisplayClassName(result.selected_target_class)}
          </Badge>
        </div>
      </div>

      {/* Interactive Explanation Controls (Alpha & Target Class) */}
      <div className="p-3.5 rounded-xl border border-slate-200/80 bg-slate-50 space-y-3">
        {/* Target class selection for Grad-CAM differentiation */}
        {onTargetClassChange && (
          <div className="flex flex-wrap items-center justify-between gap-2 pb-2.5 border-b border-slate-200/60">
            <span id="target-category-label" className="text-xs font-semibold text-slate-700">
              Grad-CAM Target Category:
            </span>
            <div
              className="flex flex-wrap items-center gap-1.5"
              role="group"
              aria-labelledby="target-category-label"
            >
              {MODEL_CLASSES.map((cls) => {
                const isSelected = result.selected_target_class.toUpperCase() === cls.toUpperCase();
                const displayLabel = getDisplayClassName(cls);
                return (
                  <button
                    key={cls}
                    type="button"
                    onClick={() => onTargetClassChange(cls)}
                    disabled={isLoading}
                    aria-pressed={isSelected}
                    aria-label={`Select ${displayLabel} as Grad-CAM explanation target`}
                    className={cn(
                      'px-2.5 py-1 rounded text-xs font-medium transition-colors select-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-clinical-500',
                      isSelected
                        ? 'bg-clinical-700 text-white shadow-xs font-semibold'
                        : 'bg-white text-slate-700 border border-slate-200 hover:bg-slate-100 disabled:opacity-50'
                    )}
                  >
                    {displayLabel}
                  </button>
                );
              })}
            </div>
          </div>
        )}

        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-clinical-600" />
            <span id="alpha-blend-label" className="text-xs font-semibold text-slate-800">
              Interactive Blend Factor (Alpha)
            </span>
          </div>

          {/* Quick alpha weight buttons */}
          {onAlphaChange && (
            <div
              className="flex items-center gap-1.5"
              role="group"
              aria-labelledby="alpha-blend-label"
            >
              {[0.2, 0.4, 0.6, 0.8].map((a) => {
                const isActive = Math.abs(result.explanation.alpha - a) < 0.05;
                return (
                  <button
                    key={a}
                    type="button"
                    onClick={() => onAlphaChange(a)}
                    disabled={isLoading}
                    aria-pressed={isActive}
                    aria-label={`Set blend factor alpha to ${a.toFixed(1)}`}
                    className={cn(
                      'px-2.5 py-1 rounded text-xs font-mono font-medium transition-colors select-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-clinical-500',
                      isActive
                        ? 'bg-clinical-700 text-white shadow-xs'
                        : 'bg-white text-slate-700 border border-slate-200 hover:bg-slate-100 disabled:opacity-50'
                    )}
                  >
                    {a.toFixed(1)}
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Explainability Guidance Notice */}
        <div className="p-3 rounded-lg border border-slate-200/60 bg-white text-[11px] text-slate-600 flex items-start gap-2.5 leading-relaxed">
          <Info className="w-4 h-4 text-slate-500 shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold text-slate-800">Understanding Grad-CAM:</span>{' '}
            Gradient-weighted Class Activation Mapping computes gradients with respect to convolutional layer{' '}
            <code className="bg-slate-100 px-1 py-0.5 rounded text-slate-800 font-mono text-[10px]">
              {result.explanation.target_layer}
            </code>.
            Warm colors highlight regions that positively influenced the neural network's decision for class{' '}
            <span className="font-semibold text-slate-800">{getDisplayClassName(result.selected_target_class)}</span>.
            This is an interpretability tool for research, not a validated medical lesion localization mechanism.
          </div>
        </div>
      </div>
    </div>
  );
}

export default GradCAMViewer;
