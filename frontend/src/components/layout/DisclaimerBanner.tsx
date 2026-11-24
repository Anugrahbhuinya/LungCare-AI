import React from 'react';
import { AlertTriangle, Info, X } from 'lucide-react';
import { cn } from '@/utils/cn';

export interface DisclaimerBannerProps {
  className?: string;
  variant?: 'warning' | 'info';
  dismissible?: boolean;
}

export const DISCLAIMER_TEXT =
  'LungCare AI is an experimental research prototype. Its predictions are not medical diagnoses and must not replace professional medical evaluation.';

export function DisclaimerBanner({
  className,
  variant = 'warning',
  dismissible = false,
}: DisclaimerBannerProps) {
  const [isDismissed, setIsDismissed] = React.useState(false);

  if (isDismissed) {
    return null;
  }

  const isWarning = variant === 'warning';

  return (
    <aside
      aria-label="Clinical Research Disclaimer"
      className={cn(
        'w-full border-b transition-all duration-200 select-none',
        isWarning
          ? 'bg-amber-50/90 border-amber-200/80 text-amber-900'
          : 'bg-clinical-50/90 border-clinical-200/80 text-clinical-900',
        className
      )}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-2.5 flex items-center justify-between gap-3 text-xs sm:text-sm">
        <div className="flex items-center gap-2.5 min-w-0">
          {isWarning ? (
            <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
          ) : (
            <Info className="w-4 h-4 text-clinical-600 shrink-0" />
          )}
          <p className="font-medium tracking-tight text-slate-800 leading-snug">
            <span className="font-semibold text-slate-900">Research Disclaimer: </span>
            {DISCLAIMER_TEXT}
          </p>
        </div>

        {dismissible && (
          <button
            type="button"
            onClick={() => setIsDismissed(true)}
            className="text-amber-800 hover:text-amber-950 hover:bg-amber-100/80 p-1 rounded-md transition-colors shrink-0"
            aria-label="Dismiss disclaimer"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        )}
      </div>
    </aside>
  );
}
