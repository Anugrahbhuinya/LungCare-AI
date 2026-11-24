import { useState, useEffect, useRef } from 'react';
import { Menu, User } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { useHealthCheck } from '@/hooks/useHealthCheck';

export interface HeaderProps {
  pageTitle: string;
  contextLabel?: string;
  onOpenMobileMenu: () => void;
}

export function Header({
  pageTitle,
  contextLabel = 'Clinical Research Workstation',
  onOpenMobileMenu,
}: HeaderProps) {
  const { backendStatus, modelStatus, lastChecked, error, refetch } = useHealthCheck();

  const [statusAnnouncement, setStatusAnnouncement] = useState<string>('');
  const prevBackendRef = useRef<string>(backendStatus);
  const prevModelRef = useRef<string>(modelStatus);
  const hasInitializedRef = useRef<boolean>(false);

  useEffect(() => {
    // Skip initial mount announcement if everything starts checking/normal
    if (!hasInitializedRef.current) {
      hasInitializedRef.current = true;
      prevBackendRef.current = backendStatus;
      prevModelRef.current = modelStatus;
      return;
    }

    const backendChanged = prevBackendRef.current !== backendStatus;
    const modelChanged = prevModelRef.current !== modelStatus;

    if (backendChanged || modelChanged) {
      const parts: string[] = [];
      if (backendChanged) {
        const readable =
          backendStatus === 'healthy'
            ? 'healthy'
            : backendStatus === 'checking'
            ? 'checking connection'
            : 'unavailable';
        parts.push(`Backend API is ${readable}.`);
      }
      if (modelChanged) {
        const readable =
          modelStatus === 'ready'
            ? 'ready'
            : modelStatus === 'checking'
            ? 'verifying model weights'
            : modelStatus === 'not_loaded'
            ? 'not loaded'
            : 'unavailable';
        parts.push(`DenseNet121 model is ${readable}.`);
      }
      if (parts.length > 0) {
        setStatusAnnouncement(parts.join(' '));
      }
      prevBackendRef.current = backendStatus;
      prevModelRef.current = modelStatus;
    }
  }, [backendStatus, modelStatus]);

  const formattedTime = lastChecked
    ? lastChecked.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    : null;

  return (
    <header className="h-16 px-4 sm:px-6 bg-white border-b border-slate-200 flex items-center justify-between gap-4 sticky top-0 z-20">
      {/* Screen Reader Live Region for Backend & Model Status Changes */}
      <div
        role="status"
        aria-live="polite"
        aria-atomic="true"
        className="sr-only"
      >
        {statusAnnouncement}
      </div>

      <div className="flex items-center gap-3 min-w-0">
        <button
          type="button"
          onClick={onOpenMobileMenu}
          className="lg:hidden p-2 -ml-2 rounded-lg text-slate-600 hover:text-slate-900 hover:bg-slate-100 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-clinical-500 focus-visible:ring-offset-2"
          aria-label="Open mobile navigation"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div className="flex flex-col min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium text-slate-500 hidden sm:inline">
              {contextLabel}
            </span>
            <span className="text-xs text-slate-500 hidden sm:inline">•</span>
            <h1 className="text-base sm:text-lg font-bold text-slate-900 truncate tracking-tight">
              {pageTitle}
            </h1>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-3 shrink-0">
        {/* Mobile compact connection status dot pair */}
        <div
          className="flex sm:hidden items-center gap-1.5 px-2 py-1 bg-slate-50 rounded-md border border-slate-200/80 cursor-pointer"
          onClick={() => refetch()}
          title={`API: ${backendStatus} | Model: ${modelStatus}${formattedTime ? ` (${formattedTime})` : ''}`}
        >
          <span
            className={`w-2 h-2 rounded-full ${
              backendStatus === 'healthy'
                ? 'bg-emerald-500'
                : backendStatus === 'checking'
                ? 'bg-amber-400 animate-pulse motion-reduce:animate-none'
                : 'bg-rose-500'
            }`}
          />
          <span
            className={`w-2 h-2 rounded-full ${
              modelStatus === 'ready'
                ? 'bg-clinical-600'
                : modelStatus === 'checking'
                ? 'bg-amber-400 animate-pulse motion-reduce:animate-none'
                : modelStatus === 'not_loaded'
                ? 'bg-amber-500'
                : 'bg-rose-500'
            }`}
          />
        </div>

        {/* Live Backend API Status Badge (sm and larger) */}
        <Badge
          variant="outline"
          size="sm"
          className={`hidden sm:inline-flex items-center gap-1.5 py-1 transition-colors ${
            backendStatus === 'healthy'
              ? 'text-slate-700 border-slate-200 bg-white'
              : backendStatus === 'checking'
              ? 'text-amber-700 border-amber-200 bg-amber-50/60'
              : 'text-rose-700 border-rose-200 bg-rose-50/60'
          }`}
          title={
            formattedTime
              ? `Backend Status: ${backendStatus} (Last checked: ${formattedTime})`
              : `Backend Status: ${backendStatus}`
          }
        >
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              backendStatus === 'healthy'
                ? 'bg-emerald-500'
                : backendStatus === 'checking'
                ? 'bg-amber-400 animate-pulse motion-reduce:animate-none'
                : 'bg-rose-500'
            }`}
          />
          <span>
            API:{' '}
            {backendStatus === 'healthy'
              ? 'Healthy'
              : backendStatus === 'checking'
              ? 'Checking'
              : 'Unavailable'}
          </span>
        </Badge>

        {/* Live ML Model Readiness Badge (md and larger) */}
        <Badge
          variant="outline"
          size="sm"
          className={`hidden md:inline-flex items-center gap-1.5 py-1 transition-colors ${
            modelStatus === 'ready'
              ? 'text-slate-700 border-slate-200 bg-white'
              : modelStatus === 'checking'
              ? 'text-slate-600 border-slate-200 bg-slate-50'
              : modelStatus === 'not_loaded'
              ? 'text-amber-700 border-amber-200 bg-amber-50/60'
              : 'text-rose-700 border-rose-200 bg-rose-50/60'
          }`}
          title={
            modelStatus === 'ready'
              ? 'DenseNet121 weights loaded into memory and ready for inference'
              : modelStatus === 'not_loaded'
              ? 'Backend is alive, but ML model is not loaded'
              : error || 'ML model service is currently unavailable'
          }
        >
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
          <span>
            Model:{' '}
            {modelStatus === 'ready'
              ? 'Ready'
              : modelStatus === 'checking'
              ? 'Checking'
              : modelStatus === 'not_loaded'
              ? 'Not Loaded'
              : 'Unavailable'}
          </span>
        </Badge>

        <div className="h-5 w-px bg-slate-200 hidden sm:block" />

        {/* Non-fictional generic workstation identity */}
        <div className="flex items-center gap-2.5 pl-1">
          <div className="w-8 h-8 rounded-full bg-slate-100 border border-slate-200 text-slate-600 flex items-center justify-center font-semibold text-xs shadow-xs">
            <User className="w-4 h-4 text-slate-500" />
          </div>
          <div className="hidden md:flex flex-col text-left">
            <span className="text-xs font-semibold text-slate-800 leading-tight">
              Research Workspace
            </span>
            <span className="text-[10px] text-slate-400 leading-tight">
              Guest User (No Auth)
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
