import React, { useRef, useState, useEffect, useCallback } from 'react';
import { UploadCloud, AlertCircle, FileImage, ShieldAlert } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { cn } from '@/utils/cn';

export interface ImageUploaderProps {
  onFileSelect: (file: File) => void;
  error?: string | null;
  disabled?: boolean;
  className?: string;
}

/**
 * Reusable Drag-and-Drop and File-Browser Upload Component for LungCare AI
 * Phase 3.4: X-ray Upload & Preview
 */
export function ImageUploader({
  onFileSelect,
  error,
  disabled = false,
  className,
}: ImageUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const dragCounter = useRef<number>(0);

  // Global window drop prevention: prevents browser from navigating away if dropped outside zone
  useEffect(() => {
    const handleGlobalDragOver = (e: DragEvent) => {
      e.preventDefault();
    };
    const handleGlobalDrop = (e: DragEvent) => {
      e.preventDefault();
    };

    window.addEventListener('dragover', handleGlobalDragOver);
    window.addEventListener('drop', handleGlobalDrop);

    return () => {
      window.removeEventListener('dragover', handleGlobalDragOver);
      window.removeEventListener('drop', handleGlobalDrop);
    };
  }, []);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current += 1;
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
      setIsDragging(true);
    }
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current -= 1;
    if (dragCounter.current === 0) {
      setIsDragging(false);
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);
      dragCounter.current = 0;

      if (disabled) return;

      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        onFileSelect(files[0]);
      }
    },
    [disabled, onFileSelect]
  );

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      onFileSelect(files[0]);
    }
    // Reset file input value so selecting the same file again triggers change event
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleZoneClick = () => {
    if (!disabled && fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.key === 'Enter' || e.key === ' ') && !disabled) {
      e.preventDefault();
      fileInputRef.current?.click();
    }
  };

  return (
    <div className={cn('space-y-3', className)}>
      {/* Hidden native file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
        onChange={handleFileInputChange}
        className="hidden"
        disabled={disabled}
        aria-hidden="true"
        tabIndex={-1}
      />

      {/* Interactive Dropzone */}
      <div
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-label="Upload chest X-ray image"
        aria-disabled={disabled}
        onClick={handleZoneClick}
        onKeyDown={handleKeyDown}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        className={cn(
          'relative border-2 border-dashed rounded-xl p-8 sm:p-10 text-center transition-all duration-200 motion-reduce:transition-none select-none outline-none',
          'flex flex-col items-center justify-center cursor-pointer',
          // Dragging state
          isDragging
            ? 'border-clinical-500 bg-clinical-50/70 ring-4 ring-clinical-500/15 scale-[1.01] motion-reduce:scale-100'
            : error
            ? 'border-rose-300 bg-rose-50/30 hover:border-rose-400'
            : 'border-slate-500 bg-slate-50/50 hover:bg-slate-50 hover:border-slate-600',
          // Focus state
          'focus-visible:ring-2 focus-visible:ring-clinical-500 focus-visible:ring-offset-2',
          // Disabled state
          disabled && 'opacity-50 cursor-not-allowed pointer-events-none'
        )}
      >
        <div
          className={cn(
            'w-14 h-14 rounded-2xl flex items-center justify-center mb-4 transition-all duration-200 motion-reduce:transition-none',
            isDragging
              ? 'bg-clinical-600 text-white shadow-md scale-110 motion-reduce:scale-100'
              : error
              ? 'bg-rose-100 text-rose-600'
              : 'bg-clinical-50 text-clinical-600'
          )}
        >
          {isDragging ? (
            <FileImage className="w-7 h-7 animate-pulse motion-reduce:animate-none" />
          ) : (
            <UploadCloud className="w-7 h-7" />
          )}
        </div>

        <h4 className="text-sm sm:text-base font-semibold text-slate-800">
          {isDragging ? 'Drop chest X-ray to inspect' : 'Drag and drop chest X-ray here'}
        </h4>

        <p className="text-xs text-slate-500 mt-1 max-w-xs leading-relaxed">
          Supports high-resolution radiographs in <span className="font-medium text-slate-700">JPEG, PNG</span>, or <span className="font-medium text-slate-700">WEBP</span> format.
        </p>

        <div className="mt-5 flex items-center gap-2">
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={disabled}
            className="pointer-events-none shadow-xs text-xs font-semibold bg-white"
          >
            Browse Local Files
          </Button>
        </div>

        {/* Format & Size Specs */}
        <div className="mt-4 pt-4 border-t border-slate-200/60 w-full max-w-xs flex items-center justify-center gap-3 text-[11px] text-slate-400">
          <span>Max: 10 MB</span>
          <span>•</span>
          <span>Target: 224 × 224 px</span>
          <span>•</span>
          <span>RGB / Grayscale</span>
        </div>
      </div>

      {/* Validation Error Banner */}
      {error && (
        <div
          role="alert"
          className="p-3.5 rounded-lg border border-rose-200 bg-rose-50/80 text-rose-800 text-xs flex items-start gap-2.5"
        >
          <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
          <div className="flex-1 leading-relaxed">
            <span className="font-semibold block mb-0.5">Validation Notice:</span>
            {error}
          </div>
        </div>
      )}

      {/* Upload Privacy Notice */}
      <div className="p-3 rounded-lg border border-slate-200/80 bg-slate-50 text-[11px] text-slate-600 flex items-start gap-2 leading-relaxed">
        <ShieldAlert className="w-4 h-4 text-slate-500 shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold text-slate-800">Privacy Notice:</span>{' '}
          Do not upload radiographs containing visible patient identifiers (such as names, dates of birth, or medical record numbers). Persisted scan history uses neutral filenames (<code className="bg-white px-1 py-0.5 rounded border border-slate-200 font-mono text-[10px] text-slate-700">radiograph-*.png</code>), but filename sanitization does not de-identify burned-in text within the image.
        </div>
      </div>
    </div>
  );
}

export default ImageUploader;
