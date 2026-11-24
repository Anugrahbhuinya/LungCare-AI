import React, { useRef, useState } from 'react';
import {
  Trash2,
  RefreshCw,
  FileImage,
  AlertTriangle,
} from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import type { SelectedImage } from '@/hooks/useImageUpload';
import { cn } from '@/utils/cn';

export interface ImagePreviewProps {
  image: SelectedImage;
  onRemove: () => void;
  onReplace: (file: File) => void;
  onDimensionsLoaded?: (dims: { width: number; height: number }) => void;
  onDecodeError?: (message: string) => void;
  className?: string;
}

/**
 * Selected Radiograph Preview Card for LungCare AI
 * Phase 3.4: X-ray Upload & Preview
 */
export function ImagePreview({
  image,
  onRemove,
  onReplace,
  onDimensionsLoaded,
  onDecodeError,
  className,
}: ImagePreviewProps) {
  const [hasDecodeError, setHasDecodeError] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const replaceInputRef = useRef<HTMLInputElement | null>(null);

  const handleReplaceClick = () => {
    replaceInputRef.current?.click();
  };

  const handleReplaceFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      onReplace(files[0]);
    }
    if (replaceInputRef.current) {
      replaceInputRef.current.value = '';
    }
  };

  const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    setIsLoaded(true);
    setHasDecodeError(false);
    if (img.naturalWidth && img.naturalHeight) {
      onDimensionsLoaded?.({
        width: img.naturalWidth,
        height: img.naturalHeight,
      });
    }
  };

  const handleImageError = () => {
    setHasDecodeError(true);
    setIsLoaded(false);
    onDecodeError?.(
      'The browser failed to render this image. The file may be corrupted, truncated, or formatted in an unsupported variant.'
    );
  };

  return (
    <div className={cn('space-y-4', className)}>
      {/* Hidden file input for Replace action */}
      <input
        ref={replaceInputRef}
        type="file"
        accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
        onChange={handleReplaceFileChange}
        className="hidden"
        aria-hidden="true"
        tabIndex={-1}
      />

      {/* Metadata strip */}
      <div className="flex flex-wrap items-center justify-between gap-2 p-3 bg-slate-50 border border-slate-200/80 rounded-lg">
        <div className="flex items-center gap-2 min-w-0">
          <FileImage className="w-4 h-4 text-clinical-600 shrink-0" />
          <span
            className="text-xs font-semibold text-slate-800 truncate max-w-[200px] sm:max-w-xs"
            title={image.name}
          >
            {image.name}
          </span>
        </div>

        <div className="flex items-center gap-1.5 shrink-0">
          <Badge variant="outline" size="sm" className="font-mono text-[11px]">
            {image.extension}
          </Badge>
          <Badge variant="outline" size="sm" className="font-mono text-[11px]">
            {image.formattedSize}
          </Badge>
          {image.dimensions && (
            <Badge variant="clinical" size="sm" className="font-mono text-[11px]">
              {image.dimensions.width} × {image.dimensions.height} px
            </Badge>
          )}
        </div>
      </div>

      {/* Medical-grade viewport container */}
      <div className="relative rounded-xl overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center min-h-[300px] max-h-[460px] p-3 shadow-inner">
        {hasDecodeError ? (
          <div className="p-6 text-center flex flex-col items-center justify-center text-rose-400">
            <AlertTriangle className="w-10 h-10 mb-2.5 text-rose-500" />
            <h5 className="text-sm font-semibold text-white">Image Decoding Failed</h5>
            <p className="text-xs text-slate-400 mt-1 max-w-xs leading-relaxed">
              Unable to render this radiograph. Please replace the file with a valid, uncorrupted JPEG, PNG, or WEBP image.
            </p>
            <Button
              variant="outline"
              size="sm"
              onClick={handleReplaceClick}
              className="mt-4 bg-slate-900 border-slate-700 text-slate-200 hover:bg-slate-800 hover:text-white"
            >
              Replace Image
            </Button>
          </div>
        ) : (
          <>
            <img
              src={image.previewUrl}
              alt={`Staged chest radiograph input: ${image.name}${image.dimensions ? ` (${image.dimensions.width} by ${image.dimensions.height} pixels)` : ''}`}
              onLoad={handleImageLoad}
              onError={handleImageError}
              className={cn(
                'max-h-[430px] w-auto max-w-full object-contain rounded select-none transition-opacity duration-300 motion-reduce:transition-none',
                isLoaded ? 'opacity-100' : 'opacity-0'
              )}
            />
            {!isLoaded && (
              <div className="absolute inset-0 flex items-center justify-center text-slate-400 text-xs">
                Rendering radiograph preview...
              </div>
            )}
          </>
        )}
      </div>

      {/* Action buttons bar */}
      <div className="flex items-center justify-between gap-3 pt-1">
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={handleReplaceClick}
          leftIcon={<RefreshCw className="w-3.5 h-3.5 text-slate-600" />}
          className="text-xs"
        >
          Replace Image
        </Button>

        <Button
          type="button"
          variant="danger"
          size="sm"
          onClick={onRemove}
          leftIcon={<Trash2 className="w-3.5 h-3.5" />}
          className="text-xs"
        >
          Remove Image
        </Button>
      </div>
    </div>
  );
}

export default ImagePreview;
