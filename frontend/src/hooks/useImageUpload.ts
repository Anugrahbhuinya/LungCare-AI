import { useState, useCallback, useEffect, useRef } from 'react';
import {
  validateImageFile,
  formatFileSize,
  getFileExtension,
} from '@/utils/fileValidation';

export interface SelectedImage {
  file: File;
  previewUrl: string;
  name: string;
  size: number;
  formattedSize: string;
  type: string;
  extension: string;
  dimensions?: { width: number; height: number };
}

export interface UseImageUploadReturn {
  selectedImage: SelectedImage | null;
  error: string | null;
  selectImage: (file: File) => boolean;
  replaceImage: (file: File) => boolean;
  removeImage: () => void;
  setDimensions: (dimensions: { width: number; height: number }) => void;
  setDecodeError: (message?: string) => void;
  clearError: () => void;
}

/**
 * Hook to manage local image selection, validation, object URL lifecycle, and metadata.
 *
 * Enforces memory safety by explicitly revoking browser object URLs when
 * an image is removed, replaced, or when the consuming component unmounts.
 */
export function useImageUpload(): UseImageUploadReturn {
  const [selectedImage, setSelectedImage] = useState<SelectedImage | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Keep a ref to the active preview URL so unmount cleanup always revokes the latest URL
  const activeUrlRef = useRef<string | null>(null);

  const cleanupActiveUrl = useCallback(() => {
    if (activeUrlRef.current) {
      URL.revokeObjectURL(activeUrlRef.current);
      activeUrlRef.current = null;
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cleanupActiveUrl();
    };
  }, [cleanupActiveUrl]);

  const selectImage = useCallback(
    (file: File): boolean => {
      // 1. Validate file format and size
      const validation = validateImageFile(file);
      if (!validation.isValid) {
        setError(validation.error);
        return false;
      }

      // 2. Revoke any previously allocated Object URL to avoid memory leakage
      cleanupActiveUrl();

      // 3. Create fresh Object URL for local browser preview
      const previewUrl = URL.createObjectURL(file);
      activeUrlRef.current = previewUrl;

      // 4. Derive image metadata
      const newImage: SelectedImage = {
        file,
        previewUrl,
        name: file.name,
        size: file.size,
        formattedSize: formatFileSize(file.size),
        type: file.type || 'image/jpeg',
        extension: getFileExtension(file.name).replace('.', '').toUpperCase() || 'IMAGE',
      };

      setSelectedImage(newImage);
      setError(null);
      return true;
    },
    [cleanupActiveUrl]
  );

  const replaceImage = useCallback(
    (file: File): boolean => {
      return selectImage(file);
    },
    [selectImage]
  );

  const removeImage = useCallback(() => {
    cleanupActiveUrl();
    setSelectedImage(null);
    setError(null);
  }, [cleanupActiveUrl]);

  const setDimensions = useCallback((dimensions: { width: number; height: number }) => {
    setSelectedImage((prev) => (prev ? { ...prev, dimensions } : null));
  }, []);

  const setDecodeError = useCallback((message?: string) => {
    setError(
      message ||
        'The selected image appears to be corrupted or unreadable by the browser. Please choose a valid chest X-ray.'
    );
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    selectedImage,
    error,
    selectImage,
    replaceImage,
    removeImage,
    setDimensions,
    setDecodeError,
    clearError,
  };
}

export default useImageUpload;
