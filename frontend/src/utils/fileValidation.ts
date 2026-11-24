/**
 * Image file validation and formatting utilities for LungCare AI
 * Phase 3.4: X-ray Upload & Preview
 */

export const SUPPORTED_IMAGE_MIME_TYPES = [
  'image/jpeg',
  'image/png',
  'image/webp',
] as const;

export const SUPPORTED_IMAGE_EXTENSIONS = [
  '.jpg',
  '.jpeg',
  '.png',
  '.webp',
] as const;

/**
 * Maximum permitted file size: 10 Megabytes (10 * 1024 * 1024 bytes)
 */
export const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10,485,760 bytes
export const MAX_FILE_SIZE_MB = 10;

export interface FileValidationResult {
  isValid: boolean;
  error: string | null;
}

/**
 * Format bytes into human-readable binary unit representation (e.g. 2.45 MB, 840 KB)
 */
export function formatFileSize(bytes: number): string {
  if (bytes < 0) return '0 B';
  if (bytes === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  const value = bytes / Math.pow(1024, i);
  return `${value.toFixed(i === 0 ? 0 : 2)} ${units[i]}`;
}

/**
 * Extract lowercased file extension with leading dot (e.g. '.jpg', '.png')
 */
export function getFileExtension(filename: string): string {
  const lastDot = filename.lastIndexOf('.');
  if (lastDot === -1) return '';
  return filename.slice(lastDot).toLowerCase();
}

/**
 * Validate a candidate file against format, extension, empty-file, and size constraints.
 *
 * Rules:
 * 1. File cannot be null/undefined or 0 bytes (empty/corrupted).
 * 2. File size cannot exceed 10 MB (10,485,760 bytes).
 * 3. File must be a supported image format: JPEG (.jpg, .jpeg), PNG (.png), or WEBP (.webp).
 */
export function validateImageFile(file: File | null | undefined): FileValidationResult {
  if (!file || file.size === 0) {
    return {
      isValid: false,
      error: 'The selected file is empty (0 bytes) or corrupted. Please provide a valid chest X-ray image.',
    };
  }

  // 1. Size Validation (Max 10 MB)
  if (file.size > MAX_FILE_SIZE_BYTES) {
    return {
      isValid: false,
      error: `File size exceeds the 10 MB limit (${formatFileSize(file.size)}). Please select an image under 10 MB.`,
    };
  }

  // 2. Format & MIME Type Validation
  const extension = getFileExtension(file.name);
  const isValidExtension = (SUPPORTED_IMAGE_EXTENSIONS as readonly string[]).includes(extension);
  const isValidMime = (SUPPORTED_IMAGE_MIME_TYPES as readonly string[]).includes(file.type);

  if (!isValidExtension && !isValidMime) {
    const displayExt = extension ? `"${extension}"` : 'without extension';
    return {
      isValid: false,
      error: `Unsupported file format ${displayExt}. Supported formats are JPEG (.jpg, .jpeg), PNG (.png), and WEBP (.webp).`,
    };
  }

  return {
    isValid: true,
    error: null,
  };
}
