"""Chest X-ray Image Preprocessing and Input Validation Service.

Provides a production-grade, defensive, and deterministic image preprocessing
pipeline tailored for the DenseNet121 chest X-ray deep learning classifier.

Pipeline Stages:
1. Byte-level defensive validation (empty checks, payload size boundaries).
2. Format verification and safe Pillow decoding with verify/reopen isolation.
3. Dimensional limits and decompression-bomb mitigation.
4. EXIF orientation detection and lossless transposition.
5. Color channel normalization to RGB with deterministic white-background alpha compositing.
6. High-quality isotropic resizing to 224x224 using Lanczos resampling.
7. NumPy array conversion and dtype enforcement.
8. Official TensorFlow DenseNet121 preprocess_input tensor preprocessing.
9. Batch dimension expansion to (1, 224, 224, 3).
10. Final tensor contract validation (shape, dtype, finite values).
"""

from dataclasses import dataclass
import io
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger("lungcare.image_preprocessor")


# ==============================================================================
# Domain Exceptions
# ==============================================================================


class ImagePreprocessingError(Exception):
    """Base exception for all image preprocessing and validation failures."""

    def __init__(
        self,
        message: str,
        error_code: str = "IMAGE_PREPROCESSING_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class EmptyImageError(ImagePreprocessingError):
    """Raised when the uploaded image byte sequence is empty or zero-length."""

    def __init__(
        self,
        message: str = "Uploaded image byte sequence is empty.",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, error_code="EMPTY_IMAGE", details=details)


class ImageTooLargeError(ImagePreprocessingError):
    """Raised when the image file size in bytes exceeds the configured maximum upload limit."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, error_code="IMAGE_TOO_LARGE", details=details)


class UnsupportedFormatError(ImagePreprocessingError):
    """Raised when the decoded image format is not among the allowed formats (JPEG, PNG, WEBP)."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, error_code="UNSUPPORTED_FORMAT", details=details)


class CorruptImageError(ImagePreprocessingError):
    """Raised when image bytes cannot be identified, decoded, or are corrupted."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, error_code="CORRUPT_IMAGE", details=details)


class DecompressionBombError(ImagePreprocessingError):
    """Raised when image pixel dimensions exceed safe limits, preventing memory exhaustion."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, error_code="DECOMPRESSION_BOMB", details=details)


class InvalidImageModeError(ImagePreprocessingError):
    """Raised when an image color mode cannot be safely normalized to RGB."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, error_code="INVALID_IMAGE_MODE", details=details)


class TensorValidationError(ImagePreprocessingError):
    """Raised when the preprocessed output tensor violates shape, dtype, or finiteness contracts."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, error_code="TENSOR_VALIDATION_ERROR", details=details)


# ==============================================================================
# Configuration & Return Contracts
# ==============================================================================


@dataclass(frozen=True)
class ImagePreprocessorConfig:
    """Configuration options for the chest X-ray image preprocessing pipeline."""

    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MB default
    max_image_pixels: int = 25_000_000        # 25 Megapixels (e.g. 5000x5000)
    allowed_formats: Tuple[str, ...] = ("JPEG", "PNG", "WEBP")
    target_size: Tuple[int, int] = (224, 224)
    target_color_mode: str = "RGB"

    @classmethod
    def from_settings(cls) -> "ImagePreprocessorConfig":
        """Instantiate configuration from application settings if available."""
        try:
            from app.core.config import get_settings

            settings = get_settings()
            return cls(
                max_upload_bytes=getattr(settings, "MAX_IMAGE_UPLOAD_BYTES", 10 * 1024 * 1024),
                max_image_pixels=getattr(settings, "MAX_IMAGE_PIXELS", 25_000_000),
                allowed_formats=tuple(getattr(settings, "ALLOWED_IMAGE_FORMATS", ["JPEG", "PNG", "WEBP"])),
                target_size=(
                    getattr(settings, "IMAGE_TARGET_SIZE", 224),
                    getattr(settings, "IMAGE_TARGET_SIZE", 224),
                ),
            )
        except Exception:
            return cls()


@dataclass(frozen=True)
class PreprocessedImageResult:
    """Immutable structured result produced by the image preprocessing service."""

    tensor: np.ndarray
    original_width: int
    original_height: int
    processed_width: int
    processed_height: int
    original_format: str
    original_mode: str
    final_mode: str
    exif_transposed: bool
    alpha_composited: bool

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary of metadata (excluding the raw tensor)."""
        return {
            "original_width": self.original_width,
            "original_height": self.original_height,
            "processed_width": self.processed_width,
            "processed_height": self.processed_height,
            "original_format": self.original_format,
            "original_mode": self.original_mode,
            "final_mode": self.final_mode,
            "exif_transposed": self.exif_transposed,
            "alpha_composited": self.alpha_composited,
            "tensor_shape": list(self.tensor.shape),
            "tensor_dtype": str(self.tensor.dtype),
        }


# ==============================================================================
# Service Implementation
# ==============================================================================


class ImagePreprocessor:
    """Production-grade Chest X-ray Image Preprocessing & Input Validation Pipeline."""

    def __init__(self, config: Optional[ImagePreprocessorConfig] = None) -> None:
        self.config = config or ImagePreprocessorConfig()

    def preprocess(self, image_bytes: bytes) -> PreprocessedImageResult:
        """Execute the full validation, transformation, and DenseNet preprocessing pipeline.

        Args:
            image_bytes: Raw binary image payload.

        Returns:
            PreprocessedImageResult containing the (1, 224, 224, 3) float32 tensor
            and associated transformation metadata.

        Raises:
            EmptyImageError: If byte sequence is empty.
            ImageTooLargeError: If bytes exceed max_upload_bytes.
            UnsupportedFormatError: If image format is not in allowed_formats.
            CorruptImageError: If byte stream cannot be decoded by PIL.
            DecompressionBombError: If pixel count exceeds max_image_pixels.
            InvalidImageModeError: If color mode cannot be converted safely to RGB.
            TensorValidationError: If final tensor violates shape, dtype, or finiteness contracts.
        """
        # Step 1: Byte-level defensive size and type validation
        self._validate_byte_size(image_bytes)

        # Step 2: Safe PIL open, header verification, and fresh reopening
        image, detected_format = self._safe_open_and_verify(image_bytes)

        # Step 3: Dimensional limits and decompression-bomb mitigation
        self._validate_pixel_dimensions(image)

        original_width, original_height = image.size
        original_mode = image.mode

        # Step 4: EXIF orientation detection and correction
        image, exif_applied = self._apply_exif_orientation(image)

        # Step 5: Color normalization to RGB with deterministic white-background alpha compositing
        image, alpha_composited = self._normalize_color_to_rgb(image)

        # Step 6: High-quality isotropic Lanczos resizing to target dimensions
        resized_image = self._resize_image(image)

        # Step 7, 8, 9: Array conversion, TensorFlow DenseNet preprocess_input, and batch dimension
        tensor = self._preprocess_tensor(resized_image)

        # Step 10: Contract validation on output tensor
        self._validate_tensor_contract(tensor)

        logger.debug(
            "Image preprocessing succeeded: %s (%dx%d, %s) -> RGB (%dx%d, tensor %s, EXIF=%s, alpha=%s)",
            detected_format,
            original_width,
            original_height,
            original_mode,
            self.config.target_size[0],
            self.config.target_size[1],
            tensor.shape,
            exif_applied,
            alpha_composited,
        )

        return PreprocessedImageResult(
            tensor=tensor,
            original_width=original_width,
            original_height=original_height,
            processed_width=self.config.target_size[0],
            processed_height=self.config.target_size[1],
            original_format=detected_format,
            original_mode=original_mode,
            final_mode=self.config.target_color_mode,
            exif_transposed=exif_applied,
            alpha_composited=alpha_composited,
        )

    # --------------------------------------------------------------------------
    # Internal Pipeline Steps
    # --------------------------------------------------------------------------

    def _validate_byte_size(self, image_bytes: bytes) -> None:
        """Validate byte sequence integrity and enforce payload size limits."""
        if not isinstance(image_bytes, (bytes, bytearray)):
            raise CorruptImageError(
                f"Input payload must be bytes or bytearray, got '{type(image_bytes).__name__}'.",
                details={"type": type(image_bytes).__name__},
            )

        byte_len = len(image_bytes)
        if byte_len == 0:
            raise EmptyImageError(
                "Uploaded image byte sequence is empty.",
                details={"size_bytes": 0},
            )

        if byte_len > self.config.max_upload_bytes:
            raise ImageTooLargeError(
                f"Image payload size ({byte_len:,} bytes) exceeds the maximum allowed limit of "
                f"{self.config.max_upload_bytes:,} bytes ({self.config.max_upload_bytes / (1024 * 1024):.1f} MB).",
                details={"size_bytes": byte_len, "max_upload_bytes": self.config.max_upload_bytes},
            )

    def _safe_open_and_verify(self, image_bytes: bytes) -> Tuple[Image.Image, str]:
        """Verify image headers safely, enforce allowed formats, and reopen clean instance."""
        # Detect XML/SVG early before Pillow parser attempts
        prefix = image_bytes[:1024].lstrip().lower()
        if prefix.startswith(b"<svg") or (b"<xml" in prefix and b"<svg" in prefix):
            raise UnsupportedFormatError(
                "Unsupported image format 'SVG'. Only raster image formats (JPEG, PNG, WEBP) are supported.",
                details={"detected_format": "SVG", "allowed_formats": list(self.config.allowed_formats)},
            )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(image_bytes)) as verify_img:
                    verify_img.verify()
                    detected_format = verify_img.format

        except Image.DecompressionBombError as exc:
            raise DecompressionBombError(
                f"Image decompression bomb detected: {exc}",
                details={"error": str(exc)},
            ) from exc
        except Image.DecompressionBombWarning as exc:
            raise DecompressionBombError(
                f"Image decompression bomb warning triggered: {exc}",
                details={"warning": str(exc)},
            ) from exc
        except (UnidentifiedImageError, OSError, SyntaxError) as exc:
            raise CorruptImageError(
                f"Failed to decode or verify image bytes: {exc}",
                details={"error": str(exc)},
            ) from exc
        except Exception as exc:
            raise CorruptImageError(
                f"Unexpected error validating image headers: {exc}",
                details={"error": str(exc)},
            ) from exc

        if not detected_format:
            raise UnsupportedFormatError(
                "Could not determine image format from byte payload.",
                details={"detected_format": None, "allowed_formats": list(self.config.allowed_formats)},
            )

        detected_format = detected_format.upper()
        if detected_format == "JPG":
            detected_format = "JPEG"

        if detected_format not in self.config.allowed_formats:
            raise UnsupportedFormatError(
                f"Unsupported image format '{detected_format}'. Supported formats: {list(self.config.allowed_formats)}.",
                details={"detected_format": detected_format, "allowed_formats": list(self.config.allowed_formats)},
            )

        # Reopen image from a fresh stream because verify() closes the stream and invalidates image data
        try:
            opened_img = Image.open(io.BytesIO(image_bytes))
            opened_img.load()
            return opened_img, detected_format
        except Exception as exc:
            raise CorruptImageError(
                f"Failed to load verified image data into memory: {exc}",
                details={"error": str(exc)},
            ) from exc

    def _validate_pixel_dimensions(self, image: Image.Image) -> None:
        """Enforce maximum pixel count to mitigate decompression-bomb risks."""
        width, height = image.size
        if width <= 0 or height <= 0:
            raise CorruptImageError(
                f"Invalid image dimensions: {width}x{height}.",
                details={"width": width, "height": height},
            )

        total_pixels = width * height
        if total_pixels > self.config.max_image_pixels:
            raise DecompressionBombError(
                f"Image pixel count ({total_pixels:,}) exceeds the maximum allowed limit of "
                f"{self.config.max_image_pixels:,} pixels ({width}x{height}).",
                details={
                    "pixel_count": total_pixels,
                    "max_pixels": self.config.max_image_pixels,
                    "width": width,
                    "height": height,
                },
            )

    def _apply_exif_orientation(self, image: Image.Image) -> Tuple[Image.Image, bool]:
        """Detect and correct EXIF orientation metadata via lossless transposition."""
        exif_applied = False
        try:
            exif = image.getexif()
            orientation = exif.get(0x0112) if exif else None
            # Standard EXIF orientation: 1=Normal, 2-8=Rotated/Mirrored
            if orientation is not None and orientation in (2, 3, 4, 5, 6, 7, 8):
                transposed = ImageOps.exif_transpose(image)
                if transposed is not None:
                    image = transposed
                    exif_applied = True
            else:
                transposed = ImageOps.exif_transpose(image)
                if transposed is not None:
                    image = transposed
        except Exception as exc:
            logger.warning("Failed to evaluate EXIF orientation: %s. Continuing with original orientation.", exc)

        return image, exif_applied

    def _normalize_color_to_rgb(self, image: Image.Image) -> Tuple[Image.Image, bool]:
        """Convert any supported image color mode into standard 3-channel RGB.

        Transparent images (RGBA, LA, or palette with alpha) are composited onto a
        deterministic solid white (255, 255, 255) background to eliminate black borders
        or non-deterministic transparency artifacts in radiographic interpretation.
        """
        alpha_composited = False
        mode = image.mode

        if mode == "RGB":
            return image, False

        # Handle modes containing alpha transparency
        if mode in ("RGBA", "LA") or (mode == "P" and ("transparency" in image.info or "A" in image.mode)):
            try:
                rgba_img = image.convert("RGBA")
                white_bg = Image.new("RGBA", rgba_img.size, (255, 255, 255, 255))
                composited = Image.alpha_composite(white_bg, rgba_img)
                rgb_img = composited.convert("RGB")
                alpha_composited = True
                return rgb_img, alpha_composited
            except Exception as exc:
                raise InvalidImageModeError(
                    f"Failed to composite transparent image mode '{mode}' on white background: {exc}",
                    details={"mode": mode, "error": str(exc)},
                ) from exc

        # Handle non-transparent modes convertible to RGB
        elif mode in ("L", "P", "CMYK", "1", "YCbCr", "LAB", "HSV"):
            try:
                rgb_img = image.convert("RGB")
                return rgb_img, False
            except Exception as exc:
                raise InvalidImageModeError(
                    f"Failed to convert image mode '{mode}' to RGB: {exc}",
                    details={"mode": mode, "error": str(exc)},
                ) from exc

        else:
            try:
                rgb_img = image.convert("RGB")
                if rgb_img.mode != "RGB" or len(rgb_img.getbands()) != 3:
                    raise ValueError(f"Converted mode '{rgb_img.mode}' does not have 3 RGB bands.")
                return rgb_img, False
            except Exception as exc:
                raise InvalidImageModeError(
                    f"Unsupported or unconvertible image mode '{mode}': {exc}",
                    details={"mode": mode, "error": str(exc)},
                ) from exc

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image to target dimensions (224x224) using high-quality Lanczos resampling.

        IMPORTANT ARCHITECTURAL NOTE ON ASPECT RATIOS:
        Direct isotropic resizing scales width and height to 224x224 independently, which
        will distort non-square aspect ratios (e.g. 1.2:1 standard chest radiographs).
        This behavior is intentionally preserved to maintain strict mathematical consistency
        with the DenseNet121 model training pipeline (which used direct resizing without
        letterboxing, zero-padding, or center-cropping). Altering this transformation
        strategy would change the spatial frequency distribution and degrade model accuracy.
        """
        if image.size == self.config.target_size:
            return image

        return image.resize(self.config.target_size, resample=Image.Resampling.LANCZOS)

    def _preprocess_tensor(self, image: Image.Image) -> np.ndarray:
        """Convert resized RGB PIL image to NumPy array and apply DenseNet preprocessing."""
        # 1. Convert to uint8 NumPy array
        img_array = np.asarray(image, dtype=np.uint8)

        expected_shape = (self.config.target_size[0], self.config.target_size[1], 3)
        if img_array.shape != expected_shape:
            raise TensorValidationError(
                f"Image array shape {img_array.shape} does not match expected {expected_shape}.",
                details={"actual_shape": img_array.shape, "expected_shape": expected_shape},
            )

        # 2. Cast to float32
        img_float = img_array.astype(np.float32)

        # 3. Scoped import of official TensorFlow/Keras DenseNet preprocess_input
        # Isolates host API startup from heavy TensorFlow imports when not preprocessing
        from tensorflow.keras.applications.densenet import preprocess_input

        # 4. Apply TensorFlow DenseNet preprocessing (tf.keras.applications.densenet.preprocess_input)
        preprocessed = preprocess_input(img_float)

        # 5. Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
        batch_tensor = np.expand_dims(preprocessed, axis=0)
        return batch_tensor

    def _validate_tensor_contract(self, tensor: np.ndarray) -> None:
        """Verify output tensor satisfies model input dimensions, dtype, and finiteness contracts."""
        expected_shape = (1, self.config.target_size[0], self.config.target_size[1], 3)
        if tensor.shape != expected_shape:
            raise TensorValidationError(
                f"Tensor shape {tensor.shape} violates model contract {expected_shape}.",
                details={"actual_shape": tensor.shape, "expected_shape": expected_shape},
            )

        if tensor.dtype != np.float32:
            raise TensorValidationError(
                f"Tensor dtype '{tensor.dtype}' violates model contract 'float32'.",
                details={"actual_dtype": str(tensor.dtype), "expected_dtype": "float32"},
            )

        if not np.all(np.isfinite(tensor)):
            has_nan = bool(np.any(np.isnan(tensor)))
            has_inf = bool(np.any(np.isinf(tensor)))
            raise TensorValidationError(
                f"Tensor contains non-finite values (has_nan={has_nan}, has_inf={has_inf}).",
                details={"has_nan": has_nan, "has_inf": has_inf},
            )


# ==============================================================================
# Thread-Safe Singleton Accessor
# ==============================================================================

_preprocessor_instance: Optional[ImagePreprocessor] = None
_preprocessor_lock = threading.Lock()


def get_image_preprocessor(config: Optional[ImagePreprocessorConfig] = None) -> ImagePreprocessor:
    """Return a thread-safe singleton instance of ImagePreprocessor."""
    global _preprocessor_instance
    if _preprocessor_instance is None or config is not None:
        with _preprocessor_lock:
            if _preprocessor_instance is None or config is not None:
                _preprocessor_instance = ImagePreprocessor(config=config)
    return _preprocessor_instance

