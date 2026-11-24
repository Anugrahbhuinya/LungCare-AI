"""Grad-CAM Visualization and Heatmap Overlay Service for LungCare AI.

Generates high-fidelity visual overlays blending normalized Grad-CAM heatmaps
with original chest radiographs using pure Pillow and NumPy (zero extra dependencies).
Guarantees channel integrity (strict RGB order), non-destructive image handling,
and zero disk persistence.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import io
import logging
import threading
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image

logger = logging.getLogger("lungcare.ml.visualizer")


# ==============================================================================
# Domain Exceptions
# ==============================================================================


class VisualizerError(Exception):
    """Base exception for all Grad-CAM visualization and overlay failures."""

    def __init__(
        self,
        message: str,
        error_code: str = "VISUALIZER_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class InvalidVisualizerInputError(VisualizerError):
    """Raised when visualizer input image or heatmap violates contract specifications."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message=message, error_code="INVALID_VISUALIZER_INPUT", details=details)


# ==============================================================================
# Structured Visualizer Result
# ==============================================================================


@dataclass(frozen=True)
class VisualizerResult:
    """Immutable result containing the blended overlay and serialization helpers."""

    overlay_array: np.ndarray  # Shape: (H, W, 3), dtype: uint8, RGB order
    colored_heatmap: np.ndarray  # Shape: (H, W, 3), dtype: uint8, RGB order
    original_width: int
    original_height: int
    alpha: float
    colormap: str

    def to_pil(self) -> Image.Image:
        """Convert blended overlay NumPy array to PIL Image in RGB mode."""
        return Image.fromarray(self.overlay_array, mode="RGB")

    def to_png_bytes(self) -> bytes:
        """Serialize blended overlay to in-memory PNG bytes."""
        buf = io.BytesIO()
        pil_img = self.to_pil()
        pil_img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    def to_base64_data_uri(self) -> str:
        """Encode blended overlay to an RFC 2397 base64 data URI (image/png)."""
        raw_bytes = self.to_png_bytes()
        encoded = base64.b64encode(raw_bytes).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def heatmap_to_pil(self) -> Image.Image:
        """Convert colored heatmap NumPy array to PIL Image in RGB mode."""
        return Image.fromarray(self.colored_heatmap, mode="RGB")

    def heatmap_to_png_bytes(self) -> bytes:
        """Serialize colored heatmap to in-memory PNG bytes."""
        buf = io.BytesIO()
        pil_img = self.heatmap_to_pil()
        pil_img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    def heatmap_to_base64_data_uri(self) -> str:
        """Encode colored heatmap to an RFC 2397 base64 data URI (image/png)."""
        raw_bytes = self.heatmap_to_png_bytes()
        encoded = base64.b64encode(raw_bytes).decode("ascii")
        return f"data:image/png;base64,{encoded}"


# ==============================================================================
# Visualizer Service
# ==============================================================================


class GradCAMVisualizer:
    """Production-grade heatmap colorizer and alpha-blending service."""

    def __init__(self, default_alpha: float = 0.4, default_colormap: str = "jet") -> None:
        """Initialize visualizer with default blending parameters.

        Args:
            default_alpha: Heatmap blend weight in range [0.0, 1.0].
            default_colormap: Colormap identifier (currently supports 'jet').
        """
        self.default_alpha = default_alpha
        self.default_colormap = default_colormap

    @staticmethod
    def apply_jet_colormap(heatmap: np.ndarray) -> np.ndarray:
        """Convert a 2D float32 heatmap in [0.0, 1.0] to an RGB uint8 array using the JET colormap.

        Uses the standard vectorized 4-interval piece-wise linear JET formulation:
            R(v) = clip(1.5 - |4v - 3|, 0, 1)
            G(v) = clip(1.5 - |4v - 2|, 0, 1)
            B(v) = clip(1.5 - |4v - 1|, 0, 1)

        Args:
            heatmap: 2D array of arbitrary spatial dimensions, normalized to [0.0, 1.0].

        Returns:
            RGB uint8 array of shape (H, W, 3).
        """
        v = np.clip(heatmap, 0.0, 1.0)
        r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)

        rgb = np.stack([r, g, b], axis=-1)
        return (rgb * 255.0).astype(np.uint8)

    def create_overlay(
        self,
        original_image: Union[Image.Image, np.ndarray, bytes],
        heatmap: np.ndarray,
        alpha: Optional[float] = None,
        colormap: Optional[str] = None,
    ) -> VisualizerResult:
        """Blend a normalized Grad-CAM heatmap over an original radiograph.

        Args:
            original_image: Input chest X-ray as PIL Image, RGB uint8 array, or binary bytes.
            heatmap: 2D float32 heatmap array with values in [0.0, 1.0].
            alpha: Optional blend ratio in [0.0, 1.0]. Defaults to self.default_alpha.
            colormap: Optional colormap name. Defaults to self.default_colormap.

        Returns:
            VisualizerResult with the blended RGB array and image serialization methods.

        Raises:
            InvalidVisualizerInputError: If inputs violate dimensional or numerical contracts.
        """
        blend_alpha = self.default_alpha if alpha is None else float(alpha)
        if not (0.0 <= blend_alpha <= 1.0):
            raise InvalidVisualizerInputError(
                f"Alpha blend ratio must be in range [0.0, 1.0], got {blend_alpha}.",
                details={"alpha": blend_alpha},
            )

        cmap_name = (colormap or self.default_colormap).lower()
        if cmap_name != "jet":
            raise InvalidVisualizerInputError(
                f"Unsupported colormap '{cmap_name}'. Supported colormaps: ['jet'].",
                details={"colormap": cmap_name, "supported": ["jet"]},
            )

        # Step 1: Validate and normalize the original image (non-destructive)
        orig_rgb, orig_w, orig_h = self._normalize_original_image(original_image)

        # Step 2: Validate heatmap
        self._validate_heatmap(heatmap)

        # Step 3: Resize heatmap to original image dimensions via bilinear interpolation
        # Convert float heatmap to 8-bit representation for PIL resampling
        heatmap_clipped = np.clip(heatmap, 0.0, 1.0)
        heatmap_pil = Image.fromarray((heatmap_clipped * 255.0).astype(np.uint8), mode="L")
        heatmap_resized_pil = heatmap_pil.resize((orig_w, orig_h), resample=Image.Resampling.BILINEAR)
        heatmap_resized = np.asarray(heatmap_resized_pil, dtype=np.float32) / 255.0

        # Step 4: Apply Colormap
        colored_heatmap = self.apply_jet_colormap(heatmap_resized)

        # Step 5: Alpha Blending in RGB Space
        # Formula: blended = alpha * heatmap + (1 - alpha) * original
        blended = (
            blend_alpha * colored_heatmap.astype(np.float32)
            + (1.0 - blend_alpha) * orig_rgb.astype(np.float32)
        )
        overlay_array = np.clip(blended, 0.0, 255.0).astype(np.uint8)

        # Verify output contract
        if overlay_array.shape != (orig_h, orig_w, 3):
            raise VisualizerError(
                f"Overlay dimensions mismatch: expected ({orig_h}, {orig_w}, 3), got {overlay_array.shape}"
            )

        return VisualizerResult(
            overlay_array=overlay_array,
            colored_heatmap=colored_heatmap,
            original_width=orig_w,
            original_height=orig_h,
            alpha=blend_alpha,
            colormap=cmap_name,
        )

    def _normalize_original_image(
        self, image_input: Union[Image.Image, np.ndarray, bytes]
    ) -> Tuple[np.ndarray, int, int]:
        """Convert input image into a standardized RGB uint8 NumPy array without mutating source."""
        if image_input is None:
            raise InvalidVisualizerInputError("Original image input cannot be None.")

        if isinstance(image_input, bytes):
            if len(image_input) == 0:
                raise InvalidVisualizerInputError("Original image byte sequence is empty.")
            try:
                with Image.open(io.BytesIO(image_input)) as img:
                    img_rgb = img.convert("RGB")
                    arr = np.asarray(img_rgb, dtype=np.uint8)
                    return arr, img_rgb.width, img_rgb.height
            except Exception as exc:
                raise InvalidVisualizerInputError(
                    f"Failed to decode original image bytes: {exc}",
                    details={"error": str(exc)},
                ) from exc

        if isinstance(image_input, Image.Image):
            img_rgb = image_input.convert("RGB")
            arr = np.asarray(img_rgb, dtype=np.uint8)
            return arr, img_rgb.width, img_rgb.height

        if isinstance(image_input, np.ndarray):
            arr = image_input.copy()
            if arr.ndim == 2:
                # Grayscale (H, W) -> expand to (H, W, 3)
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.ndim == 3 and arr.shape[2] == 1:
                arr = np.concatenate([arr] * 3, axis=-1)
            elif arr.ndim == 3 and arr.shape[2] == 4:
                # RGBA -> drop alpha channel
                arr = arr[:, :, :3]
            elif arr.ndim != 3 or arr.shape[2] != 3:
                raise InvalidVisualizerInputError(
                    f"Invalid NumPy image array shape {arr.shape}. Expected (H, W, 3) or (H, W).",
                    details={"shape": list(arr.shape)},
                )

            # Ensure uint8
            if np.issubdtype(arr.dtype, np.floating):
                # If in [0, 1] range, scale to 255
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)

            h, w = arr.shape[0], arr.shape[1]
            return arr, w, h

        raise InvalidVisualizerInputError(
            f"Unsupported image type '{type(image_input).__name__}'. "
            "Expected PIL.Image.Image, np.ndarray, or raw bytes."
        )

    def _validate_heatmap(self, heatmap: Any) -> None:
        """Validate heatmap array dimensions, finiteness, and numerical bounds."""
        if heatmap is None:
            raise InvalidVisualizerInputError("Heatmap cannot be None.")

        if not isinstance(heatmap, np.ndarray):
            raise InvalidVisualizerInputError(
                f"Heatmap must be a NumPy array, got '{type(heatmap).__name__}'."
            )

        if heatmap.ndim != 2:
            raise InvalidVisualizerInputError(
                f"Heatmap must be 2-dimensional (H, W), got shape {heatmap.shape}."
            )

        h, w = heatmap.shape
        if h <= 0 or w <= 0:
            raise InvalidVisualizerInputError(f"Invalid heatmap dimensions: {w}x{h}.")

        if not np.all(np.isfinite(heatmap)):
            raise InvalidVisualizerInputError("Heatmap contains non-finite values (NaN or Inf).")

        h_min = float(np.min(heatmap))
        h_max = float(np.max(heatmap))
        if h_min < -1e-5 or h_max > 1.0 + 1e-5:
            logger.warning("Heatmap values [%.4f, %.4f] outside [0, 1] range; clipping.", h_min, h_max)


# ==============================================================================
# Thread-Safe Singleton Accessor
# ==============================================================================

_visualizer_instance: Optional[GradCAMVisualizer] = None
_visualizer_lock = threading.Lock()


def get_gradcam_visualizer(
    default_alpha: float = 0.4, default_colormap: str = "jet"
) -> GradCAMVisualizer:
    """Return the application-wide singleton instance of GradCAMVisualizer."""
    global _visualizer_instance
    if _visualizer_instance is None:
        with _visualizer_lock:
            if _visualizer_instance is None:
                _visualizer_instance = GradCAMVisualizer(
                    default_alpha=default_alpha,
                    default_colormap=default_colormap,
                )
    return _visualizer_instance
