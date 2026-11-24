"""Test suite for Grad-CAM Visualization and Heatmap Overlay Service.

Validates:
1. Pure-NumPy JET colormap correctness, numerical bounds, and RGB ordering.
2. Non-destructive alpha blending with original chest radiographs.
3. Dimension preservation across varying spatial resolutions.
4. Input validation (dimensions, formats, alpha ranges).
5. In-memory serialization (PIL, PNG bytes, base64) with zero disk persistence.
"""

from __future__ import annotations

import io
import numpy as np
from PIL import Image
import pytest

from app.services.gradcam_visualizer import (
    GradCAMVisualizer,
    InvalidVisualizerInputError,
    VisualizerResult,
    get_gradcam_visualizer,
)


# ==============================================================================
# Helper Functions
# ==============================================================================


def create_synthetic_image(size: tuple[int, int] = (256, 256), color: tuple[int, int, int] = (100, 150, 200)) -> Image.Image:
    """Create an in-memory synthetic PIL Image."""
    return Image.new("RGB", size, color=color)


def create_synthetic_heatmap(size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Create a deterministic synthetic 2D float32 heatmap in [0.0, 1.0]."""
    h, w = size
    y, x = np.mgrid[0:h, 0:w]
    # Circular Gaussian-like activation map
    center_y, center_x = h / 2.0, w / 2.0
    dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2
    heatmap = np.exp(-dist_sq / (2.0 * (min(h, w) / 4.0) ** 2))
    return heatmap.astype(np.float32)


# ==============================================================================
# 1. Colormap Application Tests
# ==============================================================================


class TestColormapApplication:
    """Verify mathematical correctness of the pure-NumPy JET colormap."""

    def test_jet_colormap_output_shape_and_dtype(self) -> None:
        """Verify JET colormap returns an RGB uint8 array matching input spatial shape."""
        heatmap = create_synthetic_heatmap(size=(100, 120))
        colored = GradCAMVisualizer.apply_jet_colormap(heatmap)

        assert colored.shape == (100, 120, 3)
        assert colored.dtype == np.uint8

    def test_jet_colormap_boundary_values(self) -> None:
        """Verify key JET anchor values: 0.0 (blue), 0.5 (green), 1.0 (red)."""
        test_inputs = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        colored = GradCAMVisualizer.apply_jet_colormap(test_inputs)

        # Value 0.0: Dark Blue (R=0, G=0, B~127)
        assert colored[0, 0, 0] == 0
        assert colored[0, 0, 1] == 0
        assert colored[0, 0, 2] == 127

        # Value 0.5: Green (R~127, G=255, B~127)
        assert colored[0, 1, 1] == 255  # Green peak

        # Value 1.0: Dark Red (R~127, G=0, B=0)
        assert colored[0, 2, 0] == 127
        assert colored[0, 2, 1] == 0
        assert colored[0, 2, 2] == 0

    def test_jet_colormap_clips_out_of_bounds_inputs(self) -> None:
        """Verify negative values and values > 1.0 are safely clamped."""
        out_of_bounds = np.array([[-0.5, 1.5]], dtype=np.float32)
        colored = GradCAMVisualizer.apply_jet_colormap(out_of_bounds)

        # -0.5 clamped to 0.0 -> dark blue
        assert colored[0, 0, 2] == 127
        # 1.5 clamped to 1.0 -> dark red
        assert colored[0, 1, 0] == 127


# ==============================================================================
# 2. Overlay Blending & Dimension Preservation Tests
# ==============================================================================


class TestOverlayBlending:
    """Verify non-destructive blending and dimensional contracts."""

    def test_overlay_dimensions_match_original_image(self) -> None:
        """Verify the overlay matches original image dimensions, not the 224x224 heatmap."""
        visualizer = GradCAMVisualizer()
        orig_img = create_synthetic_image(size=(384, 512))  # (W, H) = 384, 512
        heatmap_224 = create_synthetic_heatmap(size=(224, 224))

        result = visualizer.create_overlay(orig_img, heatmap_224, alpha=0.4)

        assert isinstance(result, VisualizerResult)
        assert result.original_width == 384
        assert result.original_height == 512
        assert result.overlay_array.shape == (512, 384, 3)
        assert result.overlay_array.dtype == np.uint8

    def test_original_image_not_modified(self) -> None:
        """Verify original PIL image and original array are not mutated during blending."""
        visualizer = GradCAMVisualizer()
        orig_img = create_synthetic_image(size=(100, 100), color=(50, 100, 150))
        orig_bytes_before = orig_img.tobytes()

        heatmap = create_synthetic_heatmap(size=(224, 224))
        _ = visualizer.create_overlay(orig_img, heatmap)

        assert orig_img.tobytes() == orig_bytes_before

    def test_strict_rgb_channel_order_maintained(self) -> None:
        """Verify channels are preserved in RGB order without BGR inversions."""
        visualizer = GradCAMVisualizer()
        # Pure red image (255, 0, 0)
        pure_red = np.zeros((50, 50, 3), dtype=np.uint8)
        pure_red[:, :, 0] = 255

        # Flat zero heatmap (clamped to blue in JET: 0, 0, 127)
        zero_heatmap = np.zeros((224, 224), dtype=np.float32)

        # Blend with alpha=0.5:
        # Expected R: 0.5 * 0 + 0.5 * 255 = 127
        # Expected G: 0.5 * 0 + 0.5 * 0 = 0
        # Expected B: 0.5 * 127 + 0.5 * 0 = 63
        result = visualizer.create_overlay(pure_red, zero_heatmap, alpha=0.5)
        overlay = result.overlay_array

        assert np.isclose(overlay[0, 0, 0], 127, atol=2)  # R channel preserved
        assert overlay[0, 0, 1] == 0                       # G channel
        assert np.isclose(overlay[0, 0, 2], 63, atol=2)   # B channel

    def test_input_as_raw_bytes(self) -> None:
        """Verify original image supplied as raw PNG bytes is decoded and blended cleanly."""
        visualizer = GradCAMVisualizer()
        img = create_synthetic_image(size=(150, 150))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw_bytes = buf.getvalue()

        heatmap = create_synthetic_heatmap()
        result = visualizer.create_overlay(raw_bytes, heatmap)

        assert result.original_width == 150
        assert result.original_height == 150
        assert result.overlay_array.shape == (150, 150, 3)

    def test_input_as_grayscale_array(self) -> None:
        """Verify 2D grayscale image array is expanded to 3 channels without errors."""
        visualizer = GradCAMVisualizer()
        gray_arr = np.full((120, 120), 128, dtype=np.uint8)
        heatmap = create_synthetic_heatmap()

        result = visualizer.create_overlay(gray_arr, heatmap)
        assert result.overlay_array.shape == (120, 120, 3)

    def test_alpha_blend_endpoints(self) -> None:
        """Verify alpha=0.0 yields original image and alpha=1.0 yields colored heatmap."""
        visualizer = GradCAMVisualizer()
        orig = np.full((50, 50, 3), 200, dtype=np.uint8)
        heatmap = create_synthetic_heatmap(size=(50, 50))

        # Alpha = 0.0 -> exact original
        res_zero = visualizer.create_overlay(orig, heatmap, alpha=0.0)
        assert np.all(res_zero.overlay_array == 200)

        # Alpha = 1.0 -> exact colored heatmap
        res_one = visualizer.create_overlay(orig, heatmap, alpha=1.0)
        assert np.all(res_one.overlay_array == res_one.colored_heatmap)


# ==============================================================================
# 3. Defensive Validation & Error Handling Tests
# ==============================================================================


class TestDefensiveValidation:
    """Verify invalid inputs, NaN values, and out-of-range parameters are rejected."""

    def test_invalid_alpha_raises_error(self) -> None:
        """Verify alpha < 0.0 or alpha > 1.0 raises InvalidVisualizerInputError."""
        visualizer = GradCAMVisualizer()
        img = create_synthetic_image()
        heatmap = create_synthetic_heatmap()

        with pytest.raises(InvalidVisualizerInputError) as exc_info:
            visualizer.create_overlay(img, heatmap, alpha=1.5)
        assert "Alpha blend ratio" in str(exc_info.value)

        with pytest.raises(InvalidVisualizerInputError):
            visualizer.create_overlay(img, heatmap, alpha=-0.1)

    def test_unsupported_colormap_raises_error(self) -> None:
        """Verify requesting unknown colormap raises InvalidVisualizerInputError."""
        visualizer = GradCAMVisualizer()
        img = create_synthetic_image()
        heatmap = create_synthetic_heatmap()

        with pytest.raises(InvalidVisualizerInputError) as exc_info:
            visualizer.create_overlay(img, heatmap, colormap="viridis")
        assert "Unsupported colormap" in str(exc_info.value)

    def test_none_inputs_rejected(self) -> None:
        """Verify None image or None heatmap raises InvalidVisualizerInputError."""
        visualizer = GradCAMVisualizer()
        heatmap = create_synthetic_heatmap()

        with pytest.raises(InvalidVisualizerInputError):
            visualizer.create_overlay(None, heatmap)  # type: ignore

        with pytest.raises(InvalidVisualizerInputError):
            visualizer.create_overlay(create_synthetic_image(), None)  # type: ignore

    def test_non_2d_heatmap_rejected(self) -> None:
        """Verify 1D or 3D heatmap array raises InvalidVisualizerInputError."""
        visualizer = GradCAMVisualizer()
        img = create_synthetic_image()

        with pytest.raises(InvalidVisualizerInputError):
            visualizer.create_overlay(img, np.zeros((224, 224, 1), dtype=np.float32))

        with pytest.raises(InvalidVisualizerInputError):
            visualizer.create_overlay(img, np.zeros((100,), dtype=np.float32))

    def test_nan_heatmap_rejected(self) -> None:
        """Verify heatmap containing NaN raises InvalidVisualizerInputError."""
        visualizer = GradCAMVisualizer()
        img = create_synthetic_image()
        nan_heatmap = create_synthetic_heatmap()
        nan_heatmap[10, 10] = np.nan

        with pytest.raises(InvalidVisualizerInputError) as exc_info:
            visualizer.create_overlay(img, nan_heatmap)
        assert "non-finite" in str(exc_info.value)


# ==============================================================================
# 4. Serialization & Memory Safety Tests
# ==============================================================================


class TestSerializationAndMemorySafety:
    """Verify in-memory serialization without disk I/O."""

    def test_to_pil_and_to_png_bytes(self) -> None:
        """Verify overlay can be converted to PIL Image and valid PNG bytes in-memory."""
        visualizer = GradCAMVisualizer()
        img = create_synthetic_image(size=(120, 160))
        heatmap = create_synthetic_heatmap()

        result = visualizer.create_overlay(img, heatmap)

        pil_img = result.to_pil()
        assert isinstance(pil_img, Image.Image)
        assert pil_img.size == (120, 160)
        assert pil_img.mode == "RGB"

        png_bytes = result.to_png_bytes()
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0
        assert png_bytes.startswith(b"\x89PNG")

    def test_to_base64_data_uri(self) -> None:
        """Verify base64 URI conforms to data:image/png;base64 specification."""
        visualizer = GradCAMVisualizer()
        img = create_synthetic_image(size=(50, 50))
        heatmap = create_synthetic_heatmap()

        result = visualizer.create_overlay(img, heatmap)
        uri = result.to_base64_data_uri()

        assert uri.startswith("data:image/png;base64,")
        assert len(uri) > len("data:image/png;base64,")

    def test_heatmap_to_pil_and_to_png_bytes(self) -> None:
        """Verify colored heatmap serialization to PIL and PNG bytes."""
        visualizer = GradCAMVisualizer()
        img = create_synthetic_image(size=(80, 100))
        heatmap = create_synthetic_heatmap()

        result = visualizer.create_overlay(img, heatmap)

        heatmap_pil = result.heatmap_to_pil()
        assert isinstance(heatmap_pil, Image.Image)
        assert heatmap_pil.size == (80, 100)
        assert heatmap_pil.mode == "RGB"

        heatmap_png = result.heatmap_to_png_bytes()
        assert isinstance(heatmap_png, bytes)
        assert heatmap_png.startswith(b"\x89PNG")

    def test_heatmap_to_base64_data_uri(self) -> None:
        """Verify colored heatmap base64 data URI generation."""
        visualizer = GradCAMVisualizer()
        img = create_synthetic_image(size=(60, 60))
        heatmap = create_synthetic_heatmap()

        result = visualizer.create_overlay(img, heatmap)
        uri = result.heatmap_to_base64_data_uri()

        assert uri.startswith("data:image/png;base64,")
        assert len(uri) > len("data:image/png;base64,")

    def test_singleton_accessor(self) -> None:
        """Verify get_gradcam_visualizer returns persistent application singleton."""
        v1 = get_gradcam_visualizer()
        v2 = get_gradcam_visualizer()
        assert v1 is v2

