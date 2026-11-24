"""Comprehensive unit and contract test suite for Chest X-ray Image Preprocessor.

Validates:
1. Input format validation (JPEG, PNG, WEBP supported; BMP, TIFF, GIF, SVG rejected).
2. Defensive security guardrails (empty uploads, max size limits, decompression bomb).
3. Image transformations (exact 224x224, LANCZOS, EXIF orientation, RGB normalization).
4. Deterministic alpha compositing on white background for transparent images.
5. Exact numerical accuracy of TensorFlow DenseNet121 preprocess_input tensor preprocessing.
6. Deterministic reproducibility across repeated invocations.
"""

from __future__ import annotations

import io
import numpy as np
from PIL import Image, ImageOps
import pytest

from app.services.image_preprocessor import (
    CorruptImageError,
    DecompressionBombError,
    EmptyImageError,
    ImagePreprocessingError,
    ImagePreprocessor,
    ImagePreprocessorConfig,
    ImageTooLargeError,
    InvalidImageModeError,
    PreprocessedImageResult,
    TensorValidationError,
    UnsupportedFormatError,
    get_image_preprocessor,
)


# ==============================================================================
# Helper Fixtures & Image Generators
# ==============================================================================


def create_test_image_bytes(
    format_name: str = "JPEG",
    size: tuple[int, int] = (300, 300),
    mode: str = "RGB",
    color: tuple = (120, 150, 180),
    exif_orientation: int | None = None,
) -> bytes:
    """Generate in-memory synthetic image bytes for testing."""
    img = Image.new(mode, size, color=color)

    buf = io.BytesIO()
    save_kwargs: dict = {}

    if exif_orientation is not None and format_name in ("JPEG", "TIFF"):
        exif = img.getexif()
        exif[0x0112] = exif_orientation
        save_kwargs["exif"] = exif

    if format_name == "JPEG" and mode == "RGBA":
        # JPEG doesn't support alpha, convert to RGB for saving test
        img = img.convert("RGB")

    img.save(buf, format=format_name, **save_kwargs)
    return buf.getvalue()


@pytest.fixture
def preprocessor() -> ImagePreprocessor:
    """Return a standard ImagePreprocessor instance with default configuration."""
    return ImagePreprocessor()


# ==============================================================================
# 1. Valid Image Formats & Modes
# ==============================================================================


class TestValidImageInputs:
    """Verify accepted image formats and color modes are handled safely."""

    def test_valid_rgb_jpeg(self, preprocessor: ImagePreprocessor) -> None:
        """Verify RGB JPEG is accepted, normalized, and preprocessed."""
        image_bytes = create_test_image_bytes(format_name="JPEG", size=(320, 240), mode="RGB")
        result = preprocessor.preprocess(image_bytes)

        assert isinstance(result, PreprocessedImageResult)
        assert result.original_format == "JPEG"
        assert result.original_mode == "RGB"
        assert result.final_mode == "RGB"
        assert result.original_width == 320
        assert result.original_height == 240
        assert result.processed_width == 224
        assert result.processed_height == 224
        assert result.tensor.shape == (1, 224, 224, 3)
        assert result.tensor.dtype == np.float32
        assert result.exif_transposed is False
        assert result.alpha_composited is False

    def test_valid_rgb_png(self, preprocessor: ImagePreprocessor) -> None:
        """Verify RGB PNG is processed cleanly without alpha compositing."""
        image_bytes = create_test_image_bytes(format_name="PNG", size=(250, 250), mode="RGB")
        result = preprocessor.preprocess(image_bytes)

        assert result.original_format == "PNG"
        assert result.original_mode == "RGB"
        assert result.final_mode == "RGB"
        assert result.tensor.shape == (1, 224, 224, 3)
        assert result.alpha_composited is False

    def test_valid_webp(self, preprocessor: ImagePreprocessor) -> None:
        """Verify WEBP format is supported."""
        image_bytes = create_test_image_bytes(format_name="WEBP", size=(400, 400), mode="RGB")
        result = preprocessor.preprocess(image_bytes)

        assert result.original_format == "WEBP"
        assert result.final_mode == "RGB"
        assert result.tensor.shape == (1, 224, 224, 3)

    def test_valid_grayscale_image(self, preprocessor: ImagePreprocessor) -> None:
        """Verify single-channel 8-bit grayscale ('L') image is converted to 3-channel RGB."""
        # 100x100 grayscale image with intensity 128
        img = Image.new("L", (100, 100), color=128)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        result = preprocessor.preprocess(image_bytes)

        assert result.original_mode == "L"
        assert result.final_mode == "RGB"
        assert result.tensor.shape == (1, 224, 224, 3)
        assert result.alpha_composited is False

        # In grayscale converted to RGB, all 3 channels must be equal
        tensor_2d = result.tensor[0]
        channel_r = tensor_2d[:, :, 0]
        # Notice that DenseNet preprocess_input uses different mean/std for each channel:
        # R: (128/255 - 0.485)/0.229 = 0.0740
        # G: (128/255 - 0.456)/0.224 = 0.2052
        # B: (128/255 - 0.406)/0.225 = 0.4265
        # But each individual channel across space must be uniform:
        assert np.allclose(channel_r, channel_r[0, 0])

    def test_valid_rgba_image_with_transparency(self, preprocessor: ImagePreprocessor) -> None:
        """Verify RGBA image triggers deterministic white-background alpha compositing."""
        # RGBA image with fully transparent background and semi-transparent patch
        img = Image.new("RGBA", (100, 100), color=(0, 0, 0, 0))
        # Top half: opaque red (255, 0, 0, 255)
        for x in range(100):
            for y in range(50):
                img.putpixel((x, y), (255, 0, 0, 255))

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        result = preprocessor.preprocess(image_bytes)

        assert result.original_mode == "RGBA"
        assert result.final_mode == "RGB"
        assert result.alpha_composited is True
        assert result.tensor.shape == (1, 224, 224, 3)

    def test_valid_palette_image_with_transparency(self, preprocessor: ImagePreprocessor) -> None:
        """Verify palette ('P') image with transparency metadata composites onto white."""
        rgba = Image.new("RGBA", (50, 50), color=(0, 0, 0, 0))
        rgba.putpixel((0, 0), (255, 0, 0, 255))
        p_img = rgba.quantize()

        buf = io.BytesIO()
        p_img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        result = preprocessor.preprocess(image_bytes)

        assert result.original_mode == "P"
        assert result.final_mode == "RGB"
        assert result.alpha_composited is True
        assert result.tensor.shape == (1, 224, 224, 3)

    def test_valid_palette_image_without_transparency(self, preprocessor: ImagePreprocessor) -> None:
        """Verify palette ('P') image without transparency converts directly without compositing."""
        rgb = Image.new("RGB", (50, 50), color=(100, 150, 200))
        p_img = rgb.quantize()

        buf = io.BytesIO()
        p_img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        result = preprocessor.preprocess(image_bytes)

        assert result.original_mode == "P"
        assert result.final_mode == "RGB"
        assert result.alpha_composited is False
        assert result.tensor.shape == (1, 224, 224, 3)


# ==============================================================================
# 2. Image Transformations & Normalization Behavior
# ==============================================================================


class TestImageTransformations:
    """Verify resizing, EXIF orientation, and color normalization accuracy."""

    @pytest.mark.parametrize("input_size", [(100, 100), (300, 400), (640, 480), (1024, 768), (1200, 1600)])
    def test_exact_224x224_output_dimensions(
        self, preprocessor: ImagePreprocessor, input_size: tuple[int, int]
    ) -> None:
        """Verify every accepted input is resized to exactly 224x224 dimensions."""
        image_bytes = create_test_image_bytes(format_name="PNG", size=input_size)
        result = preprocessor.preprocess(image_bytes)

        assert result.original_width == input_size[0]
        assert result.original_height == input_size[1]
        assert result.processed_width == 224
        assert result.processed_height == 224
        assert result.tensor.shape == (1, 224, 224, 3)

    def test_exif_orientation_correction_applied(self, preprocessor: ImagePreprocessor) -> None:
        """Verify images with non-standard EXIF orientation (e.g. tag 6, 90 deg CW) are transposed."""
        # Create non-square image: 300 wide x 150 high
        # Tag 6 indicates the 0th row represents the visual right column (requires 90 CW rotate)
        image_bytes = create_test_image_bytes(
            format_name="JPEG",
            size=(300, 150),
            exif_orientation=6,
        )

        result = preprocessor.preprocess(image_bytes)

        assert result.exif_transposed is True
        assert result.tensor.shape == (1, 224, 224, 3)

    def test_exif_orientation_normal_not_transposed(self, preprocessor: ImagePreprocessor) -> None:
        """Verify standard EXIF orientation tag 1 (Normal) does not flag unnecessary transposition."""
        image_bytes = create_test_image_bytes(
            format_name="JPEG",
            size=(200, 200),
            exif_orientation=1,
        )
        result = preprocessor.preprocess(image_bytes)
        assert result.exif_transposed is False

    def test_deterministic_white_background_alpha_compositing(
        self, preprocessor: ImagePreprocessor
    ) -> None:
        """Verify transparent regions are composited onto pure solid white (255, 255, 255)."""
        # Create a 224x224 image entirely composed of transparent black (0, 0, 0, 0)
        img = Image.new("RGBA", (224, 224), color=(0, 0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        transparent_bytes = buf.getvalue()

        # Create a 224x224 image entirely composed of opaque white (255, 255, 255)
        white_img = Image.new("RGB", (224, 224), color=(255, 255, 255))
        buf_white = io.BytesIO()
        white_img.save(buf_white, format="PNG")
        white_bytes = buf_white.getvalue()

        result_transparent = preprocessor.preprocess(transparent_bytes)
        result_white = preprocessor.preprocess(white_bytes)

        assert result_transparent.alpha_composited is True
        assert result_white.alpha_composited is False

        # Compositing transparent black over pure white must produce identical output to pure white
        np.testing.assert_allclose(
            result_transparent.tensor,
            result_white.tensor,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Transparent pixels did not composite to pure white (255, 255, 255).",
        )


# ==============================================================================
# 3. Defensive Security & Error Handling
# ==============================================================================


class TestDefensiveValidation:
    """Verify defensive guardrails reject invalid, oversized, or malicious inputs."""

    def test_empty_bytes_raises_empty_image_error(self, preprocessor: ImagePreprocessor) -> None:
        """Verify empty byte sequence raises EmptyImageError."""
        with pytest.raises(EmptyImageError) as exc_info:
            preprocessor.preprocess(b"")

        assert exc_info.value.error_code == "EMPTY_IMAGE"
        assert "empty" in str(exc_info.value).lower()

    def test_non_bytes_input_raises_corrupt_image_error(self, preprocessor: ImagePreprocessor) -> None:
        """Verify non-bytes input raises CorruptImageError."""
        with pytest.raises(CorruptImageError) as exc_info:
            preprocessor.preprocess("not bytes")  # type: ignore

        assert exc_info.value.error_code == "CORRUPT_IMAGE"

    def test_random_garbage_bytes_raises_corrupt_image_error(
        self, preprocessor: ImagePreprocessor
    ) -> None:
        """Verify random binary garbage raises CorruptImageError."""
        garbage = b"PK\x03\x04\x14\x00\x00\x00random-corrupted-binary-data"
        with pytest.raises(CorruptImageError) as exc_info:
            preprocessor.preprocess(garbage)

        assert exc_info.value.error_code == "CORRUPT_IMAGE"

    def test_truncated_header_raises_corrupt_image_error(
        self, preprocessor: ImagePreprocessor
    ) -> None:
        """Verify truncated image header raises CorruptImageError."""
        truncated_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01"
        with pytest.raises(CorruptImageError) as exc_info:
            preprocessor.preprocess(truncated_jpeg)

        assert exc_info.value.error_code == "CORRUPT_IMAGE"

    @pytest.mark.parametrize("unsupported_format", ["BMP", "TIFF", "GIF"])
    def test_unsupported_formats_rejected(
        self, preprocessor: ImagePreprocessor, unsupported_format: str
    ) -> None:
        """Verify unsupported raster formats (BMP, TIFF, GIF) are rejected."""
        img = Image.new("RGB", (50, 50), color=(10, 20, 30))
        buf = io.BytesIO()
        img.save(buf, format=unsupported_format)
        image_bytes = buf.getvalue()

        with pytest.raises(UnsupportedFormatError) as exc_info:
            preprocessor.preprocess(image_bytes)

        assert exc_info.value.error_code == "UNSUPPORTED_FORMAT"
        assert unsupported_format in str(exc_info.value) or unsupported_format in exc_info.value.details.get(
            "detected_format", ""
        )

    def test_svg_format_rejected(self, preprocessor: ImagePreprocessor) -> None:
        """Verify vector XML/SVG format is rejected cleanly."""
        svg_bytes = b"<svg xmlns='http://www.w3.org/2000/svg' width='100' height='100'></svg>"
        with pytest.raises(UnsupportedFormatError) as exc_info:
            preprocessor.preprocess(svg_bytes)

        assert exc_info.value.error_code == "UNSUPPORTED_FORMAT"
        assert "SVG" in str(exc_info.value)

    def test_upload_exceeding_byte_limit_raises_image_too_large(self) -> None:
        """Verify payloads exceeding max_upload_bytes raise ImageTooLargeError."""
        custom_config = ImagePreprocessorConfig(max_upload_bytes=100)
        small_preprocessor = ImagePreprocessor(config=custom_config)

        # Create valid image that exceeds 100 bytes
        image_bytes = create_test_image_bytes(format_name="PNG", size=(50, 50))
        assert len(image_bytes) > 100

        with pytest.raises(ImageTooLargeError) as exc_info:
            small_preprocessor.preprocess(image_bytes)

        assert exc_info.value.error_code == "IMAGE_TOO_LARGE"
        assert exc_info.value.details["max_upload_bytes"] == 100

    def test_decompression_bomb_pixel_limit_exceeded(self) -> None:
        """Verify images with pixel dimensions exceeding max_image_pixels are blocked."""
        # Limit to 10,000 pixels (e.g. 100x100)
        custom_config = ImagePreprocessorConfig(max_image_pixels=10000)
        strict_preprocessor = ImagePreprocessor(config=custom_config)

        # Create 150x150 = 22,500 pixels (exceeds 10,000)
        image_bytes = create_test_image_bytes(format_name="PNG", size=(150, 150))

        with pytest.raises(DecompressionBombError) as exc_info:
            strict_preprocessor.preprocess(image_bytes)

        assert exc_info.value.error_code == "DECOMPRESSION_BOMB"
        assert exc_info.value.details["pixel_count"] == 22500
        assert exc_info.value.details["max_pixels"] == 10000


# ==============================================================================
# 4. DenseNet Preprocessing & Numerical Verification
# ==============================================================================


class TestDenseNetNumericalVerification:
    """Empirically verify TensorFlow DenseNet121 ImageNet normalization formulas."""

    def test_deterministic_output_for_identical_inputs(
        self, preprocessor: ImagePreprocessor
    ) -> None:
        """Verify identical input bytes produce bit-for-bit identical output tensors."""
        image_bytes = create_test_image_bytes(format_name="PNG", size=(300, 300))

        res1 = preprocessor.preprocess(image_bytes)
        res2 = preprocessor.preprocess(image_bytes)

        assert np.array_equal(res1.tensor, res2.tensor)
        assert res1.to_dict() == res2.to_dict()

    def test_tensor_finite_contract(self, preprocessor: ImagePreprocessor) -> None:
        """Verify tensor contains no NaN or infinite values."""
        image_bytes = create_test_image_bytes(format_name="JPEG", size=(224, 224))
        result = preprocessor.preprocess(image_bytes)

        assert np.all(np.isfinite(result.tensor))
        assert not np.any(np.isnan(result.tensor))
        assert not np.any(np.isinf(result.tensor))

    def test_densenet_imagenet_torch_normalization_formula(
        self, preprocessor: ImagePreprocessor
    ) -> None:
        """Verify tensor values match the TensorFlow DenseNet121 ImageNet normalization formula:

        Formula:
            scaled = raw_uint8 / 255.0
            norm_R = (scaled[..., 0] - 0.485) / 0.229
            norm_G = (scaled[..., 1] - 0.456) / 0.224
            norm_B = (scaled[..., 2] - 0.406) / 0.225
        """
        # Create image with known pure colors: R=255, G=128, B=0
        raw_r, raw_g, raw_b = 255, 128, 0
        img = Image.new("RGB", (224, 224), color=(raw_r, raw_g, raw_b))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        result = preprocessor.preprocess(image_bytes)
        tensor = result.tensor[0]  # Shape: (224, 224, 3)

        # Calculate theoretical ground-truth values
        expected_r = (255.0 / 255.0 - 0.485) / 0.229  # (1.0 - 0.485) / 0.229 = 2.2489083
        expected_g = (128.0 / 255.0 - 0.456) / 0.224  # (0.50196 - 0.456) / 0.224 = 0.2051822
        expected_b = (0.0 / 255.0 - 0.406) / 0.225    # (0.0 - 0.406) / 0.225 = -1.8044444

        np.testing.assert_allclose(
            tensor[:, :, 0],
            expected_r,
            rtol=1e-4,
            atol=1e-4,
            err_msg="Red channel normalization failed.",
        )
        np.testing.assert_allclose(
            tensor[:, :, 1],
            expected_g,
            rtol=1e-4,
            atol=1e-4,
            err_msg="Green channel normalization failed.",
        )
        np.testing.assert_allclose(
            tensor[:, :, 2],
            expected_b,
            rtol=1e-4,
            atol=1e-4,
            err_msg="Blue channel normalization failed.",
        )

    def test_comparison_with_official_keras_preprocess_input(
        self, preprocessor: ImagePreprocessor
    ) -> None:
        """Compare output against official tf.keras.applications.densenet.preprocess_input."""
        from tensorflow.keras.applications.densenet import preprocess_input

        # Deterministic pseudo-random 224x224 RGB image
        np.random.seed(42)
        raw_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(raw_array, mode="RGB")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        result = preprocessor.preprocess(image_bytes)

        # Run official function on exact raw array
        official_preprocessed = preprocess_input(raw_array.astype(np.float32))
        expected_tensor = np.expand_dims(official_preprocessed, axis=0)

        np.testing.assert_allclose(
            result.tensor,
            expected_tensor,
            rtol=1e-6,
            atol=1e-6,
            err_msg="ImagePreprocessor tensor diverges from official Keras DenseNet preprocess_input.",
        )


# ==============================================================================
# 5. Singleton & Metadata Contracts
# ==============================================================================


class TestSingletonAndMetadata:
    """Verify singleton accessor and serialization metadata behavior."""

    def test_singleton_accessor(self) -> None:
        """Verify get_image_preprocessor returns a persistent singleton."""
        p1 = get_image_preprocessor()
        p2 = get_image_preprocessor()
        assert p1 is p2

    def test_result_to_dict_metadata(self, preprocessor: ImagePreprocessor) -> None:
        """Verify result.to_dict() returns valid, JSON-serializable telemetry."""
        image_bytes = create_test_image_bytes(format_name="JPEG", size=(300, 200))
        result = preprocessor.preprocess(image_bytes)

        meta = result.to_dict()
        assert meta["original_width"] == 300
        assert meta["original_height"] == 200
        assert meta["processed_width"] == 224
        assert meta["processed_height"] == 224
        assert meta["original_format"] == "JPEG"
        assert meta["original_mode"] == "RGB"
        assert meta["final_mode"] == "RGB"
        assert meta["tensor_shape"] == [1, 224, 224, 3]
        assert meta["tensor_dtype"] == "float32"
        assert "tensor" not in meta  # Must not leak large array
