"""Chest X-ray prediction endpoints for LungCare AI.

Provides versioned REST API endpoints for four-class chest radiograph classification
using the pre-trained DenseNet121 transfer learning model.
"""

import logging
import time
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.api.deps import (
    get_gradcam_service_dep,
    get_gradcam_visualizer_dep,
    get_image_preprocessor_dep,
    get_inference_service_dep,
    get_settings_dep,
)
from app.core.config import Settings, get_settings
from app.core.request_context import get_request_id
from app.schemas.error import PREDICTION_ERROR_RESPONSES
from app.schemas.gradcam import ExplanationMetadata, GradCAMExplainResponse
from app.schemas.prediction import PredictionResponse
from app.services.image_preprocessor import (
    CorruptImageError,
    DecompressionBombError,
    EmptyImageError,
    ImagePreprocessor,
    ImageTooLargeError,
    InvalidImageModeError,
    PreprocessedImageResult,
    UnsupportedFormatError,
)
from app.services.inference_service import (
    DEFAULT_CLASS_MAPPING,
    DEFAULT_DISPLAY_NAMES,
    InferenceError,
    InferenceExecutionError,
    InferenceService,
    InvalidModelOutputError,
    ModelUnavailableError,
    PredictionResult,
)
from app.services.gradcam_service import (
    CLASS_DISPLAY_NAMES,
    CLASS_INDICES,
    GradCAMError,
    GradCAMGenerationError,
    GradCAMGradientError,
    GradCAMLayerNotFoundError,
    GradCAMResult,
    GradCAMService,
    InvalidGradCAMInputError,
)
from app.services.gradcam_visualizer import (
    GradCAMVisualizer,
    InvalidVisualizerInputError,
    VisualizerError,
    VisualizerResult,
)

logger = logging.getLogger("lungcare.api.predictions")

router = APIRouter()

__all__ = [
    "router",
    "predict_chest_xray",
    "explain_chest_xray",
    "get_inference_service_dep",
    "get_gradcam_service_dep",
    "get_gradcam_visualizer_dep",
    "get_image_preprocessor_dep",
]


@router.post(
    "/predictions",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    responses=PREDICTION_ERROR_RESPONSES,
    summary="Chest X-Ray Diagnostic Category Prediction",
    description=(
        "Accepts a chest radiograph image (JPEG, PNG, WEBP), executes DenseNet121 inference, "
        "and returns softmax probabilities across four classes (COVID19, NORMAL, PNEUMONIA, TURBERCULOSIS). "
        "Outputs are experimental predictions for research and educational purposes only."
    ),
    tags=["Predictions"],
)
async def predict_chest_xray(
    image: UploadFile = File(..., description="Chest radiograph image file (JPEG, PNG, WEBP)."),
    service: InferenceService = Depends(get_inference_service_dep),
    settings: Settings = Depends(get_settings),
) -> PredictionResponse:
    """Execute four-class DenseNet121 inference on an uploaded chest X-ray image."""
    max_bytes = settings.MAX_IMAGE_UPLOAD_BYTES

    # Bounded read to prevent memory exhaustion from oversized stream attacks
    try:
        content = await image.read(max_bytes + 1)
    except Exception as exc:
        logger.error("Failed to read uploaded image stream: %s", exc)
        raise CorruptImageError("Failed to read uploaded file payload.") from exc
    finally:
        await image.close()

    # Reject uploads exceeding the configured limit before executing decoding
    if len(content) > max_bytes:
        raise ImageTooLargeError(
            f"Uploaded image size ({len(content):,} bytes) exceeds the maximum allowed "
            f"limit of {max_bytes:,} bytes ({max_bytes / (1024 * 1024):.1f} MB)."
        )

    # Reject empty uploads
    if len(content) == 0:
        raise EmptyImageError("Uploaded image byte sequence is empty.")

    # Offload CPU-bound synchronous inference to worker thread pool
    result: PredictionResult = await run_in_threadpool(service.predict, content)
    req_id = get_request_id()
    resp_dict = result.to_dict()
    resp_dict["status"] = "success"
    resp_dict["request_id"] = req_id if req_id != "-" else None
    if resp_dict.get("display_name") is None:
        resp_dict["display_name"] = result.predicted_label
    return PredictionResponse(**resp_dict)


# ==============================================================================
# Grad-CAM Explainability Endpoint (Phase 2.2.7)
# ==============================================================================


def _resolve_target_class_index(target_class_input: Optional[str]) -> Optional[int]:
    """Parse and validate optional target class string (index or class name)."""
    if target_class_input is None:
        return None

    cleaned = str(target_class_input).strip()
    if not cleaned:
        return None

    if cleaned.isdigit() or (cleaned.startswith("-") and cleaned[1:].isdigit()):
        idx = int(cleaned)
        if 0 <= idx <= 3:
            return idx
        raise InvalidGradCAMInputError(
            f"Target class index '{idx}' out of bounds. Must be an integer between 0 and 3."
        )

    normalized = cleaned.upper()
    name_to_index = {
        "COVID19": 0,
        "COVID-19": 0,
        "NORMAL": 1,
        "PNEUMONIA": 2,
        "TURBERCULOSIS": 3,
        "TUBERCULOSIS": 3,
    }
    if normalized in name_to_index:
        return name_to_index[normalized]

    raise InvalidGradCAMInputError(
        f"Unrecognized target class '{target_class_input}'. "
        "Supported classes: ['COVID19', 'NORMAL', 'PNEUMONIA', 'TURBERCULOSIS'] or indices 0-3."
    )


def _execute_explanation(
    content: bytes,
    target_idx: Optional[int],
    blend_alpha: float,
    preprocessor: ImagePreprocessor,
    inference_service: InferenceService,
    gradcam_service: GradCAMService,
    visualizer: GradCAMVisualizer,
) -> Dict[str, Any]:
    """Synchronous pipeline executed in worker threadpool."""
    t_start = time.perf_counter()

    # Step 1: Preprocess raw image bytes
    t_prep_start = time.perf_counter()
    preprocessed: PreprocessedImageResult = preprocessor.preprocess(content)
    t_prep_end = time.perf_counter()
    prep_time_ms = (t_prep_end - t_prep_start) * 1000.0

    # Step 2: Forward classification inference
    pred_result: PredictionResult = inference_service.predict_tensor(
        tensor=preprocessed.tensor,
        original_width=preprocessed.original_width,
        original_height=preprocessed.original_height,
        preprocessing_time_ms=prep_time_ms,
    )

    # Step 3: Determine Grad-CAM target class (default: predicted class)
    if target_idx is not None:
        selected_idx = target_idx
    else:
        name_to_idx = {"COVID19": 0, "NORMAL": 1, "PNEUMONIA": 2, "TURBERCULOSIS": 3}
        selected_idx = name_to_idx.get(pred_result.predicted_class, 0)

    # Step 4: Grad-CAM heatmap generation
    t_cam_start = time.perf_counter()
    gradcam_result: GradCAMResult = gradcam_service.generate_heatmap(
        image_tensor=preprocessed.tensor,
        class_index=selected_idx,
    )
    t_cam_end = time.perf_counter()
    cam_time_ms = (t_cam_end - t_cam_start) * 1000.0

    # Step 5: Visual overlay and base64 serialization
    visualizer_result: VisualizerResult = visualizer.create_overlay(
        original_image=content,
        heatmap=gradcam_result.heatmap,
        alpha=blend_alpha,
    )

    overlay_base64 = visualizer_result.to_base64_data_uri()
    heatmap_base64 = visualizer_result.heatmap_to_base64_data_uri()

    t_end = time.perf_counter()
    total_time_ms = (t_end - t_start) * 1000.0

    selected_class = CLASS_INDICES.get(selected_idx, "UNKNOWN")
    selected_label = CLASS_DISPLAY_NAMES.get(selected_class, selected_class)

    return {
        "predicted_class": pred_result.predicted_class,
        "predicted_label": pred_result.predicted_label,
        "display_name": pred_result.predicted_label,
        "confidence": pred_result.confidence,
        "probabilities": pred_result.probabilities,
        "selected_target_class": selected_class,
        "selected_target_label": selected_label,
        "target_class_index": selected_idx,
        "heatmap_base64": heatmap_base64,
        "overlay_base64": overlay_base64,
        "explanation": {
            "target_layer": gradcam_result.target_layer,
            "differentiation_target": "post-softmax probability S_c in [0.0, 1.0]",
            "status": gradcam_result.status,
            "alpha": blend_alpha,
            "colormap": visualizer_result.colormap,
            "input_width": preprocessed.original_width,
            "input_height": preprocessed.original_height,
            "processed_width": 224,
            "processed_height": 224,
            "inference_time_ms": pred_result.inference_time_ms,
            "gradcam_time_ms": cam_time_ms,
            "total_time_ms": total_time_ms,
        },
        "disclaimer": gradcam_result.disclaimer,
    }


@router.post(
    "/predictions/explain",
    response_model=GradCAMExplainResponse,
    status_code=status.HTTP_200_OK,
    responses=PREDICTION_ERROR_RESPONSES,
    summary="Chest X-Ray Diagnostic Prediction with Grad-CAM Explainability",
    description=(
        "Accepts a chest radiograph image (JPEG, PNG, WEBP), executes DenseNet121 inference, "
        "and computes class-specific Grad-CAM visual activation heatmaps and overlays. "
        "Outputs are experimental visual explanations for research and educational purposes only."
    ),
    tags=["Predictions", "Explainability"],
)
async def explain_chest_xray(
    request: Request,
    image: UploadFile = File(..., description="Chest radiograph image file (JPEG, PNG, WEBP)."),
    target_class: Optional[str] = Form(None, description="Optional target class (0-3 or name). Defaults to predicted class."),
    alpha: Optional[float] = Form(None, description="Optional alpha blend factor in [0.0, 1.0]. Defaults to 0.4."),
    inference_service: InferenceService = Depends(get_inference_service_dep),
    gradcam_service: GradCAMService = Depends(get_gradcam_service_dep),
    visualizer: GradCAMVisualizer = Depends(get_gradcam_visualizer_dep),
    preprocessor: ImagePreprocessor = Depends(get_image_preprocessor_dep),
    settings: Settings = Depends(get_settings),
) -> GradCAMExplainResponse:
    """Execute four-class DenseNet121 inference and generate Grad-CAM visual explanations."""
    max_bytes = settings.MAX_IMAGE_UPLOAD_BYTES

    # Bounded read to prevent memory exhaustion from oversized stream attacks
    try:
        content = await image.read(max_bytes + 1)
    except Exception as exc:
        logger.error("Failed to read uploaded image stream for explainability: %s", exc)
        raise CorruptImageError("Failed to read uploaded file payload.") from exc
    finally:
        await image.close()

    # Reject uploads exceeding the configured limit before executing decoding
    if len(content) > max_bytes:
        raise ImageTooLargeError(
            f"Uploaded image size ({len(content):,} bytes) exceeds the maximum allowed "
            f"limit of {max_bytes:,} bytes ({max_bytes / (1024 * 1024):.1f} MB)."
        )

    # Reject empty uploads
    if len(content) == 0:
        raise EmptyImageError("Uploaded image byte sequence is empty.")

    # Resolve target class parameter (form takes precedence, falls back to query parameter)
    query_target = request.query_params.get("target_class")
    raw_target = target_class if target_class is not None else query_target
    resolved_target_idx = _resolve_target_class_index(raw_target)

    # Resolve alpha parameter (form takes precedence, falls back to query parameter)
    query_alpha = request.query_params.get("alpha")
    if alpha is not None:
        blend_alpha = alpha
    elif query_alpha is not None:
        try:
            blend_alpha = float(query_alpha)
        except (ValueError, TypeError) as exc:
            raise InvalidGradCAMInputError(
                f"Alpha blend ratio must be a valid float in range [0.0, 1.0], got '{query_alpha}'."
            ) from exc
    else:
        blend_alpha = 0.4

    if not (0.0 <= blend_alpha <= 1.0):
        raise InvalidGradCAMInputError(
            f"Alpha blend ratio must be in range [0.0, 1.0], got {blend_alpha}."
        )

    # Offload CPU-bound ML and visualization work to worker thread pool
    result_dict = await run_in_threadpool(
        _execute_explanation,
        content=content,
        target_idx=resolved_target_idx,
        blend_alpha=blend_alpha,
        preprocessor=preprocessor,
        inference_service=inference_service,
        gradcam_service=gradcam_service,
        visualizer=visualizer,
    )
    req_id = get_request_id()
    result_dict["status"] = "success"
    result_dict["request_id"] = req_id if req_id != "-" else None
    return GradCAMExplainResponse(**result_dict)

