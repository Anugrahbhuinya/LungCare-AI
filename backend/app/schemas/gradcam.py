"""Grad-CAM Visual Explainability Schemas for LungCare AI.

Defines Pydantic response models for class-specific Gradient-weighted
Class Activation Mapping (Grad-CAM) HTTP representations.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.services.gradcam_service import (
    CLASS_DISPLAY_NAMES,
    CLASS_INDICES,
    RESEARCH_DISCLAIMER,
    TARGET_CONV_LAYER,
    GradCAMResult,
)

__all__ = [
    "TARGET_CONV_LAYER",
    "CLASS_INDICES",
    "CLASS_DISPLAY_NAMES",
    "RESEARCH_DISCLAIMER",
    "GradCAMResult",
    "GradCAMResponse",
    "ExplanationMetadata",
    "GradCAMExplainResponse",
]


class ExplanationMetadata(BaseModel):
    """Metadata regarding Grad-CAM explainability graph and visualization parameters."""

    target_layer: str = Field(
        default=TARGET_CONV_LAYER,
        description="Target convolutional layer name in the DenseNet121 backbone.",
        examples=[TARGET_CONV_LAYER],
    )
    differentiation_target: str = Field(
        default="post-softmax probability S_c in [0.0, 1.0]",
        description="Mathematical target evaluated inside tf.GradientTape for backpropagation.",
        examples=["post-softmax probability S_c in [0.0, 1.0]"],
    )
    status: str = Field(
        ...,
        description="Explanation status: 'success' for active heatmaps, 'uninformative' for flat/zero gradients.",
        examples=["success"],
    )
    alpha: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Alpha blending weight applied to the visual overlay.",
        examples=[0.4],
    )
    colormap: str = Field(
        default="jet",
        description="Colormap applied to the normalized activation values.",
        examples=["jet"],
    )
    input_width: int = Field(
        ...,
        description="Width of the original uploaded radiograph in pixels.",
        examples=[1024],
    )
    input_height: int = Field(
        ...,
        description="Height of the original uploaded radiograph in pixels.",
        examples=[1024],
    )
    processed_width: int = Field(
        default=224,
        description="Preprocessed tensor width.",
        examples=[224],
    )
    processed_height: int = Field(
        default=224,
        description="Preprocessed tensor height.",
        examples=[224],
    )
    inference_time_ms: float = Field(
        ...,
        description="Forward inference latency in milliseconds.",
        examples=[45.2],
    )
    gradcam_time_ms: float = Field(
        ...,
        description="Grad-CAM backpropagation and heatmap generation latency in milliseconds.",
        examples=[95.4],
    )
    total_time_ms: float = Field(
        ...,
        description="Total endpoint processing duration in milliseconds.",
        examples=[155.1],
    )


class GradCAMExplainResponse(BaseModel):
    """Structured response schema for POST /api/v1/predictions/explain."""

    status: str = Field(
        default="success",
        description="Status flag indicating a successful response.",
        examples=["success"],
    )
    predicted_class: str = Field(
        ...,
        description="Predicted diagnostic class identifier (e.g., COVID19, NORMAL, PNEUMONIA, TURBERCULOSIS).",
        examples=["PNEUMONIA"],
    )
    predicted_label: str = Field(
        ...,
        description="Human-readable category label for predicted class.",
        examples=["Pneumonia"],
    )
    display_name: str = Field(
        ...,
        description="Sanitized display label corresponding to the predicted class.",
        examples=["Pneumonia"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence probability for the predicted class.",
        examples=[0.9854],
    )
    probabilities: Dict[str, float] = Field(
        ...,
        description="Softmax probability distribution across all four diagnostic categories.",
        examples=[
            {
                "COVID19": 0.0051,
                "NORMAL": 0.0042,
                "PNEUMONIA": 0.9854,
                "TURBERCULOSIS": 0.0053,
            }
        ],
    )
    selected_target_class: str = Field(
        ...,
        description="Class targeted by the Grad-CAM visual explanation.",
        examples=["PNEUMONIA"],
    )
    selected_target_label: str = Field(
        ...,
        description="Human-readable category label for the Grad-CAM target class.",
        examples=["Pneumonia"],
    )
    target_class_index: int = Field(
        ...,
        ge=0,
        le=3,
        description="Zero-indexed classification target (0 to 3).",
        examples=[2],
    )
    heatmap_base64: str = Field(
        ...,
        description="Base64-encoded PNG data URI of the colorized Grad-CAM activation heatmap.",
        examples=["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."],
    )
    overlay_base64: str = Field(
        ...,
        description="Base64-encoded PNG data URI of the blended Grad-CAM overlay on original radiograph.",
        examples=["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."],
    )
    explanation: ExplanationMetadata = Field(
        ...,
        description="Technical explainability and graph execution metadata.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the explanation was generated.",
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Unique request correlation identifier (matches X-Request-ID response header).",
        examples=["f81d4fae-7dec-11d0-a765-00a0c91e6bf6"],
    )
    disclaimer: str = Field(
        default=RESEARCH_DISCLAIMER,
        description="Mandatory medical and research neutrality disclaimer.",
    )


class GradCAMResponse(BaseModel):
    """Pydantic v2 response model for Grad-CAM explainability representations."""

    selected_class: str = Field(
        ...,
        description="Internal model class identifier (e.g. COVID19, NORMAL, PNEUMONIA, TURBERCULOSIS).",
        examples=["PNEUMONIA"],
    )
    selected_label: str = Field(
        ...,
        description="Sanitized display label corresponding to the selected class.",
        examples=["Pneumonia"],
    )
    class_index: int = Field(
        ...,
        ge=0,
        le=3,
        description="Zero-indexed classification target (0 to 3).",
        examples=[2],
    )
    target_layer: str = Field(
        default=TARGET_CONV_LAYER,
        description="Target convolutional feature map layer name in the DenseNet121 architecture.",
        examples=[TARGET_CONV_LAYER],
    )
    heatmap_width: int = Field(
        default=224,
        description="Spatial width of the normalized Grad-CAM activation heatmap.",
        examples=[224],
    )
    heatmap_height: int = Field(
        default=224,
        description="Spatial height of the normalized Grad-CAM activation heatmap.",
        examples=[224],
    )
    original_width: Optional[int] = Field(
        default=None,
        description="Spatial width of original uploaded chest radiograph.",
    )
    original_height: Optional[int] = Field(
        default=None,
        description="Spatial height of original uploaded chest radiograph.",
    )
    status: str = Field(
        ...,
        description="Explanation generation status: 'success' or 'uninformative'.",
        examples=["success"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model output softmax confidence for the selected class.",
        examples=[0.9854],
    )
    probabilities: Dict[str, float] = Field(
        ...,
        description="Softmax probability distribution across all four diagnostic categories.",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Grad-CAM backpropagation and heatmap generation execution duration in milliseconds.",
        examples=[1245.5],
    )
    disclaimer: str = Field(
        default=RESEARCH_DISCLAIMER,
        description="Mandatory medical and research neutrality disclaimer.",
    )
    heatmap_grid: Optional[List[List[float]]] = Field(
        default=None,
        description="Optional 2D grid of float values normalized in [0.0, 1.0].",
    )
