"""Pydantic validation schemas for model prediction responses.

Defines the medically neutral, structured data contract for DenseNet121 research
predictions. The contract strictly represents:
- Predicted model class and human-readable category label.
- Model output confidence and softmax probability distribution.
- Model provenance metadata and input/output dimensions.
- Monotonic latency telemetry (CPU baseline: ~35-65 ms).
- Mandatory research and educational disclaimer.

IMPORTANT MEDICAL SAFETY DIRECTIVE:
Model outputs represent statistical pattern associations from research datasets.
They do not constitute a confirmed clinical diagnosis, severity assessment, or
treatment plan.
"""

from datetime import datetime, timezone
from typing import Dict, Optional
from pydantic import BaseModel, Field

RESEARCH_DISCLAIMER: str = (
    "This output is an experimental AI model prediction for research and educational purposes. "
    "It is not a medical diagnosis and must not be used as a substitute for professional medical evaluation."
)


class PredictionResponse(BaseModel):
    """Schema representing the four-class chest X-ray research prediction response."""

    status: str = Field(
        default="success",
        description="Status flag indicating a successful response.",
        examples=["success"],
    )
    predicted_class: str = Field(
        ...,
        description="Internal machine-readable class identifier (e.g. PNEUMONIA, TURBERCULOSIS).",
        examples=["PNEUMONIA"],
    )
    predicted_label: str = Field(
        ...,
        description="Human-readable research category label corresponding to the predicted class.",
        examples=["Pneumonia"],
    )
    display_name: Optional[str] = Field(
        default=None,
        description="Sanitized display label corresponding to the predicted class.",
        examples=["Pneumonia"],
    )
    confidence: float = Field(
        ...,
        description="Confidence probability for the predicted class in range [0.0, 1.0].",
        examples=[0.9854],
    )
    probabilities: Dict[str, float] = Field(
        ...,
        description="Softmax probability distribution across all four research model classes.",
        examples=[
            {
                "COVID19": 0.0051,
                "NORMAL": 0.0042,
                "PNEUMONIA": 0.9854,
                "TURBERCULOSIS": 0.0053,
            }
        ],
    )
    model_name: str = Field(
        ...,
        description="Name of the deployed deep learning model artifact.",
        examples=["LungCare DenseNet121 Weighted"],
    )
    model_architecture: str = Field(
        default="DenseNet121",
        description="Underlying convolutional neural network architecture.",
        examples=["DenseNet121"],
    )
    input_width: int = Field(
        ...,
        description="Original width of the uploaded image in pixels.",
        examples=[1024],
    )
    input_height: int = Field(
        ...,
        description="Original height of the uploaded image in pixels.",
        examples=[1024],
    )
    processed_width: int = Field(
        default=224,
        description="Standardized model input width.",
        examples=[224],
    )
    processed_height: int = Field(
        default=224,
        description="Standardized model input height.",
        examples=[224],
    )
    inference_time_ms: float = Field(
        ...,
        description=(
            "Duration of the forward neural network inference pass in milliseconds. "
            "Typical CPU baseline is approximately 35-65 ms per synthetic image on modern x86_64 hardware."
        ),
        examples=[45.2],
    )
    preprocessing_time_ms: float = Field(
        default=0.0,
        description="Duration of image validation, normalization, and resizing in milliseconds.",
        examples=[14.2],
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the prediction was generated.",
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Unique request correlation identifier (matches X-Request-ID response header).",
        examples=["f81d4fae-7dec-11d0-a765-00a0c91e6bf6"],
    )
    disclaimer: str = Field(
        default=RESEARCH_DISCLAIMER,
        description="Mandatory experimental research and educational disclaimer.",
    )
