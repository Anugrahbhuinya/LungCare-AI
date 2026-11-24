"""Pydantic schemas for image preprocessing metadata."""

from typing import List
from pydantic import BaseModel, Field


class ImagePreprocessingMetadata(BaseModel):
    """Pydantic schema representing the transformation metadata of a preprocessed image."""

    original_width: int = Field(
        ...,
        description="Original image width in pixels before transformation.",
        examples=[1024],
    )
    original_height: int = Field(
        ...,
        description="Original image height in pixels before transformation.",
        examples=[1024],
    )
    processed_width: int = Field(
        default=224,
        description="Standardized output width in pixels.",
        examples=[224],
    )
    processed_height: int = Field(
        default=224,
        description="Standardized output height in pixels.",
        examples=[224],
    )
    original_format: str = Field(
        ...,
        description="Original detected image format (e.g. JPEG, PNG, WEBP).",
        examples=["JPEG"],
    )
    original_mode: str = Field(
        ...,
        description="Original image color mode (e.g. RGB, RGBA, L, P).",
        examples=["RGB"],
    )
    final_mode: str = Field(
        default="RGB",
        description="Normalized output color mode.",
        examples=["RGB"],
    )
    exif_transposed: bool = Field(
        ...,
        description="Whether EXIF orientation correction was applied.",
        examples=[False],
    )
    alpha_composited: bool = Field(
        ...,
        description="Whether alpha transparency was composited onto white background.",
        examples=[False],
    )
    tensor_shape: List[int] = Field(
        ...,
        description="Preprocessed model input tensor dimensions.",
        examples=[[1, 224, 224, 3]],
    )
    tensor_dtype: str = Field(
        ...,
        description="Preprocessed model input tensor data type.",
        examples=["float32"],
    )

