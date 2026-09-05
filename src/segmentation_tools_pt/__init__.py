"""Modules:

- `segmentation_tools_pt.onnx_export`: Export a :class:`~segmentation_tools_pt.model_wrapper.TiledInferenceModelONNX` to ONNX format.
- `segmentation_tools_pt.tiling_core`: Shared data contracts, enumerations, and image-padding utilities used throughout the tiling-stitching pipeline. All heavy lifting (tiling, stitching, model wrapping, ONNX export) lives in the sibling modules. This module provides the shared types and padding helpers (requires PyTorch for tensor ops)."""

__version__ = "0.0.1"
