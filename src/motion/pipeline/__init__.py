"""End-to-end orchestration (two → N objects) and the JSON manifest contract."""

from motion.pipeline.manifest import (
    DetectionRecord,
    MorphSegment,
    MotionManifest,
    ObjectSpec,
)
from motion.pipeline.merge import MergeConfig, MergePipeline

__all__ = [
    "MotionManifest",
    "ObjectSpec",
    "DetectionRecord",
    "MorphSegment",
    "MergePipeline",
    "MergeConfig",
]
