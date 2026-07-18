"""Image / video / mask I/O. Pure CPU (OpenCV + imageio-ffmpeg)."""

from motion.io.images import ensure_rgb, load_image, resize_to_match, save_image
from motion.io.masks import load_mask, overlay_mask, save_mask
from motion.io.video import read_video_frames, write_video

__all__ = [
    "load_image",
    "save_image",
    "ensure_rgb",
    "resize_to_match",
    "load_mask",
    "save_mask",
    "overlay_mask",
    "write_video",
    "read_video_frames",
]
