from pathlib import Path

from motion.io.images import save_image
from motion.io.masks import save_mask
from motion.pipeline.manifest import MotionManifest
from motion.pipeline.merge import MergeConfig, MergePipeline


def _write_inputs(tmp_path, circle, square):
    (ia, ma), (ib, mb) = circle, square
    save_image(tmp_path / "a.png", ia)
    save_image(tmp_path / "b.png", ib)
    save_mask(tmp_path / "ma.png", ma)
    save_mask(tmp_path / "mb.png", mb)
    return tmp_path


def test_merge_pair_classical_with_masks(tmp_path, circle, square):
    d = _write_inputs(tmp_path, circle, square)
    pipe = MergePipeline(MergeConfig(backend="classical", n_frames=10, max_side=0))
    manifest = pipe.merge_pair(
        d / "a.png",
        d / "b.png",
        mask_a=str(d / "ma.png"),
        mask_b=str(d / "mb.png"),
        out_video=d / "out.mp4",
    )
    assert Path(manifest.output_video).exists()
    assert manifest.total_frames == 10
    assert manifest.segments[0].backend == "classical"
    # manifest is written next to the video
    assert (d / "out.json").exists()
    assert MotionManifest.load(d / "out.json").total_frames == 10


def test_merge_pair_flow_no_masks(tmp_path, circle, square):
    d = _write_inputs(tmp_path, circle, square)
    pipe = MergePipeline(MergeConfig(backend="flow", n_frames=8, max_side=0, allow_grabcut=False))
    manifest = pipe.merge_pair(d / "a.png", d / "b.png", out_video=d / "flow.mp4")
    assert Path(manifest.output_video).exists()
    assert manifest.total_frames == 8


def test_sequence_of_three(tmp_path, circle, square):
    d = _write_inputs(tmp_path, circle, square)
    # reuse a.png as the third keyframe → chained morph A→B→A
    pipe = MergePipeline(MergeConfig(backend="flow", n_frames=6, max_side=0, allow_grabcut=False))
    manifest = pipe.merge_sequence(
        images=[str(d / "a.png"), str(d / "b.png"), str(d / "a.png")],
        out_video=d / "seq.mp4",
    )
    assert len(manifest.segments) == 2
    # 6 + (6-1) shared-frame-dropped = 11
    assert manifest.total_frames == 11
