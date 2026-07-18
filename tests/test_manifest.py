from motion.pipeline.manifest import MorphSegment, MotionManifest, ObjectSpec


def test_manifest_roundtrip(tmp_path):
    m = MotionManifest(
        device="cpu",
        fps=30,
        size=(200, 200),
        objects=[ObjectSpec(image="a.png", prompt="cat"), ObjectSpec(image="b.png", prompt="dog")],
        segments=[
            MorphSegment(
                from_index=0,
                to_index=1,
                backend="classical",
                n_frames=48,
                easing="ease_in_out",
                start_frame=0,
            )
        ],
        output_video="out.mp4",
        total_frames=48,
    )
    path = m.save(tmp_path / "manifest.json")
    loaded = MotionManifest.load(path)

    assert loaded.version == m.version
    assert loaded.size == (200, 200)  # tuple survives json roundtrip
    assert loaded.segments[0].backend == "classical"
    assert len(loaded.objects) == 2
