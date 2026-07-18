from motion.device import describe, device_info, resolve_device


def test_resolve_device_valid():
    assert resolve_device() in ("cuda", "mps", "cpu")


def test_prefer_cpu():
    assert resolve_device("cpu") == "cpu"


def test_describe_is_string():
    assert isinstance(describe(), str)
    assert "resolved device" in describe()


def test_device_info_cached():
    assert device_info() is device_info()
