"""Shared synthetic fixtures — two simple objects with known masks."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

SIZE = 200


@pytest.fixture
def circle():
    img = np.full((SIZE, SIZE, 3), 40, np.uint8)
    cv2.circle(img, (SIZE // 2, SIZE // 2), SIZE // 3, (220, 60, 60), -1)
    mask = np.zeros((SIZE, SIZE), np.uint8)
    cv2.circle(mask, (SIZE // 2, SIZE // 2), SIZE // 3, 255, -1)
    return img, mask > 0


@pytest.fixture
def square():
    img = np.full((SIZE, SIZE, 3), 40, np.uint8)
    s, c = SIZE // 3, SIZE // 2
    cv2.rectangle(img, (c - s, c - s), (c + s, c + s), (60, 90, 230), -1)
    mask = np.zeros((SIZE, SIZE), np.uint8)
    cv2.rectangle(mask, (c - s, c - s), (c + s, c + s), 255, -1)
    return img, mask > 0
