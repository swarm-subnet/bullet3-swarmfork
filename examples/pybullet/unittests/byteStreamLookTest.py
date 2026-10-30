"""Prove byte-frame look identity and the exact 256-value conversion."""
import unittest

import numpy as np
import pybullet as p


class ByteStreamLookTest(unittest.TestCase):
  """Compare the byte entry to the released float arithmetic."""

  def test_every_byte_round_trips(self):
    """Float32 division and rounded multiplication preserve all 256 bytes."""
    values = np.arange(256, dtype=np.uint8)
    normalised = values.astype(np.float32) / np.float32(255.0)
    recovered = np.rint(np.clip(normalised, 0, 1) * np.float32(255.0))
    np.testing.assert_array_equal(recovered.astype(np.uint8), values)

  def test_rgb_rgba_all_qualities(self):
    """RGB and RGBA bytes give every float bit of the float entry."""
    rng = np.random.default_rng(73)
    rgba = rng.integers(0, 256, (48, 80, 4), dtype=np.uint8)
    for source in (rgba, rgba[:, :, :3].copy(), np.zeros_like(rgba), np.full_like(rgba, 255)):
      floats = source[:, :, :3].astype(np.float32) / np.float32(255.0)
      expected = np.empty_like(floats)
      actual = np.empty_like(floats)
      for quality in range(1, 101):
        p.swarmStreamLook(floats, expected, quality)
        p.swarmStreamLookBytes(source, actual, quality)
        self.assertEqual(actual.tobytes(), expected.tobytes())

  def test_invalid_byte_frames(self):
    """Reject wrong types, noncontiguous bytes and invalid channel counts."""
    out = np.empty((16, 16, 3), np.float32)
    for source in (out, np.zeros((16, 16, 2), np.uint8), np.zeros((16, 32, 3), np.uint8)[:, ::2]):
      with self.assertRaises(Exception):
        p.swarmStreamLookBytes(source, out, 70)


if __name__ == '__main__':
  unittest.main()
