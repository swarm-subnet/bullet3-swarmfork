"""Prove byte-frame look identity and the exact 256-value conversion."""
import unittest

import numpy as np

import pybullet as p


class ByteStreamLookTest(unittest.TestCase):
  """Compare the byte entry to the released float arithmetic."""

  def test_every_byte_round_trips(self) -> None:
    """Float32 division and rounded multiplication preserve all 256 bytes."""
    values = np.arange(256, dtype=np.uint8)
    normalised = values.astype(np.float32) / np.float32(255.0)
    recovered = np.rint(np.clip(normalised, 0, 1) * np.float32(255.0))
    np.testing.assert_array_equal(recovered.astype(np.uint8), values)

  def test_rgb_rgba_all_qualities(self) -> None:
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

  def test_invalid_byte_frames(self) -> None:
    """Reject wrong types, noncontiguous bytes and invalid channel counts."""
    out = np.empty((16, 16, 3), np.float32)
    strided = np.zeros((16, 32, 3), np.uint8)[:, ::2]
    for source, error in ((out, p.error), (np.zeros((16, 16, 2), np.uint8), p.error),
                          (strided, ValueError)):
      with self.subTest(dtype=source.dtype, shape=source.shape), self.assertRaises(error):
        p.swarmStreamLookBytes(source, out, 70)

  def test_overlapping_rgb_and_rgba_are_refused(self) -> None:
    """Exact and partial overlap in either direction fails before either buffer changes."""
    for channels in (3, 4):
      for source_offset, out_offset in ((0, 0), (0, 4), (4, 0)):
        with self.subTest(channels=channels, source_offset=source_offset, out_offset=out_offset):
          storage = np.arange(4096, dtype=np.uint8)
          source = np.ndarray((16, 16, channels), np.uint8, storage, offset=source_offset)
          out = np.ndarray((16, 16, 3), np.float32, storage, offset=out_offset)
          before = storage.tobytes()
          with self.assertRaisesRegex(p.error, "must not overlap"):
            p.swarmStreamLookBytes(source, out, 70)
          self.assertEqual(storage.tobytes(), before)

  def test_adjacent_rgb_and_rgba_are_allowed(self) -> None:
    """Buffers touching at either end are disjoint and give the separate-buffer look."""
    for channels in (3, 4):
      source_bytes = 16 * 16 * channels
      out_bytes = 16 * 16 * 3 * 4
      for source_offset, out_offset in ((0, source_bytes), (out_bytes, 0)):
        storage = np.arange(source_bytes + out_bytes, dtype=np.uint8)
        source = np.ndarray((16, 16, channels), np.uint8, storage, offset=source_offset)
        out = np.ndarray((16, 16, 3), np.float32, storage, offset=out_offset)
        expected = np.empty_like(out)
        p.swarmStreamLookBytes(source.copy(), expected, 70)
        p.swarmStreamLookBytes(source, out, 70)
        self.assertEqual(out.tobytes(), expected.tobytes())

  def test_invalid_outputs_and_qualities_are_refused(self) -> None:
    """Read-only, wrong dtype or shape, strided output and bad qualities fail without writes."""
    readonly = np.zeros((16, 16, 3), np.float32)
    readonly.flags.writeable = False
    for channels in (3, 4):
      source = np.zeros((16, 16, channels), np.uint8)
      for out, quality, error in ((readonly, 70, ValueError),
                                  (np.zeros((16, 16, 3), np.float64), 70, p.error),
                                  (np.zeros((16, 16, 4), np.float32), 70, p.error),
                                  (np.zeros((16, 32, 3), np.float32)[:, ::2], 70, ValueError),
                                  (np.zeros((16, 16, 3), np.float32), 0, p.error),
                                  (np.zeros((16, 16, 3), np.float32), 101, p.error)):
        with self.subTest(channels=channels, dtype=out.dtype, shape=out.shape, quality=quality):
          before = out.tobytes()
          with self.assertRaises(error):
            p.swarmStreamLookBytes(source, out, quality)
          self.assertEqual(out.tobytes(), before)


if __name__ == '__main__':
  unittest.main()
