"""swarmStreamLook: the live video look of a colour frame, the same bytes on every machine, and its argument checks."""
import hashlib
import unittest

import numpy as np
import pybullet as p

# The look of frame() at quality 70, the value Swarm Sentinel's own numpy look pins for the same frame.
PINNED_SHA256 = "d02fe9460af74d5061cb6f586a608c41787672f86ecd1cde7121d627bffc1ed1"


def frame(seed=3):
  """A colour frame with smooth shading, hard edges and fine grain, like a daylight render of a park."""
  rng = np.random.default_rng(seed)
  yy, xx = np.mgrid[0:480, 0:640].astype(np.float32)
  colour = np.stack([xx / 640.0, yy / 480.0, 0.5 + 0.4 * np.sign(np.sin(xx / 9.0) * np.cos(yy / 13.0))], axis=-1)
  colour += rng.normal(0.0, 0.02, colour.shape).astype(np.float32)
  return np.clip(colour, 0.0, 1.0).astype(np.float32)


class StreamLookTest(unittest.TestCase):
  """The compiled stream look against its pinned bytes and its input rules."""

  def test_look_matches_the_pinned_bytes(self):
    """A fixed frame's look hashes to the value the reference numpy look gives."""
    out = np.empty((480, 640, 3), np.float32)
    p.swarmStreamLook(frame(), out, 70)
    self.assertEqual(hashlib.sha256(out.tobytes()).hexdigest(), PINNED_SHA256)

  def test_bad_frames_are_refused(self):
    """Wrong types, shapes, sizes off the 16-dot grid and qualities outside 1 to 100 raise instead of drawing."""
    good = np.zeros((32, 32, 3), np.float32)
    for source, out, quality in ((good.astype(np.float64), good.copy(), 70), (good, np.zeros((32, 16, 3), np.float32), 70),
                                 (np.zeros((24, 32, 3), np.float32), np.zeros((24, 32, 3), np.float32), 70),
                                 (good, good.copy(), 0), (good, good.copy(), 101)):
      with self.assertRaises(Exception):
        p.swarmStreamLook(source, out, quality)
    out = np.zeros((32, 32, 3), np.float32)
    out.flags.writeable = False
    with self.assertRaises(Exception):
      p.swarmStreamLook(good, out, 70)


if __name__ == '__main__':
  unittest.main()
