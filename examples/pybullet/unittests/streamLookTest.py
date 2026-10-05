"""swarmStreamLook: the live video look of a colour frame, the same bytes on every machine, and its argument checks."""
import hashlib
import os
import subprocess
import sys
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

  def test_thread_counts_give_the_same_bytes(self):
    """1, 2 and 4 render threads, each in its own process, give the pinned look and one look of other frames."""
    hashes = set()
    for threads in ("1", "2", "4"):
      env = dict(os.environ, SWARM_RENDER_THREADS=threads)
      lines = subprocess.check_output([sys.executable, __file__, "looks"], env=env, text=True).split()
      self.assertEqual(lines[0], PINNED_SHA256)
      hashes.add(" ".join(lines))
    self.assertEqual(len(hashes), 1)

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


def looks():
  """Print the hash of frame()'s look at quality 70, then of small and odd-sized frames at other qualities."""
  out = np.empty((480, 640, 3), np.float32)
  p.swarmStreamLook(frame(), out, 70)
  print(hashlib.sha256(out.tobytes()).hexdigest())
  for seed, (height, width), quality in ((5, (16, 16), 1), (6, (48, 80), 35), (7, (96, 32), 95), (8, (480, 640), 100)):
    source = frame(seed)[:height, :width].copy()
    out = np.empty_like(source)
    p.swarmStreamLook(source, out, quality)
    print(hashlib.sha256(out.tobytes()).hexdigest())


if __name__ == '__main__':
  if len(sys.argv) > 1 and sys.argv[1] == "looks":
    looks()
  else:
    unittest.main()
