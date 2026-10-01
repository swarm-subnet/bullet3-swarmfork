"""getCameraImage lets go of the GIL while the frame is drawn: another Python thread keeps running through the call, and
the frame's bytes are those of the same call with no other thread about."""
import os
import tempfile
import threading
import time
import unittest
import numpy as np
import pybullet as p

from raycastColourTest import build_world, write_checker_tga

NEEDS_RELEASE = unittest.skipUnless(hasattr(p, "CAMERA_RELEASES_GIL"), "wheel that holds the GIL while it draws")
BIG = 1024  # long enough a frame that a thread kept off the GIL would stall for most of it


def frame():
  """The test scene's rasterised colour frame at BIG x BIG with shadows, as bytes."""
  view = p.computeViewMatrix([3.0, -3.0, 2.5], [0, 0, 0.4], [0, 0, 1])
  proj = p.computeProjectionMatrixFOV(70, 1.0, 0.1, 30.0)
  _, _, rgb, _, _ = p.getCameraImage(BIG, BIG, view, proj, shadow=1, renderer=p.ER_TINY_RENDERER)
  return np.asarray(rgb).tobytes()


@NEEDS_RELEASE
class TestReleaseGil(unittest.TestCase):
  """Draws the colour test scene while a second thread counts."""

  def setUp(self):
    """Connects and builds the scene with its checker texture."""
    p.connect(p.DIRECT)
    self.tmp = tempfile.mkdtemp()
    self.tex_path = os.path.join(self.tmp, 'checker.tga')
    write_checker_tga(self.tex_path)
    build_world(self.tex_path)

  def tearDown(self):
    """Disconnects and removes the temporary texture."""
    p.disconnect()
    os.remove(self.tex_path)
    os.rmdir(self.tmp)

  def test_another_thread_runs_while_the_frame_is_drawn(self):
    """The counting thread's longest pause during each call is a small part of the call, and the bytes match."""
    alone = frame()
    state = {"last": time.perf_counter(), "gap": 0.0}
    stop = threading.Event()

    def count():
      """Notes the longest time between two of its own turns until told to stop."""
      while not stop.is_set():
        now = time.perf_counter()
        state["gap"] = max(state["gap"], now - state["last"])
        state["last"] = now

    worker = threading.Thread(target=count)
    worker.start()
    try:
      for _ in range(3):
        time.sleep(0.02)
        state["gap"] = 0.0
        start = time.perf_counter()
        drawn = frame()
        spent = time.perf_counter() - start
        self.assertEqual(drawn, alone)
        self.assertGreater(spent, 0.04)
        self.assertLess(state["gap"], 0.5 * spent)
    finally:
      stop.set()
      worker.join()


if __name__ == "__main__":
  unittest.main()
