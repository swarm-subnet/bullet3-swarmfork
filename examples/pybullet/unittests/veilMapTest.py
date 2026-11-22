"""ER_SWARM_VEIL_MAP: where a far cut-out such as a chain-link fence is a veil covering less than half of a pixel, the
mask names the body seen through it, in daylight colour and thermal frames alike; without the flag, under a veil
covering most of the pixel and at a near fence the cut-out keeps its own id, and the colour bytes never change."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pybullet as p

from daylightTest import DAYLIGHT, NOON, PICTURE, RAYCAST, SKY, chain_link_card

SIZE = 96
THERMAL = getattr(p, "ER_SWARM_THERMAL", 0)
VEIL_MAP = getattr(p, "ER_SWARM_VEIL_MAP", 0)
COLOUR = PICTURE | DAYLIGHT
THERMAL_FRAME = RAYCAST | THERMAL | getattr(p, "ER_ALPHA_CUTOUT", 0)
FAR = ((0, -25, 10), (0, 0, 10))
NEAR = ((0, -0.5, 10), (0, 0, 10))
BEHIND = slice(44, 52)


def scene(folder, wire=3):
  """A chain-link card with lines `wire` texels thick and a 6 m red box standing 2 m behind it; returns (card, box)."""
  card = chain_link_card(folder, wire)
  vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[3, 3, 3], rgbaColor=[0.8, 0.1, 0.1, 1], specularColor=[0, 0, 0])
  return card, p.createMultiBody(0, -1, vis, basePosition=[0, 5, 10])


def render(eye, flags):
  """(rgb, seg) of one frame from `eye`, with the thermal arguments passed whatever the flags."""
  view = p.computeViewMatrix(list(eye[0]), list(eye[1]), [0, 0, 1])
  proj = p.computeProjectionMatrixFOV(90, 1.0, 0.1, 100.0)
  _, _, rgb, _, seg = p.getCameraImage(SIZE, SIZE, view, proj, lightDirection=NOON, renderer=p.ER_TINY_RENDERER,
                                       flags=flags, airTemperature=20.0, skyTemperature=-20.0, thermalSeed=1)
  return np.asarray(rgb, dtype=np.uint8).reshape(SIZE, SIZE, 4), np.asarray(seg).reshape(SIZE, SIZE)


@unittest.skipUnless(RAYCAST and DAYLIGHT and SKY and THERMAL and VEIL_MAP, "wheel without ER_SWARM_VEIL_MAP")
class TestVeilMap(unittest.TestCase):
  """A 6 m box standing 2 m behind a chain-link card, seen from far away and from up close."""

  def setUp(self):
    """A DIRECT client and a temporary folder."""
    p.connect(p.DIRECT)
    self.folder = tempfile.mkdtemp(prefix="veil_map_")

  def tearDown(self):
    """Drops the client and the folder."""
    p.disconnect()
    shutil.rmtree(self.folder, ignore_errors=True)

  def test_colour_frame_names_the_body_behind_a_far_fence(self):
    """In a daylight colour frame every pixel of the box behind a far 18 % fence is named as the box."""
    card, box = scene(self.folder)
    seg = render(FAR, COLOUR | VEIL_MAP)[1][BEHIND, BEHIND]
    self.assertTrue(np.all(seg == box), np.unique(seg))
    self.assertFalse(np.any(seg == card))

  def test_thermal_frame_names_the_body_behind_a_far_fence(self):
    """In a thermal frame every pixel of the box behind a far 18 % fence is named as the box."""
    _card, box = scene(self.folder)
    seg = render(FAR, THERMAL_FRAME | VEIL_MAP)[1][BEHIND, BEHIND]
    self.assertTrue(np.all(seg == box), np.unique(seg))

  def test_without_the_flag_the_fence_keeps_its_id(self):
    """Without the flag the far fence names its own pixels, in colour and thermal frames."""
    card, _box = scene(self.folder)
    for flags in (COLOUR, THERMAL_FRAME):
      self.assertTrue(np.all(render(FAR, flags)[1][BEHIND, BEHIND] == card))

  def test_veil_covering_most_of_a_pixel_keeps_its_id(self):
    """A far card that is 61 % wire still names its own pixels, in colour and thermal frames."""
    card, _box = scene(self.folder, wire=12)
    for flags in (COLOUR, THERMAL_FRAME):
      self.assertTrue(np.all(render(FAR, flags | VEIL_MAP)[1][BEHIND, BEHIND] == card))

  def test_near_fence_keeps_its_wire_and_shows_the_box_through_its_holes(self):
    """Up close the wire is sharp: some pixels name the card and the holes between them name the box."""
    card, box = scene(self.folder)
    seg = render(NEAR, COLOUR | VEIL_MAP)[1]
    self.assertGreater(int((seg == card).sum()), 0)
    self.assertGreater(int((seg == box).sum()), int((seg == card).sum()))

  def test_colour_bytes_are_the_same_with_and_without_the_flag(self):
    """The flag changes the mask only: colour frames far and near and the thermal frame keep every byte."""
    scene(self.folder)
    for eye, flags in ((FAR, COLOUR), (NEAR, COLOUR), (FAR, THERMAL_FRAME)):
      self.assertTrue(np.array_equal(render(eye, flags)[0], render(eye, flags | VEIL_MAP)[0]))

  def test_threads_give_the_same_mask(self):
    """The far colour and thermal masks are identical at 1, 2 and 4 render threads, each count in its own process."""
    masks = []
    for threads in ("1", "2", "4"):
      out = os.path.join(self.folder, "mask_%s.npy" % threads)
      env = dict(os.environ, SWARM_RENDER_THREADS=threads)
      subprocess.check_call([sys.executable, os.path.abspath(__file__), "--mask", out], env=env,
                            cwd=os.path.dirname(os.path.abspath(__file__)))
      masks.append(np.load(out))
    self.assertTrue(np.array_equal(masks[0], masks[1]))
    self.assertTrue(np.array_equal(masks[0], masks[2]))


def save_masks(path):
  """Renders the far colour and thermal masks with the flag and saves them, for the thread test's child processes."""
  p.connect(p.DIRECT)
  folder = tempfile.mkdtemp(prefix="veil_map_child_")
  scene(folder)
  np.save(path, np.stack([render(FAR, COLOUR | VEIL_MAP)[1], render(FAR, THERMAL_FRAME | VEIL_MAP)[1]]))
  p.disconnect()
  shutil.rmtree(folder, ignore_errors=True)


if __name__ == "__main__":
  if len(sys.argv) == 3 and sys.argv[1] == "--mask":
    save_masks(sys.argv[2])
  else:
    unittest.main()
